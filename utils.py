from __future__ import print_function, division

from collections import OrderedDict
from copy import copy
from mpi4py import MPI
import numpy as np

import dask.array as da
from nbodykit import CurrentMPIComm
from nbodykit import logging
from nbodykit.mpirng import MPIRandomState
from nbodykit.source.mesh.bigfile import BigFileMesh
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.source.catalog.array import ArrayCatalog
from pmesh.pm import RealField, ComplexField

from pm_utils import ltoc_index_arr, cgetitem_index_arr


def readout_mesh_at_cat_pos(mesh=None, cat=None, readout_window='cic'):
    """
    Readout field at catalog positions.

    Parameters
    ----------
    mesh : MeshSource object
    cat : CatalogSource object
    """
    layout = mesh.pm.decompose(cat['Position'], smoothing=readout_window)
    mesh_at_cat_pos = mesh.compute(mode='real').readout(
        cat['Position'], resampler=readout_window, layout=layout)

    return mesh_at_cat_pos

def catalog_persist(cat, columns=None):
    """
    Return a CatalogSource, where the selected columns are
    computed and persist in memory.
    """
    if columns is None:
        columns = cat.columns

    r = {}
    for key in columns:
        if key in cat.columns:
            r[key] = cat[key]

    r = da.compute(r)[0] # particularity of dask

    c = ArrayCatalog(r, comm=cat.comm)
    c.attrs.update(cat.attrs)

    return c


def get_cstat(data, statistic, comm=None):
    """
    Compute a collective statistic across all ranks and return as float.
    Must be called by all ranks.
    """
    #if isinstance(data, MeshSource):
    #    data = data.compute().value
    if isinstance(data, RealField) or isinstance(data, ComplexField):
        data = data.value
    else:
        assert type(data) == np.ndarray
    if comm is None:
        from nbodykit import CurrentMPIComm
        comm = CurrentMPIComm.get()

    if statistic == 'min':
        return comm.allreduce(data.min(), op=MPI.MIN)
    elif statistic == 'max':
        return comm.allreduce(data.max(), op=MPI.MAX)
    elif statistic == 'mean':
        # compute the mean
        csum = comm.allreduce(data.sum(), op=MPI.SUM)
        csize = comm.allreduce(data.size, op=MPI.SUM)
        return csum / float(csize)
    elif statistic == 'rms':
        rsum = comm.allreduce((data**2).sum())
        csize = comm.allreduce(data.size)
        rms = (rsum / float(csize))**0.5
        return rms
    else:
        raise Exception("Invalid statistic %s" % statistic)

def get_cmean(data, comm=None):
    return get_cstat(data, 'mean', comm=comm)

def get_cstats_string(data, comm=None):
    """
    Get collective statistics (rms, min, mean, max) of data and return as string.
    Must be called by all ranks.
    """
    stat_names = ['rms', 'min', 'mean', 'max']
    cstats = OrderedDict()
    iscomplex = False
    for s in stat_names:
        cstats[s] = get_cstat(data, s)
        if np.iscomplex(cstats[s]):
            iscomplex = True

    if iscomplex:
        return 'rms, min, mean, max: %s %s %s %s' % (str(
            cstats['rms']), str(cstats['min']), str(
                cstats['mean']), str(cstats['max']))
    else:
        return 'rms, min, mean, max: %g %g %g %g' % (
            cstats['rms'], cstats['min'], cstats['mean'], cstats['max'])


def print_cstats(data, prefix="", logger=None, comm=None):
    """
    Must be called by all ranks.
    """
    if comm is None:
        comm = CurrentMPIComm.get()
    if logger is None:
        logger = logging.getLogger("utils")
    cstats = get_cstats_string(data, comm)
    if comm.rank == 0:
        logger.info('%s%s' % (prefix, cstats))
        
        print('%s%s' % (prefix, cstats))


class CosmoModel:
    # TODO: use namedtuple
    def __init__(self,
                 Om_L=None,
                 Om_m=None,
                 Om_K=None,
                 Om_r=None,
                 h0=None,
                 n_s=None,
                 m_nu=None,
                 fnl=None):
        self.Om_L = Om_L
        self.Om_m = Om_m
        self.Om_K = Om_K
        self.Om_r = Om_r
        self.h0 = h0
        self.n_s = n_s
        self.m_nu = m_nu
        self.fnl = fnl

def linear_rescale_fac(current_scale_factor,
                       desired_scale_factor,
                       cosmo_params=None):
    if desired_scale_factor is None or current_scale_factor is None:
        raise Exception("scale factors must be not None")
    if desired_scale_factor > 1.0 or current_scale_factor > 1.0:
        raise Exception("scale factors must be <=1")

    if desired_scale_factor == current_scale_factor:
        rescalefac = 1.0
    else:
        # Factor to linearly rescale delta to desired redshift
        assert (cosmo_params is not None)
        cosmo = CosmoModel(**cosmo_params)
        calc_Da = generate_calc_Da(cosmo=cosmo, verbose=False)
        rescalefac = calc_Da(desired_scale_factor) / calc_Da(
            current_scale_factor)
        #del cosmo
    return rescalefac



def generate_calc_Da(test_plot=False,
                     N_integration=10000,
                     cosmo=None,
                     verbose=True):
    """
    Return the function calc_Da, which takes a as argument
    and returns D(a).
    """
    if verbose:
        print("Compute D(a)")
    a = np.linspace(1.0e-4, 1.0, N_integration)
    H_over_H0 = np.sqrt(cosmo.Om_r / a**4 + cosmo.Om_m / a**3 + cosmo.Om_L +
                        cosmo.Om_K / a**2)
    Da = np.zeros(a.shape)
    # compute the integral
    for imax, aa in enumerate(a):
        Da[imax] = np.trapz(1.0 / (a[:imax + 1] * H_over_H0[:imax + 1])**3,
                            a[:imax + 1])
    # Prefactors
    Da = Da * 5. / 2. * cosmo.Om_m * H_over_H0
    if verbose:
        print("Got D(a)")

    def calc_Da(aeval):
        return np.interp(aeval, a, Da)

    return calc_Da


def calc_f_log_growth_rate(a=None, calc_Da=None, cosmo=None, do_test=False):
    """Calculate f=dlnD/dlna"""
    # derived from formula for D
    assert cosmo.Om_K == 0.
    H_over_H0 = np.sqrt(cosmo.Om_r / a**4 + cosmo.Om_m / a**3 + cosmo.Om_L +
                        cosmo.Om_K / a**2)
    f_analytical = 1. / (a * H_over_H0)**2 * (-2. * cosmo.Om_r / a**2 -
                                              3. / 2. * cosmo.Om_m / a +
                                              5. / 2. * cosmo.Om_m / calc_Da(a))

    if do_test:
        # compute numerically and compare
        avec = np.linspace(1.0e-4, 1.0, 1000)
        d_D_d_a = np.interp(a, 0.5 * (avec[1:] + avec[:-1]),
                            np.diff(calc_Da(avec)) / np.diff(avec))
        f_numerical = a / calc_Da(a) * d_D_d_a
        print('f_numerical=%g' % f_numerical)
        print('f_numerical/f_analytical-1 = %g' %
              (f_numerical / f_analytical - 1.))
        assert np.isclose(f_numerical, f_analytical, rtol=1e-4)

    return f_analytical


def read_delta_from_rho_bigfile(fname, verbose=False):
    """Compute delta from input file."""
    # Compute delta from input file containing rho(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    if verbose:
        print('cmean before getting delta: ', cmean)
    # compute delta = rho/rhobar - 1
    outfield = outfield/cmean - 1.0
    return FieldMesh(outfield)

def read_delta_from_1plusdelta_bigfile(fname, verbose=False):
    """Compute delta from input file."""
    # Compute delta from input file containing 1+delta(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    if verbose:
        print('cmean before getting delta: ', cmean)
        print('subtracting 1')
    # compute delta
    outfield = outfield - 1.0
    return FieldMesh(outfield)

def read_delta_from_2plusdelta_bigfile(fname, verbose=False):
    """Compute delta from input file."""
    # Compute delta from input file containing 2+delta(x).
    outfield = BigFileMesh(fname, 'Field').compute(mode='real')
    cmean = outfield.cmean()
    if verbose:
        print('cmean before getting delta: ', cmean)
        print('subtracting 2')
    # compute delta
    outfield = outfield - 2.0
    return FieldMesh(outfield)



def read_vel_from_bigfile(fname):
    """Read rho in the file, don't divide by mean or subtract mean."""
    return BigFileMesh(fname, 'Field')


def calc_divergence_of_3_meshs(meshsource_tuple, prefactor=1.0):
    """
    Compute divergence of 3 MeshSource objects.

    Parameters
    ----------
    meshsource_tuple : 3-tuple of MeshSource objects
    """
    out_field = None
    for direction in [0,1,2]:
        # copy so we don't modify the input
        cfield = meshsource_tuple[direction].compute(mode='complex').copy()

        def derivative_function(k, v, d=direction):
            return prefactor * k[d] * 1j * v

        # i k_d field_d
        if out_field is None:
            out_field = cfield.apply(derivative_function)
        else:
            out_field += cfield.apply(derivative_function)

        del cfield
        
    return FieldMesh(out_field.c2r())


def get_displacement_from_density_rfield(in_density_rfield,
                                         in_density_cfield=None,
                                         component=None,
                                         Psi_type=None,
                                         smoothing=None,
                                         smoothing_Psi3LPT=None,
                                         prefac_Psi_1storder=1.0,
                                         prefac_Psi_2ndorder=1.0,
                                         prefac_Psi_3rdorder=1.0,
                                         RSD=False,
                                         RSD_line_of_sight=None,
                                         RSD_f_log_growth=None):
    """
    Given density delta(x) in real space, compute Zeldovich displacemnt Psi_component(x)
    given by Psi_component(\vk) = k_component / k^2 * W(k) * delta(\vk),
    where W(k) is smoothing window.

    For Psi_type='Zeldovich' compute 1st order displacement.
    For Psi_type='2LPT' compute 1st plus 2nd order displacement.
    etc

    Supply either in_density_rfield or in_density_cfield.

    Multiply 1st order displacement by prefac_Psi_1storder, 2nd order by 
    prefac_Psi_2ndorder, etc. Use this for getting time derivative of Psi.


    Follow http://rainwoodman.github.io/pmesh/intro.html.

    Parameters
    ----------
    RSD : boolean
        If True, include RSD by displacing by \vecPsi(q)+f (\e_LOS.\vecPsi(q)) \e_LOS, 
        where \ve_LOS is unit vector in line of sight direction.

    RSD_line_of_sight : array_like, (3,)
        Line of sight direction, e.g. [0,0,1] for z axis.
    """
    assert (component in [0, 1, 2])
    assert Psi_type in ['Zeldovich', '2LPT', '-2LPT']

    
    comm = CurrentMPIComm.get()

    if in_density_cfield is None:
        # copy so we don't do any in-place changes by accident
        density_rfield = in_density_rfield.copy()
        density_cfield = density_rfield.r2c()
    else:
        assert in_density_rfield is None
        density_cfield = in_density_cfield.copy()


    if Psi_type in ['Zeldovich', '2LPT', '-2LPT']:

        # get zeldovich displacement in direction given by component

        def potential_transfer_function(k, v):
            k2 = sum(ki**2 for ki in k)
            return np.where(k2 == 0.0, 0 * v, v / (k2))

        # get potential pot = delta/k^2
        #pot_k = density_rfield.r2c().apply(potential_transfer_function)
        pot_k = density_cfield.apply(potential_transfer_function)
        #print("pot_k head:\n", pot_k[:2,:2,:2])

        # apply smoothing
        if smoothing is not None:
            pot_k = smoothen_cfield(pot_k, **smoothing)

            #print("pot_k head2:\n", pot_k[:2,:2,:2])

        # get zeldovich displacement
        def force_transfer_function(k, v, d=component):
            # MS: not sure if we want a factor of -1 here.
            return k[d] * 1j * v

        Psi_component_rfield = pot_k.apply(force_transfer_function).c2r()

        if RSD:
            # Add linear RSD displacement f (\e_LOS.\vecPsi^(1)(q)) \e_LOS.
            assert RSD_f_log_growth is not None
            if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                # If [0,0,1] simply shift by Psi_z along z axis. Similarly in the other cases.
                if RSD_line_of_sight[component] == 0:
                    # nothing to do in this direction
                    pass
                elif RSD_line_of_sight[component] == 1:
                    # add f Psi_component(q)
                    Psi_component_rfield += RSD_f_log_growth * Psi_component_rfield
                    if comm.rank == 0:
                        print('%d: Added RSD in direction %d' %
                              (comm.rank, component))
            else:
                # Need to compute (\e_LOS.\vecPsi(q)) which requires all Psi components.
                raise Exception('RSD_line_of_sight %s not implemented' %
                                str(RSD_line_of_sight))

        Psi_component_rfield *= prefac_Psi_1storder


        # if comm.rank == 0:
        #     print('mean, rms, max Psi^{1}_%d: %g, %g, %g' % (
        #         component, np.mean(Psi_component_rfield), 
        #         np.mean(Psi_component_rfield**2)**0.5,
        #         np.max(Psi_component_rfield)))



        if Psi_type in ['2LPT', '-2LPT']:

            # add 2nd order Psi on top of Zeldovich

            if in_density_rfield is not None:
                in_density_fieldmesh = FieldMesh(in_density_rfield)
            else:
                in_density_fieldmesh = FieldMesh(in_density_cfield)

            # compute G2
            G2_cfield = calc_quadratic_field(
                base_field_mesh=in_density_fieldmesh,
                quadfield='tidal_G2',
                smoothing_of_base_field=smoothing).compute(mode='complex')

            # compute Psi_2ndorder = -3/14 ik/k^2 G2(k). checked sign: improves rcc with deltaNL
            # if we use -3/14, but get worse rcc when using +3/14.
            Psi_2ndorder_rfield = -3. / 14. * (
                G2_cfield.apply(potential_transfer_function).apply(
                    force_transfer_function).c2r())
            del G2_cfield


            if Psi_type == '-2LPT':
                # this is just to test sign
                Psi_2ndorder_rfield *= -1.0


            if RSD:
                # Add 2nd order RSD displacement 2*f*(\e_LOS.\vecPsi^(2)(q)) \e_LOS.
                # Notice factor of 2 b/c \dot\psi enters for RSD.
                if RSD_line_of_sight in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
                    # If [0,0,1] simply shift by Psi_z along z axis. Similarly in the other cases.
                    if RSD_line_of_sight[component] == 0:
                        # nothing to do in this direction
                        pass
                    elif RSD_line_of_sight[component] == 1:
                        # add 2 f Psi^{(2)}_component(q)
                        Psi_2ndorder_rfield += (
                            2.0 * RSD_f_log_growth * Psi_2ndorder_rfield)
                        if comm.rank == 0:
                            print('%d: Added 2nd order RSD in direction %d' %
                                  (comm.rank, component))
                else:
                    # Need to compute (\e_LOS.\vecPsi(q)) which requires all Psi components.
                    raise Exception('RSD_line_of_sight %s not implemented' %
                                    str(RSD_line_of_sight))


            # if comm.rank == 0:
            #     print('mean, rms, max Psi^{2}_%d: %g, %g, %g' % (
            #         component, np.mean(Psi_2ndorder_rfield), 
            #         np.mean(Psi_2ndorder_rfield**2)**0.5,
            #         np.max(Psi_2ndorder_rfield)))

            Psi_2ndorder_rfield *= prefac_Psi_2ndorder

            # add 2nd order to Zeldoivhc displacement
            Psi_component_rfield += Psi_2ndorder_rfield
            del Psi_2ndorder_rfield




    return Psi_component_rfield



def mass_weighted_paint_cat_to_delta(cat,
                                    weight=None,
                                    Nmesh=None,
                                    to_mesh_kwargs={
                                        'window': 'cic',
                                        'compensated': False,
                                        'interlaced': False
                                    },
                                    set_mean=0,
                                    verbose=False):
    """Paint mass-weighted halo density to delta, summing contributions so
    field gets larger if more particles are in a cell.
    """
    # get rho
    delta, attrs = weighted_paint_cat_to_delta(cat,
                                weight=weight,
                                weighted_paint_mode='sum',
                                normalize=False,
                                Nmesh=Nmesh,
                                to_mesh_kwargs=to_mesh_kwargs,
                                set_mean=None,
                                verbose=verbose)
    cmean = get_cmean(delta)
    #cmean = delta.cmean()
    if verbose:
        print('mean0:', cmean)
    if np.abs(cmean<1e-5):
        print('WARNING: dividing by small number when dividing by mean')
    delta /= cmean
    if verbose:
        print('mean1:', get_cmean(delta))
    delta -= get_cmean(delta)
    delta += set_mean
    if verbose:
        print('mean2:', get_cmean(delta))
    if weight is None:
        print_cstats(delta, prefix='delta: ')
    else:
        print_cstats(delta, prefix='mass weighted delta: ')

    return delta, attrs



def weighted_paint_cat_to_delta(cat,
                                weight=None,
                                weighted_paint_mode=None,
                                normalize=True,
                                Nmesh=None,
                                to_mesh_kwargs={
                                    'window': 'cic',
                                    'compensated': False,
                                    'interlaced': False
                                },
                                set_mean=None,
                                verbose=True):
    """
    - weighted_paint_mode='sum': In each cell, sum up the weight of all particles in the cell.
        So this gets larger if there are more particles in a cell. 
    - weighted_paint_mode='avg': In each cell, sum up the weight of all particles in the cell
        and divide by the number of contributions. This does not increase if there are more
        particles in a cell with the same weight.

    Note: 
    In nbodykit nomenclature this is called 'value' instead of 'weight', but only implements our 'sum' not 'avg' mode (it seems).

    Note:
    mass_weighted_paint_cat_to_delta below is a bit cleaner so better use that.
    """

    #print('MYDBG to_mesh_kwargs:', to_mesh_kwargs)

    if weighted_paint_mode not in ['sum', 'avg']:
        raise Exception("Invalid weighted_paint_mode %s" % weighted_paint_mode)

    #if (normalize == True) and (weight is not None):
    #    raise Exception('Do not use normalize with weights -- normalize yourself.')

    assert 'value' not in to_mesh_kwargs.keys()

    # We want to sum up weight. Use value not weight for this b/c each ptlce should contribute equally. Later we divide by number of contributions if mode='avg'.
    if weight is not None:
        meshsource = cat.to_mesh(Nmesh=Nmesh, value=weight, **to_mesh_kwargs)
    else:
        # no weight so assume each ptcle has weight 1
        meshsource = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs)
    meshsource.attrs['weighted_paint_mode'] = weighted_paint_mode

    # get outfield = 1+delta
    #outfield = meshsource.paint(mode='real')
    # Paint. If normalize=True, outfield = 1+delta; if normalize=False: outfield=rho
    if to_mesh_kwargs.get('compensated', False):
        # have to use compute to compensate window; to_real_field does not compensate.
        if normalize:
            # compute 1+delta, compensate window
            outfield = meshsource.compute()
        else:
            raise Exception('Not implemented: compensated and not normalized')
    else:
        # no window compensation
        outfield = meshsource.to_real_field(normalize=normalize)


    if weighted_paint_mode == 'avg':
        # count contributions per cell (no value or weight).
        # outfield_count = 1+delta_unweighted = number of contributions per cell
        # (or rho_unweighted if normalize=False)
        #outfield_count = cat.to_mesh(Nmesh=Nmesh, **to_mesh_kwargs).paint(mode='real')
        if to_mesh_kwargs.get('compensated', False):
            if normalize:
                # compensate window
                outfield_count = cat.to_mesh(
                Nmesh=Nmesh, **to_mesh_kwargs).compute()
            else:
                raise Exception('Not implemented: compensated and not normalized')
        else:
            outfield_count = cat.to_mesh(
                Nmesh=Nmesh, **to_mesh_kwargs).to_real_field(normalize=normalize)

    if verbose:
        comm = meshsource.comm
        print_cstats(outfield, prefix='outfield tmp: ', comm=comm)
        if weighted_paint_mode == 'avg':
            print_cstats(outfield_count, prefix='outfield count: ', comm=comm)

    # divide weighted 1+delta by number of contributions
    if weighted_paint_mode == 'avg':
        outfield /= outfield_count
        del outfield_count

    # set the mean
    if set_mean is not None:
        outfield = outfield - outfield.cmean() + set_mean

    if verbose:
        print_cstats(outfield, prefix='outfield final: ', comm=comm)

    return outfield, meshsource.attrs

def avg_value_mass_weighted_paint_cat_to_rho(
    cat=None,
    value_column=None,
    weight_ptcles_by=None,
    Ngrid=None,
    fill_empty_cells='RandNeighb',
    RandNeighbSeed=1234,
    raise_exception_if_too_many_empty_cells=True,
    to_mesh_kwargs=None,
    verbose=False
    ):
    """
    Helper function that paints cat[value_column] to grid, averaging over
    values of all particles belonging to a cell, and allowing for 
    additional particle mass weights. Also has several methods to fill empty
    cells.
    """
    # In the code 'value' is called 'chi', because value is chi in reconstruction
    # code.

    if to_mesh_kwargs is None:
        to_mesh_kwargs = {
            'window': 'cic',
            'compensated': False,
            'interlaced': False}

    comm = CurrentMPIComm.get()
    logger = logging.getLogger('paint_utils')

    ## Get mass density rho so we can normalize chi later. Assume mass=1, or given by
    # weight_ptcles_by.
    # This is to get avg chi if multiple ptcles are in same cell.
    # 1 Sep 2017: Want chi_avg = sum_i m_i chi_i / sum_j m_i where m_i is particle mass,
    # because particle mass says how much the average should be dominated by a single ptcle
    # that can represent many original no-mass particles.

    # Compute rho4chi = sum_i m_i
    rho4chi, rho4chi_attrs = weighted_paint_cat_to_delta(
        cat,
        weight=weight_ptcles_by,
        weighted_paint_mode='sum',
        to_mesh_kwargs=to_mesh_kwargs,
        normalize=False,  # want rho not 1+delta
        Nmesh=Ngrid,
        set_mean=None,
        verbose=verbose)

    # compute chi weighted by ptcle mass chi(x)m(x)
    weighted_col = 'TMP weighted %s' % value_column
    if weight_ptcles_by is not None:
        cat[weighted_col] = cat[weight_ptcles_by] * cat[value_column]
    else:
        # weight 1 for each ptcle
        cat[weighted_col] = cat[value_column]
    thisChi, thisChi_attrs = weighted_paint_cat_to_delta(
        cat,
        weight=weighted_col,  # chi weighted by ptcle mass
        weighted_paint_mode='sum',
        to_mesh_kwargs=to_mesh_kwargs,
        normalize=False,  # want rho not 1+delta (TODO: check)
        Nmesh=Ngrid,
        set_mean=None,
        verbose=verbose)

    # Normalize Chi by dividing by rho: So far, our chi will get larger if there are
    # more particles, because it sums up displacements over all particles.
    # To normalize, divide by rho (=mass density on grid if all ptcles have mass m=1,
    # or mass given by weight_ptcles_by).
    # (i.e. divide by number of contributions to a cell)
    if fill_empty_cells in [None, 'SetZero']:
        # Set chi=0 if there are not ptcles in grid cell. Used until 7 April 2017.
        # Seems ok for correl coeff and BAO, but gives large-scale bias in transfer
        # function or broad-band power because violates mass conservation.
        raise Exception('Possible bug: converting to np array only uses root rank?')
        thisChi = FieldMesh(
            np.where(
                rho4chi.compute(mode='real') == 0,
                rho4chi.compute(mode='real') * 0,
                thisChi.compute(mode='real') /
                rho4chi.compute(mode='real')))
        #thisChi = np.where(gridx.G['rho4chi']==0, thisChi*0, thisChi/gridx.G['rho4chi'])

    elif fill_empty_cells in [
        'RandNeighb', 'RandNeighbReadout', 'AvgAndRandNeighb']:

        # Set chi in empty cells equal to a random neighbor cell. Do this until all empty
        # cells are filled.
        # First set all empty cells to nan.
        #thisChi = np.where(gridx.G['rho4chi']==0, thisChi*0+np.nan, thisChi/gridx.G['rho4chi'])
        thisChi = thisChi / rho4chi  # get nan when rho4chi=0
        if True:
            # test if nan ok
            ww1 = np.where(rho4chi == 0)
            #ww2 = np.where(np.isnan(thisChi.compute(mode='real')))
            ww2 = np.where(np.isnan(thisChi))
            assert np.allclose(ww1, ww2)
            del ww1, ww2

        # Progressively replace nan by random neighbors:
        Ng = Ngrid
        #thisChi = thisChi.reshape((Ng,Ng,Ng))
        logger.info('thisChi.shape: %s' % str(thisChi.shape))
        #assert thisChi.shape == (Ng,Ng,Ng)
        # indices of empty cells on this rank
        ww = np.where(np.isnan(thisChi))
        # number of empty cells across all ranks
        Nfill = comm.allreduce(ww[0].shape[0], op=MPI.SUM)
        have_empty_cells = (Nfill > 0)

        if fill_empty_cells in ['RandNeighb', 'RandNeighbReadout']:
            i_iter = -1
            while have_empty_cells:
                i_iter += 1
                if comm.rank == 0:
                    logger.info(
                        "Fill %d empty chi cells (%g percent) using random neighbors"
                        % (Nfill, Nfill / float(Ng)**3 * 100.))
                if Nfill / float(Ng)**3 >= 0.999:
                    if raise_exception_if_too_many_empty_cells:
                        raise Exception(
                            "Stop because too many empty chi cells")
                    else:
                        logger.warning(
                            "More than 99.9 percent of cells are empty")
                # draw -1,0,+1 for each empty cell, in 3 directions
                # r = np.random.randint(-1,2, size=(ww[0].shape[0],3), dtype='int')
                rng = MPIRandomState(comm,
                                     seed=RandNeighbSeed + i_iter * 100,
                                     size=ww[0].shape[0],
                                     chunksize=100000)
                r = rng.uniform(low=-2, high=2, dtype='int', itemshape=(3,))
                assert np.all(r >= -1)
                assert np.all(r <= 1)

                # Old serial code to replace nan by random neighbors.
                # thisChi[ww[0],ww[1],ww[2]] = thisChi[(ww[0]+r[:,0])%Ng, (ww[1]+r[:,1])%Ng, (ww[2]+r[:,2])%Ng]

                if fill_empty_cells == 'RandNeighbReadout':
                    # New parallel code, 1st implementation.
                    # Use readout to get field at positions [(ww+rank_offset+r)%Ng] dx.
                    BoxSize = cat.attrs['BoxSize']
                    dx = BoxSize / (float(Ng))
                    #pos_wanted = ((np.array(ww).transpose() + r) % Ng) * dx   # ranges from 0 to BoxSize
                    # more carefully:
                    pos_wanted = np.zeros((ww[0].shape[0], 3)) + np.nan
                    for idir in [0, 1, 2]:
                        pos_wanted[:, idir] = (
                            (np.array(ww[idir] + thisChi.start[idir]) +
                             r[:, idir]) %
                            Ng) * dx[idir]  # ranges from 0..BoxSize

                    # use readout to get neighbors
                    readout_window = 'nnb'
                    layout = thisChi.pm.decompose(pos_wanted,
                                                  smoothing=readout_window)
                    # interpolate field to particle positions (use pmesh 'readout' function)
                    thisChi_neighbors = thisChi.readout(
                        pos_wanted, resampler=readout_window, layout=layout)
                    if False:
                        # print dbg info
                        for ii in range(10000, 10004):
                            if comm.rank == 1:
                                logger.info(
                                    'chi manual neighbor: %g' %
                                    thisChi[(ww[0][ii] + r[ii, 0]) % Ng,
                                            (ww[1][ii] + r[ii, 1]) % Ng,
                                            (ww[2][ii] + r[ii, 2]) % Ng])
                                logger.info('chi readout neighbor: %g' %
                                            thisChi_neighbors[ii])
                    thisChi[ww] = thisChi_neighbors

                elif fill_empty_cells == 'RandNeighb':
                    # New parallel code, 2nd implementation.
                    # Use collective getitem and only work with indices.
                    # http://rainwoodman.github.io/pmesh/pmesh.pm.html#pmesh.pm.Field.cgetitem.

                    # Note ww are indices of local slab, need to convert to global indices.
                    thisChi_neighbors = None
                    my_cindex_wanted = None
                    for root in range(comm.size):
                        # bcast to all ranks b/c must call cgetitem collectively with same args on each rank
                        if comm.rank == root:
                            # convert local index to collective index using ltoc which gives 3 tuple
                            assert len(ww) == 3
                            wwarr = np.array(ww).transpose()

                            #cww = np.array([
                            #    ltoc(field=thisChi, index=[ww[0][i],ww[1][i],ww[2][i]])
                            #    for i in range(ww[0].shape[0]) ])
                            cww = ltoc_index_arr(field=thisChi,
                                                 lindex_arr=wwarr)
                            #logger.info('cww: %s' % str(cww))

                            #my_cindex_wanted = [(cww[:,0]+r[:,0])%Ng, (cww[1][:]+r[:,1])%Ng, (cww[2][:]+r[:,2])%Ng]
                            my_cindex_wanted = (cww + r) % Ng
                            #logger.info('my_cindex_wanted: %s' % str(my_cindex_wanted))
                        cindex_wanted = comm.bcast(my_cindex_wanted,
                                                   root=root)
                        glob_thisChi_neighbors = cgetitem_index_arr(
                            thisChi, cindex_wanted)

                        # slower version doing the same
                        # glob_thisChi_neighbors = [
                        #     thisChi.cgetitem([cindex_wanted[i,0], cindex_wanted[i,1], cindex_wanted[i,2]])
                        #     for i in range(cindex_wanted.shape[0]) ]

                        if comm.rank == root:
                            thisChi_neighbors = np.array(
                                glob_thisChi_neighbors)
                        #thisChi_neighbors = thisChi.cgetitem([40,42,52])

                    #print('thisChi_neighbors:', thisChi_neighbors)

                    if False:
                        # print dbg info (rank 0 ok, rank 1 fails to print)
                        for ii in range(11000, 11004):
                            if comm.rank == 1:
                                logger.info(
                                    'ww: %s' %
                                    str([ww[0][ii], ww[1][ii], ww[2][ii]]))
                                logger.info(
                                    'chi[ww]: %g' %
                                    thisChi[ww[0][ii], ww[1][ii], ww[2][ii]]
                                )
                                logger.info(
                                    'chi manual neighbor: %g' %
                                    thisChi[(ww[0][ii] + r[ii, 0]) % Ng,
                                            (ww[1][ii] + r[ii, 1]) % Ng,
                                            (ww[2][ii] + r[ii, 2]) % Ng])
                                logger.info('chi bcast neighbor: %g' %
                                            thisChi_neighbors[ii])
                        raise Exception('just dbg')
                    thisChi[ww] = thisChi_neighbors

                ww = np.where(np.isnan(thisChi))
                Nfill = comm.allreduce(ww[0].shape[0], op=MPI.SUM)
                have_empty_cells = (Nfill > 0)
                comm.barrier()

        elif fill_empty_cells == 'AvgAndRandNeighb':
            raise NotImplementedError
            # while have_empty_cells:
            #     print("Fill %d empty chi cells (%g percent) using avg and random neighbors" % (
            #         ww[0].shape[0],ww[0].shape[0]/float(Ng)**3*100.))
            #     # first take average (only helps empty cells surrounded by filled cells)
            #     thisChi[ww[0],ww[1],ww[2]] = 0.0
            #     for r0 in range(-1,2):
            #         for r1 in range(-1,2):
            #             for r2 in range(-1,2):
            #                 if (r0==0) and (r1==0) and (r2==0):
            #                     # do not include center point in avg b/c this is nan
            #                     continue
            #                 else:
            #                     # average over 27-1 neighbor points
            #                     thisChi[ww[0],ww[1],ww[2]] += thisChi[(ww[0]+r0)%Ng, (ww[1]+r1)%Ng, (ww[2]+r2)%Ng]/26.0
            #     # get indices of cells that are still empty (happens if a neighbor was nan above)
            #     ww = np.where(np.isnan(thisChi))
            #     have_empty_cells = (ww[0].shape[0] > 0)
            #     if have_empty_cells:
            #         # draw -1,0,+1 for each empty cell, in 3 directions
            #         r = np.random.randint(-1,2, size=(ww[0].shape[0],3), dtype='int')
            #         # replace nan by random neighbors
            #         thisChi[ww[0],ww[1],ww[2]] = thisChi[(ww[0]+r[:,0])%Ng, (ww[1]+r[:,1])%Ng, (ww[2]+r[:,2])%Ng]
            #         # recompute indices of nan cells
            #         ww = np.where(np.isnan(thisChi))
            #         have_empty_cells = (ww[0].shape[0] > 0)

    else:
        raise Exception("Invalid fill_empty_cells option: %s" %
                        str(fill_empty_cells))

    return thisChi, thisChi_attrs



