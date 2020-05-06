from __future__ import print_function, division

from collections import Counter, OrderedDict
from copy import copy
import numpy as np

from nbodykit.source.mesh.field import FieldMesh

from utils import catalog_persist, get_cstats_string, get_displacement_from_density_rfield, calc_divergence_of_3_meshs, mass_weighted_paint_cat_to_delta, avg_value_mass_weighted_paint_cat_to_rho, readout_mesh_at_cat_pos


class Reconstructor(object):
    """
    Class for performing BAO/initial condition reconstruction.

    Displacement_factor should be 1 for DM and 1/b1 for biased tracers. It
    is probably fine to choose it smaller, but it's probably bad to choose
    it larger because of potential overshooting.
    """

    def __init__(
        self,
        smoothers=None,
        displacement_factor=1.0,
        Nsteps=2,
        displacement_type='Zeldovich',
        method_to_get_lin_density_from_displ_cat='div_chi',
        Nmesh=None,
        paint_window='cic',
        readout_window='cic',
        name=None
        ):
        self.smoothers = smoothers
        self.displacement_factor = displacement_factor
        self.Nsteps = Nsteps
        self.displacement_type = displacement_type
        self.method_to_get_lin_density_from_displ_cat = (
            method_to_get_lin_density_from_displ_cat)
        self.Nmesh = Nmesh
        self.paint_window = paint_window
        self.readout_window = readout_window
        self.name = name

        if self.smoothers is None:
            self.smoothers = []

    def to_dict(self):
        return dict(
            smoothers=[s.to_dict() for s in self.smoothers],
            displacement_factor=self.displacement_factor,
            Nsteps=self.Nsteps,
            displacement_type=self.displacement_type,
            method_to_get_lin_density_from_displ_cat=(
                self.method_to_get_lin_density_from_displ_cat),
            Nmesh=self.Nmesh,
            paint_window=self.paint_window,
            readout_window=self.readout_window,
            name=self.name)


    def reconstruct_linear_density_from_catalog(self, cat, verbose=True):
        """
        Given catalog of objects, reconstruct initial conditions and return.
        """

        # Do Nsteps-many displacement iterations of catalog.

        # make a deep copy
        cat_displaced = catalog_persist(cat, columns=cat.columns)

        for istep in range(self.Nsteps):
            if cat.comm.rank == 0:
                print('Start displacement iteration %d of %d' % (
                    istep+1, self.Nsteps))
            cat_displaced = self.displace_catalog_Nth_iteration(
                cat=cat_displaced, N=istep)

        # Now cat_displaced contains the estimated Lagrangian positions \hat q.
        # Its density perturbation should be very small.

        # Store the total displacement chi of each particle, as function of 
        # estimated Lagrangian positions \hat q.
        if cat.comm.rank == 0:
            print('Get chi(q) displacement')
        
        for component in [0,1,2]:
            cat_displaced['chi_%d' % component] = (
                cat['Position'][:,component] 
                - cat_displaced['Position'][:,component])

            if verbose:
                s = get_cstats_string(
                    cat_displaced['chi_%d' % component].compute())
                if cat.comm.rank == 0:
                    print('chi_%d: '%component, s)

        # Estimate linear density from displaced catalog
        deltalin_rec = self.estimate_linear_density_from_displaced_catalog(
            cat_displaced,
            method=self.method_to_get_lin_density_from_displ_cat,
            chi_columns=['chi_0', 'chi_1', 'chi_2']
            )

        if verbose:
            s = get_cstats_string(deltalin_rec.compute(mode='real'))
            if cat_displaced.comm.rank == 0:
                print('Reconstructed deltalin: ', s)

        return deltalin_rec


    def estimate_linear_density_from_displaced_catalog(
        self,
        cat_displaced,
        method='div_chi',
        chi_columns=None,
        second_cat_displaced=None,
        fill_empty_cells='RandNeighb'
        ):
        """
        Estimate linear density from a displaced catalog.

        Parameters
        ----------
        method : str
            If 'div_chi', take the divergence of chi_columns.
            If 'div_chi_2ndorder', include 2nd order correction.
            If 'displaced_catalogs_difference', take difference of cat_displaced
            (usually displaced clustered catalog) and second_cat_displaced
            (usually displaced random catalog). This is done in standard
            Eisenstein et al. 2007 reconstruction.
        """
        if cat_displaced.comm.rank == 0:
            print('Estimate linear density from displaced catalog, method=%s'%(
                method))

        if method == 'div_chi':
            # Paint chi to grid. Use 'average' painting, so that a cell that has
            # many particles with the same value of chi receives the average value
            # and not sum of all chi values.
            chi_meshs = []
            for component in [0,1,2]:

                chi_mesh, attrs = avg_value_mass_weighted_paint_cat_to_rho(
                    cat=cat_displaced,
                    value_column=chi_columns[component],
                    weight_ptcles_by=None, # TODO: add support for ptcle mass
                    Ngrid=self.Nmesh,
                    fill_empty_cells=fill_empty_cells,
                    RandNeighbSeed=1234,
                    raise_exception_if_too_many_empty_cells=True,
                    to_mesh_kwargs={
                        'window': self.paint_window,
                        'compensated': False,
                        'interlaced': False})

                # fill with 0
                # chi_mesh, attrs = mass_avg_weighted_paint_cat_to_rho(
                #     cat_displaced,
                #     weight=chi_columns[component],
                #     Nmesh=self.Nmesh,
                #     to_mesh_kwargs={
                #         'window': self.paint_window,
                #         'compensated': False,
                #         'interlaced': False
                #     },
                #     rho_of_empty_cells=0.0,
                #     verbose=False)

                s = get_cstats_string(chi_mesh)
                if cat_displaced.comm.rank == 0:
                    print('chi_%d on grid: ' % component, s)

                chi_meshs.append(FieldMesh(chi_mesh))


            # Compute divergence
            deltalin_rec = calc_divergence_of_3_meshs(chi_meshs, prefactor=-1.0)

        elif method == 'div_chi_2ndorder':
            # TODO
            raise NotImplementedError

        elif method == 'displaced_catalogs_difference':
            # TODO
            raise NotImplementedError

        else:
            raise Exception('Invalid argument method=%s' % str(method))

        return deltalin_rec




    def displace_catalog_Nth_iteration(self, cat, N, verbose=True):
        """
        Modifies cat in argument, displacing to new positions.
        """

        # Paint the catalog to a mesh. delta is pmesh.pm.RealField object.
        delta, mesh_attrs = mass_weighted_paint_cat_to_delta(cat,
                                    weight=None,
                                    Nmesh=self.Nmesh,
                                    to_mesh_kwargs={
                                        'window': self.paint_window,
                                        'compensated': False,
                                        'interlaced': False
                                    },
                                    set_mean=0,
                                    verbose=False)
        # convert to meshsource
        delta = FieldMesh(delta)

        # Apply smoothing of Nth iteration
        for smoother in self.smoothers:
            delta = smoother.apply_smoothing_of_Nth_iteration(
                meshsource=delta, N=N)

        if verbose:
            # slow b/c needs fft
            s = get_cstats_string(delta.compute(mode='real'))
            if cat.comm.rank == 0:
                print('delta after smoothing: ', s)

        # Compute Zeldovich displacement
        for component in [0,1,2]:
            Psi = get_displacement_from_density_rfield(
                in_density_rfield=None,
                in_density_cfield=delta.compute(mode='complex'),
                component=component,
                Psi_type=self.displacement_type,
                smoothing=None,
                smoothing_Psi3LPT=None,
                prefac_Psi_1storder=1.0,
                prefac_Psi_2ndorder=1.0,
                prefac_Psi_3rdorder=1.0,
                RSD=False,
                RSD_line_of_sight=None,
                RSD_f_log_growth=None)

            if verbose:
                s = get_cstats_string(Psi*self.displacement_factor)
                if cat.comm.rank == 0:
                    print('Psi_%d: ' % component, s)

            # read out Psi component at catalog positions
            Psi_at_cat_pos = readout_mesh_at_cat_pos(
                mesh=FieldMesh(Psi),
                cat=cat, 
                readout_window=self.readout_window)

            # convert to (Nobjects,3) array
            Psi3d_at_cat_pos = np.zeros(cat['Position'].shape)
            Psi3d_at_cat_pos[:,component] = Psi_at_cat_pos

            # displace catalog
            cat['Position'] -= self.displacement_factor * Psi3d_at_cat_pos

            del Psi_at_cat_pos, Psi3d_at_cat_pos

        return cat


    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()

    

