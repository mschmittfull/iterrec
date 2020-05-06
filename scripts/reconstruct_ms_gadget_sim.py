from __future__ import print_function, division
from argparse import ArgumentParser
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os

from nbodykit import style
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.algorithms.fftpower import FFTPower

from iterrec import reconstruction as rec
from iterrec.model_target_pair import Model, Target
from iterrec.smoothing import GaussianSmoother
from iterrec.utils import linear_rescale_fac, mass_weighted_paint_cat_to_delta


def main():
    """
    Run reconstruction on ms_gadget simulations. These are the 500Mpc/h sims
    used in arxiv:1811.10640.
    """

    # Parse arguments
    ap = ArgumentParser()

    ap.add_argument('--input_catalog',
                    type=str,
                    default='../data/snap_0.6250_sub_sr0.00025_ssseed40400.bigfile',
                    help='Input clustered catalog.')

    ap.add_argument('--Nmesh',
                    type=int,
                    default=64,
                    help='Nmesh used for painting catalog to mesh.')

    # Reconstruction settings
    ap.add_argument('--Nsteps',
                    type=int,
                    default=4,
                    help='Number of reconstruction steps.')

    # More optional arguments.
    s = ('True linear density, used to compute power spectra and '
        'assess reconstruction performance. Set to empty string if not wanted.')
    ap.add_argument('--true_linear_density',
                    type=str,
                    default='../data/IC_LinearMesh_z0_Ng64',
                    help=s)

    ap.add_argument('--compute_power',
                    default=False, action='store_true',
                    help='Compute power spectra.')

    # The redshifts are used to rescale the true linear density.
    ap.add_argument('--input_scale_factor',
                    default=0.625,
                    type=float,
                    help='Redshift of input catalog.')

    ap.add_argument('--redshift_of_true_linear_density',
                    default=0.0,
                    type=float,
                    help='Redshift of true linear density file. Needed to '
                        'convert to redshift of input catalog')

    ap.add_argument('--out_density',
                    default='out/reconstructed_density.bigfile',
                    type=str,
                    help='File where to write reconstructed linear density')

    ap.add_argument('--out_power',
                    type=str,
                    default='out/power',
                    help='File where to write power spectra.')

    args = ap.parse_args()


    # Global params

    Nmesh = args.Nmesh

    # cosmology of ms_gadget sims (to compute D_lin(z) so we can rescale true linear density)
    # omega_m = 0.307494
    # omega_bh2 = 0.022300
    # omega_ch2 = 0.118800
    # h = math.sqrt((omega_bh2 + omega_ch2) / omega_m) = 0.6774
    cosmo_params = dict(Om_m=0.307494,
                       Om_L=1.0 - 0.307494,
                       Om_K=0.0,
                       Om_r=0.0,
                       h0=0.6774)
    #f_log_growth = np.sqrt(0.61826)


    # Input catalogs
    inputcat_spec = Target(
        name='input',
        in_fname=args.input_catalog,
        position_column='Position'
    )

    if args.true_linear_density != '':
        # Model mesh
        z_rescalefac = linear_rescale_fac(
            current_scale_factor=1./(1.+args.redshift_of_true_linear_density),
            desired_scale_factor=args.input_scale_factor,
            cosmo_params=cosmo_params)
        deltalin = Model(
            name='deltalin',
            in_fname=args.true_linear_density,
            rescale_factor=z_rescalefac,
            read_mode='delta from 1+delta',
            filters=None,
            readout_window='cic')
    else:
        deltalin = None


    # Reconstruction settings
    smoothers = []
    smoothers.append(GaussianSmoother(
        R1=20.0, # smoothing scale R in first iteration step
        R_reduction_factor=2.0, # factor by which R is reduced in each step.
        Rmin=1.0, # minimum smoothing scale allowed.
    ))
    reconstructor = rec.Reconstructor(
        smoothers=smoothers,
        displacement_factor=1.0,
        Nsteps=args.Nsteps,
        Nmesh=Nmesh,
        paint_window='cic',
        name='A reconstruction test')

    # Load catalog and run reconstruction
    #cat = DMcat.get_catalog()
    cat = inputcat_spec.get_catalog()
    mesh_deltalin_rec = reconstructor.reconstruct_linear_density_from_catalog(
        cat)

    # Save reconstructed density to disk
    mesh_deltalin_rec.save(args.out_density)
    if cat.comm.rank == 0:
        print('Reconstruction done.')
        print('Saved reconstructed linear density to %s' % 
            args.out_density)

    power = None
    if args.compute_power:
        if cat.comm.rank == 0:
            print('Compute power spectra of various densities.')
        power = calc_power_spectra(
            mesh_deltalin_rec, cat, deltalin, reconstructor, args.out_power)

    return mesh_deltalin_rec, power


def calc_power_spectra(mesh_deltalin_rec, cat, deltalin, reconstructor,
    out_fname):
    
    power = OrderedDict()
    BoxSize = cat.attrs['BoxSize']

    power['Prr'] = calc_power(mesh_deltalin_rec, BoxSize=BoxSize)    

    if deltalin is not None:

        # Compute more power spectra with true linear density to assess
        # reconstruction performance.

        # Load true linear density
        mesh_deltalin = deltalin.get_mesh()

        # Compute density of input catalg
        mesh_delta_input, attrs = mass_weighted_paint_cat_to_delta(
            cat,
            weight=None,
            Nmesh=reconstructor.Nmesh,
            to_mesh_kwargs={
                'window': reconstructor.paint_window,
                'compensated': False,
                'interlaced': False},
            verbose=False)

        # Compute power spectra of reconstructed and true linear density
        
        # t: True linear density
        # r: Reconstructed linear density
        # i: Input catalog density
        power['Ptt'] = calc_power(mesh_deltalin, BoxSize=BoxSize)
        power['Prt'] = calc_power(mesh_deltalin_rec, second=mesh_deltalin, 
            BoxSize=BoxSize)
        power['Pii'] = calc_power(mesh_delta_input, BoxSize=BoxSize)
        power['Pit'] = calc_power(mesh_delta_input, second=mesh_deltalin,
            BoxSize=BoxSize)

    # Save power in structured numpy array and save to disk
    dtype = [('k', 'f8')]
    for key in power.keys():
        dtype.append((key, 'f8'))
    power_arr = np.empty(shape=power['Prr'].power['k'].shape, dtype=dtype)
    power_arr['k'] = power['Prr'].power['k']
    for key in power.keys():
        power_arr[key] = power[key].power['power'].real

    # save as ascii file
    if not os.path.exists(os.path.dirname(out_fname)):
        os.makedirs(os.path.dirname(out_fname))
    np.savetxt(
        out_fname+'.txt', power_arr,
        header=(
            'r: reconstructed, t: true linear density, i: input catalog.\n'
            'Columns:\n' + ' '.join(power_arr.dtype.names)))
    if cat.comm.rank == 0:
        print('Wrote %s' % (out_fname+'.txt'))

    # also save all power spectra as json
    for key, val in power.items():
        val.save('%s_%s.json' % (out_fname, key))

    return power
   

# Function to compute power spectrum
def calc_power(mesh, second=None, mode='1d', k_bin_width=1.0, verbose=False, 
    los=None, BoxSize=None):
    """
    Basically the same as nbodykit FFTPower, but specify a certain binning in k.
    """
            
    if BoxSize is None:
        BoxSize = mesh.attrs['BoxSize']
    if len(BoxSize) > 1:
        assert BoxSize[0] == BoxSize[1]
        assert BoxSize[0] == BoxSize[2]
    boxsize = BoxSize[0]
    dk = 2.0 * np.pi / boxsize * k_bin_width
    kmin = 2.0 * np.pi / boxsize / 2.0

    if mode == '1d':
        res = FFTPower(first=mesh,
                        second=second,
                        mode=mode,
                        dk=dk,
                        kmin=kmin)
    elif mode == '2d':
        res = FFTPower(first=mesh,
                            second=second,
                            mode=mode,
                            dk=dk,
                            kmin=kmin,
                            poles=[0,2,4],
                            Nmu=5,
                            los=los)
    else:
        raise Exception("Mode not implemented: %s" % mode)

    return res


if __name__ == '__main__':
    main()



