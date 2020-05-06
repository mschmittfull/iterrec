iterrec
=========================================
Iterative reconstruction of linear BAO and linear initial conditions.

The code reads a clustered catalog of objects and performs reconstruction. 
The current implementation supports dark matter in real space.
For details of the algorithm see https://arxiv.org/abs/1704.06634.

Running in notebook
-------------------

- To run the code in a notebook, see `reconstruction_test.ipynb`_.

.. _reconstruction_test.ipynb: notebooks/reconstruction_test.ipynb

- The basic usage is like this:

  .. code-block:: python

    from iterrec import reconstruction as rec
    from iterrec.smoothing import GaussianSmoother

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
        Nsteps=4,
        Nmesh=64,
        paint_window='cic',
        name='A reconstruction test')

    # Load catalog
    cat = DMcat_small.get_catalog()

    # Run reconstruction
    mesh_deltalin_rec = reconstructor.reconstruct_linear_density_from_catalog(cat)

Running from the command line
-----------------------------

- To run the code from the command line, see `reconstruct_ms_gadget_sim.py`_. 

.. _reconstruct_ms_gadget_sim.py: scripts/reconstruct_ms_gadget_sim.py

- For an example SLURM script to run on a cluster see `reconstruct_ms_gadget_sim.job.helios`_.

.. _reconstruct_ms_gadget_sim.job.helios: scripts/reconstruct_ms_gadget_sim.job.helios

- When running the script, the rms displacements in each iteration step and the rms density after each displacement step are displayed. Ideally both should decrease with every step.

- The script saves the reconstructed linear density to disk. If the optional --compute_power argument is set, the script also computes power spectra of the input catalog and the reconstructed linear density.

- General usage: 

  .. code-block:: bash

    $ python reconstruct_ms_gadget_sim.py [-h] [--input_catalog INPUT_CATALOG]
                                    [--Nmesh NMESH] [--Nsteps NSTEPS]
                                    [--true_linear_density TRUE_LINEAR_DENSITY]
                                    [--compute_power]
                                    [--input_scale_factor INPUT_SCALE_FACTOR]
                                    [--redshift_of_true_linear_density REDSHIFT_OF_TRUE_LINEAR_DENSITY]
                                    [--out_density OUT_DENSITY]
                                    [--out_power OUT_POWER]

- On a cluster: 

  .. code-block:: bash

    $ sbatch reconstruct_ms_gadget_sim.job.helios

- The runtime depends on the reconstruction settings and the size of the input catalog. For example, running 8 iteration steps with Nmesh=512 on an input catalog with 90 million particles takes 10 minutes on 14 cores.


Installation
------------
The code requires `nbodykit <https://github.com/bccp/nbodykit>`_ version 0.3.x or higher.

To install this it is best to follow the instructions on the nbodykit website.

To install in a new anaconda environment, use for example

.. code-block:: bash

  $ cd ~/anaconda/anaconda/envs
  $ conda create -n nbodykit-0.3.7-env -c bccp -c astropy python=2.7 nbodykit=0.3.7 bigfile pmesh ujson

Newer versions of nbodykit should also work but are not tested. 

To activate the environment, use

.. code-block:: bash

  $ source activate nbodykit-0.3.7-env

To deactivate it, use 

.. code-block:: bash

  $ source deactivate

To run the reconstruction code, clone the github repository to a local folder. Then add it to your PYTHONPATH by adding this line to ~/.bash_profile:

.. code-block:: bash

  export PYTHONPATH=/Users/mschmittfull/Dropbox/CODE/iterrec:$PYTHONPATH


Contributing
------------
To contribute, create a fork on github, make changes and commits, and submit a pull request on github.

To get consistent code style, run

.. code-block:: bash

  $ yapf -i *.py */*.py
