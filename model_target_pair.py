from __future__ import print_function, division

from collections import Counter, OrderedDict
import json
import numpy as np

from nbodykit import CurrentMPIComm
from nbodykit.source.catalog import BigFileCatalog
from nbodykit.source.mesh.field import FieldMesh


from utils import catalog_persist, get_cstats_string
from utils import read_delta_from_rho_bigfile, read_delta_from_1plusdelta_bigfile, read_vel_from_bigfile, readout_mesh_at_cat_pos, read_delta_from_2plusdelta_bigfile



class ModelTargetPair(object):
    """
    Class for a pair of model and target. Specify a model and a target, and
    then use methods to load them, on a grid or as catalog at target positions.

    We use this to compare velocity models.

    The target is assumed to be a catalog. The model is assumed to be a mesh.
    """

    def __init__(
        self,
        model=None,
        target=None,
        name=None,
        ):
        self.model = model
        self.target = target
        if name is None:
            self.name = '%s :: %s' % (model.name, target.name)

    def to_dict(self):
        return {
            'model': self.model.to_dict(),
            'target': self.target.to_dict(),
            'name': self.name
        }

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def readout_model_at_target_pos(self):
        # get model on mesh
        model_mesh = self.model.get_mesh()

        # get target catalog
        target_cat = self.target.get_catalog()

        return readout_mesh_at_cat_pos(
            mesh=model_mesh,
            cat=target_cat, 
            readout_window=self.model.readout_window)

    def get_target_val_at_target_pos(self):
        return self.target.get_target_val_at_target_pos()


class Model(object):
    """
    A model which lives on a mesh.
    """
    def __init__(
        self,
        name=None, # string to describe the model
        in_fname=None,  # filename of bigfile mesh
        rescale_factor=None,
        read_mode=None, # 'density' or 'velocity' (don't divide out mean)
        filters=None, # e.g. filter function to apply k_0/k^2
        readout_window='cic'):

        self.name = name
        self.in_fname = in_fname
        self.rescale_factor = rescale_factor
        self.read_mode = read_mode
        self.filters = filters
        self.readout_window = readout_window

        if self.filters is None:
            self.filters = []

    def to_dict(self):
        return {
            'name': self.name,
            'in_fname': self.in_fname,
            'rescale_factor': self.rescale_factor,
            'read_mode': self.read_mode,
            'filters': self.filters,
            'readout_window': self.readout_window
        }

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def get_mesh(self):
        """
        Get the model as a MeshSource object.
        """
        comm = CurrentMPIComm.get()
        if comm.rank == 0:
            print('Read %s' % self.in_fname)

        # read mesh from disk
        if self.read_mode == 'velocity':
            # get rho (don't divide by mean)
            mesh = read_vel_from_bigfile(self.in_fname)
        elif self.read_mode == 'delta from rho':
            # compute fractional delta (taking out mean), assuming file has rho
            mesh = read_delta_from_rho_bigfile(self.in_fname)
        elif self.read_mode == 'delta from 1+delta':
            # compute delta, assuming file has 1+delta
            mesh = read_delta_from_1plusdelta_bigfile(self.in_fname)
        elif self.read_mode == 'delta from 2+delta':
            # compute delta, assuming file has 1+delta
            mesh = read_delta_from_2plusdelta_bigfile(self.in_fname)
        else:
            raise Exception('Invalid read_mode: %s' % str(self.read_mode))

        # multiply mesh by rescale factor
        if self.rescale_factor not in [None, 1.0]:
            if comm.rank == 0:
                print('Apply rescale fac to %s: %s' % (
                    self.name, str(self.rescale_factor)))
            if type(self.rescale_factor) in [float,np.float,np.float64]:
                mesh = FieldMesh(mesh.compute(mode='real')*self.rescale_factor)
            else:
                print(type(self.rescale_factor))
                raise Exception("Invalid rescale factor: %s" % 
                    str(self.rescale_factor))

        # apply filters to mesh
        for filter_fcn in self.filters:
            # make copy
            mesh2 = FieldMesh(mesh.compute())

            # apply filter
            mesh2 = mesh2.apply(filter_fcn, kind='wavenumber', mode='complex')
            
            # compute and save in mesh
            mesh = FieldMesh(mesh2.compute(mode='real'))

        cstats = get_cstats_string(mesh.compute())
        if comm.rank == 0:
            print('MESH %s: %s\n' % (self.name, cstats))
   
        return mesh



class Target(object):
    def __init__(
        self,
        name=None, # string to describe the target catalog
        in_fname=None, # filename of bigfile catalog
        in_catalog=None, # nbodykit catalog, used if in_fname is None
        position_column=None, # must be in Mpc/h
        velocity_column=None,
        apply_RSD_to_position=False,
        RSDFactor=None, # applied to velocity_column to get RSD displacement
        RSD_los=None,
        val_column=None,
        val_component=None,
        rescale_factor=None, # applied to val_column
        cuts=None
        ):
        """
        Example: 

        kwargs= {
            'in_fname': fof_fname,
            'position_column': 'CMPosition',
            'val_column': 'CMVelocity',
            'val_component': direction,
            'rescale_factor': 'RSDFactor',
            'cuts': [('log10M', 'min', 10.8),
                      ('log10M', 'max', 11.8),
                      ('CMPosition', 'max', [100.,100.,20.])
         ]
        CatalogSpec(**kwargs)


        Use either in_fname or in_catalog.

        Parameters
        ----------
        cuts : list of 3-tuples and SimGalaxyCatalogCreator objects
            List of cuts to be applied to the catalog. Each list element
            can be a 3-tuple specifying the cut column, operation and value,
            or it can be a SimGalaxyCatalogCreator object. 
        """
        self.name = name
        self.in_catalog = in_catalog
        self.in_fname = in_fname
        self.position_column = position_column
        self.velocity_column = velocity_column
        self.apply_RSD_to_position = apply_RSD_to_position
        self.RSDFactor = RSDFactor
        self.RSD_los = RSD_los
        self.val_column = val_column
        self.val_component = val_component
        self.rescale_factor = rescale_factor # applied to value
        self.cuts = cuts

        if self.cuts is None:
            self.cuts = []

    def to_dict(self):
        return dict(
            name=self.name,
            in_fname=self.in_fname,
            position_column=self.position_column,
            velocity_column=self.velocity_column,
            apply_RSD_to_position=self.apply_RSD_to_position,
            RSDFactor=self.RSDFactor,
            RSD_los=self.RSD_los,
            val_column=self.val_column,
            val_component=self.val_component,
            rescale_factor=self.rescale_factor,
            cuts=self.cuts
            )

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def get_target_val_at_target_pos(self):
        cat = self.get_catalog()
        return cat['val']

    def get_catalog(self, keep_all_columns=False):
        """
        Get the target catalog, with columns 'Position' and 'val'.
        """
        comm = CurrentMPIComm.get()

        if self.in_fname is None:
            # use copy of in_catalog in memory
            cat = catalog_persist(
                self.in_catalog, columns=self.in_catalog.columns)
        else:
            assert self.in_catalog is None

            if comm.rank == 0:
                print('Read %s' % self.in_fname)

            # Read rho, not dividing or subtracting mean, to get velocity correctly
            cat = BigFileCatalog(self.in_fname, dataset='./', header='Header')

        # Set Position column
        if self.position_column != 'Position':
            print('position_column: ', self.position_column)
            cat['Position'] = cat[self.position_column]
            print('Catalog columns:', cat.columns)


        # add RSD
        if self.apply_RSD_to_position:
            print('Apply RSD')
            cat['Position'] += (
                self.RSDFactor * cat[self.velocity_column] * self.RSD_los)


        # cuts
        Ngal_before_cuts = cat.csize
        for cut in self.cuts:
            if type(cut) == tuple:
                assert len(cut) == 3
                cut_column, cut_op, cut_value = cut
                print('cut:', cut_column, cut_op, cut_value)
                if type(cut_value) in (list, tuple):
                    # cut value is a list or tuple, e.g. to cut 'Position'
                    for i, cv in enumerate(cut_value):
                        if cut_op == 'min':
                            cat = cat[ cat[cut_column][:,i] >= cv ]
                        elif cut_op == 'max':
                            cat = cat[ cat[cut_column][:,i] < cv ]
                        else:
                            raise Exception('Invalid cut operation %s'%str(cut_op))
                else:
                    # cut value is a scalar, e.g. to cut mass
                    if cut_op == 'min':
                        cat = cat[ cat[cut_column] >= cut_value ]
                    elif cut_op == 'max':
                        cat = cat[ cat[cut_column] < cut_value ]
                    else:
                        raise Exception('Invalid cut operation %s' % str(cut_op))

            elif isinstance(cut, SimGalaxyCatalogCreator):
                cat = cut.get_galaxy_catalog_from_source_catalog(cat)

            else:
                raise Exception('Invalid cut %s' % str(cut))

        Ngal_after_cuts = cat.csize
        print('Cuts removed %g%% of objects' % (
            100.*float(Ngal_after_cuts-Ngal_before_cuts)/float(Ngal_before_cuts)))


        # compute the value we are interested in, save in 'val' column
        component = self.val_component
        if self.val_column is not None:
            if component is None:
                cat['val'] = cat[self.val_column][:]
            else:
                cat['val'] = cat[self.val_column][:, component]

            # catalog rescale factor to be applied to value column
            if self.rescale_factor is not None:
                if self.rescale_factor == 'RSDFactor':
                    fac = cat.attrs['RSDFactor'][0]
                elif type(self.rescale_factor) == float:
                    fac = self.rescale_factor
                else:
                    raise Exception('Invalid rescale_factor %s' % 
                        self.rescale_factor)

                if cat.comm.rank == 0:
                    print('Apply rescale factor: %g' % fac)
                cat['val'] *= fac

        if keep_all_columns:
            pass
        else:
            # keep only Position and val columns, delete all other columns
            cat = catalog_persist(cat, columns=['Position','val'])

        if 'val' in cat.columns:
            cstats = get_cstats_string(cat['val'].compute())
            if comm.rank == 0:
                print('TARGET CATALOG %s: %s\n' % (self.name, cstats))

        return cat

    



def get_model_target_pair_from_list(model_target_pairs=None, name=None):
    """
    From list of ModelTargetPair objects, return the one where name
    is the one specified by the argument.
    
    Parameters
    ----------
    model_target_pairs : list of ModelTargetPair objects
        List of ModelTargetPair objects to search.
    
    name : str
        Find the object that has this name.
    """
    names = [mtp.name for mtp in model_target_pairs]
    if names.count(name) != 1:
        raise Exception('Could not find ModelTargetPair %s' % name)
    idx = names.index(name)
    return model_target_pairs[idx]

