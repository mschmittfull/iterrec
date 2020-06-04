from __future__ import print_function, division

from collections import Counter, OrderedDict
import numpy as np

from nbodykit.source.mesh.field import FieldMesh


class Smoother(object):
    """
    Class to apply smoothing to field in n-th iteration of reconstruction.
    """
    def __init__(self):
        raise NotImplementedError

    def get_smoothing_kernel_of_Nth_iteration(self, N):
        raise NotImplementedError


class GaussianSmoother(Smoother):
    """
    Apply Gaussian smoothing to field in n-th iteration of reconstruction.
    """
    def __init__(
        self,
        R1=20.0, # smoothing scale R in first iteration step
        R_reduction_factor=2.0, # factor by which R is reduced in each step.
        Rmin=1.0, # minimum smoothing scale allowed.
        name='Gaussian'
        ):
        self.R1 = R1
        self.R_reduction_factor = R_reduction_factor
        self.Rmin = Rmin
        self.name = name

    def get_smoothing_kernel_of_Nth_iteration(self, N):
        R = float(self.R1) / float(self.R_reduction_factor)**N
        if R < self.Rmin:
            R = self.Rmin
        if R is None or R==0.:
            def kernel_fcn(k3vec, val):
                return val
        else:
            def kernel_fcn(k3vec, val):
                k2 = sum(ki**2 for ki in k3vec)  # |\vk|^2 on the mesh
                return np.exp(- 0.5 * R**2 * k2) * val
        return kernel_fcn

    def apply_smoothing_of_Nth_iteration(self, N, meshsource):
        # make a copy
        out = FieldMesh(meshsource.compute(mode='complex'))
        kernel_fcn = self.get_smoothing_kernel_of_Nth_iteration(N=N)
        out = out.apply(kernel_fcn, kind='wavenumber', mode='complex')
        return out

    def to_dict(self):
        return dict(
            R1=self.R1, R_reduction_factor=self.R_reduction_factor,
            Rmin=self.Rmin, name=self.name)

    def __str__(self):
        return json.dumps(self.to_dict())

    def __repr__(self):
        return self.__str__()
