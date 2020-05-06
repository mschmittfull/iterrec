from __future__ import print_function, division
from mpi4py import MPI
import numpy as np
from pmesh.pm import RealField, ComplexField


def ltoc(field, index):
    """
    Convert local to collective index, inverting pm.pmesh.Field._ctol.
    Index must be a single tuple with length field.ndim
    """
    assert isinstance(field, RealField)
    assert np.array(index).shape == (field.ndim,)
    return tuple(list(index + field.start))


def ltoc_index_arr(field, lindex_arr):
    """
    Convert local to collective index, inverting pm.pmesh.Field._ctol. 
    lindex_arr is an array of local indices, with last axis labeling
    the dimension of the field.

    Example: If we want to convert N indices of a 3-D field, have
    lindex_arr.shape=(N,3).
    """
    assert isinstance(field, RealField)
    assert type(lindex_arr) == np.ndarray
    assert np.all(lindex_arr >= 0)
    assert lindex_arr.shape[-1] == field.ndim
    cindex_arr = lindex_arr + field.start
    return cindex_arr


def cgetitem_index_arr(field, cindex_arr):
    """
    Get values of field for an array of collective cindices (ranging from
    0 to Ngrid-1 in each dimension).

    This is a vector version of pmesh.pm.Field.cgetitem.

    Do this by getting field[cindex_arr-field.start] but only if
    cindex_arr item is between field.start and field.start+field.shape,
    which depends on the MPI rank.
    
    Do this essentially by running
    if all(index1 >= self.start) and all(index1 < self.start + self.shape):
        return field[index1 - self.start]
    else:
        return 0
    Then do allreduce to get field value across all ranks.
    """
    assert isinstance(field, RealField)
    assert type(cindex_arr) == np.ndarray
    assert np.all(cindex_arr >= 0)
    assert cindex_arr.shape[-1] == field.ndim
    assert field.ndim == 3
    value_arr = np.zeros(cindex_arr.shape[:-1], dtype=field.value.dtype)
    www = np.where(
        np.all(cindex_arr >= field.start, axis=-1) &
        np.all(cindex_arr < (field.start + field.shape), axis=-1))[0]
    lindex_wanted = (cindex_arr[www, :] - field.start)
    value_arr[www] = field[lindex_wanted[:, 0], lindex_wanted[:, 1],
                           lindex_wanted[:, 2]]
    value_arr = field.pm.comm.allreduce(value_arr, op=MPI.SUM)
    return value_arr
