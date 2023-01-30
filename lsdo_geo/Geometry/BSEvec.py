"""
B-spline surface-modeling engine (BSE) vec class
"""

# pylint: disable=E1101

from __future__ import division
import numpy



class BSEvec(object):

    def __init__(self, name, size, ndim, hidden):
        self.array = numpy.zeros((size, ndim))
        self.name = name
        self.size = size
        self.ndim = ndim
        self._hidden = hidden
        self._file = None
        self._default_var_names = ['v' + str(idim) 
                                   for idim in range(self.ndim)]

class BSEvecUns(BSEvec):

    pass

class BSEvecStr(BSEvec):

    def __init__(self, name, size, ndim, surf_sizes, hidden,
                 bse=None):
        super(BSEvecStr, self).__init__(name, size, ndim, hidden)

        self._bse = bse
        self.surfs = []
        if surf_sizes is not None:
            ind1, ind2 = 0, 0
            for isurf in range(surf_sizes.shape[0]):
                num_u, num_v = surf_sizes[isurf, :]
                ind2 += num_u * num_v
                surf = self.array[ind1:ind2]
                surf = surf.reshape((num_u, num_v, ndim), 
                                    order='F')
                ind1 += num_u * num_v
                self.surfs.append(surf)

    def __call__(self, isurf):
        return self.surfs[isurf]
