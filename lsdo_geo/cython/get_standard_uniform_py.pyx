import numpy as np
cimport numpy as np

from lsdo_geo.cython.get_standard_uniform cimport get_standard_uniform

def get_standard_uniform(int order, int num_control_points, np.ndarray[double] knot_vector):
  get_standard_uniform(order, num_control_points, &knot_vector[0])