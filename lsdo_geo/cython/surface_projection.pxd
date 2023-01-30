from libc.stdlib cimport malloc, free

from lsdo_geo.cython.get_open_uniform cimport get_open_uniform
from lsdo_geo.cython.basis0 cimport get_basis0
from lsdo_geo.cython.basis1 cimport get_basis1
from lsdo_geo.cython.basis2 cimport get_basis2
from lsdo_geo.cython.basis_matrix_surface cimport get_basis_surface_matrix


cdef compute_surface_projection(
    int order_u, int num_control_points_u,
    int order_v, int num_control_points_v,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v,
    double* u_vec, double* v_vec,
    int guess_grid_n,
)