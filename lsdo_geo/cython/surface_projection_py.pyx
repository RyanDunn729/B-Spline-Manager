import numpy as np
cimport numpy as np

from lsdo_geo.cython.surface_projection cimport compute_surface_projection


def compute_surface_projection(
    int order_u, int num_control_points_u,
    int order_v, int num_control_points_v,
    int num_points, int max_iter,
    np.ndarray[double] pts,  np.ndarray[double] cps,
    np.ndarray[double] knot_vector_u, np.ndarray[double] knot_vector_v,
    np.ndarray[double] u_vec, np.ndarray[double] v_vec,
    int guess_grid_n
):
    compute_surface_projection(
        order_u, num_control_points_u,
        order_v, num_control_points_v,
        num_points, max_iter,
        &pts[0], &cps[0],
        &knot_vector_u[0], &knot_vector_v[0],
        &u_vec[0], &v_vec[0],
        guess_grid_n,
    )