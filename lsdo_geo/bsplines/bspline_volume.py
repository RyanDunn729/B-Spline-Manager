import numpy as np

import scipy.sparse as sps
from lsdo_geo.cython.basis_matrix_volume_py import get_basis_volume_matrix
from lsdo_geo.cython.volume_projection_py import compute_volume_projection


class BSplineVolume:
    def __init__(self, name, order_u, order_v, order_w, knots_u, knots_v, knots_w, shape, control_points):
        self.name = name
        self.order_u = order_u
        self.knots_u = knots_u
        self.order_v = order_v
        self.knots_v = knots_v
        self.order_w = order_w
        self.knots_w = knots_w
        self.shape = shape
        self.control_points = control_points
        self.shape_u = int(shape[0])
        self.shape_v = int(shape[1])
        self.shape_w = int(shape[2])
        self.num_control_points = int(self.shape_u * self.shape_v * self.shape_w)


    def get_basis_matrix(self, u_vec, v_vec, w_vec, du, dv, dw):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            self.order_v, self.shape_v, dv, v_vec, self.knots_v,
            self.order_w, self.shape_w, dw, w_vec, self.knots_w, 
            len(u_vec), data, row_indices, col_indices
            )
        
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis


    def get_basis_matrix_indices(self, u_vec, v_vec, w_vec, du, dv, dw):
        data = np.zeros(len(u_vec) * self.order_u * self.order_v * self.order_w)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_volume_matrix(
            self.order_u, self.shape_u, du, u_vec, self.knots_u, 
            self.order_v, self.shape_v, dv, v_vec, self.knots_v,
            self.order_w, self.shape_w, dw, w_vec, self.knots_w, 
            len(u_vec), data, row_indices, col_indices
            )
        
        basis = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return row_indices, col_indices, basis


    def evaluate_points(self, u_vec, v_vec, w_vec):

        basis0 = self.get_basis_matrix(u_vec, v_vec, w_vec, 0, 0, 0)
        points = basis0.dot(self.control_points)

        return points


    def evaluate_der(self, u_vec, v_vec, w_vec, du, dv, dw):

        basis = self.get_basis_matrix(u_vec, v_vec, w_vec, du, dv, dw)
        derivs = basis.dot(self.control_points)

        return derivs


    def project(self, points_to_project, max_iter=100, guess_grid=0):

        num_points = len(points_to_project)
        
        u_vec = 0.5 * np.ones(num_points)
        v_vec = 0.5 * np.ones(num_points)
        w_vec = 0.5 * np.ones(num_points)

        compute_volume_projection(
            self.order_u, self.shape_u,
            self.order_v, self.shape_v,
            self.order_w, self.shape_w,
            num_points, max_iter,
            points_to_project.reshape(num_points * 3), 
            self.control_points.reshape(self.num_control_points * 4),
            self.knots_u, self.knots_v, self.knots_w,
            u_vec, v_vec, w_vec, guess_grid
        )

        return u_vec, v_vec, w_vec

    def compute_projection_eval_map(self, points_to_project, max_iter=100):
        u_vec, v_vec, w_vec = self.project(points_to_project, max_iter)

        basis0 = self.compute_eval_map(u_vec, v_vec, w_vec)
        
        return basis0

