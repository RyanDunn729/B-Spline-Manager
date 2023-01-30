import numpy as np

import scipy.sparse as sps
from lsdo_geo.cython.basis_matrix_curve_py import get_basis_curve_matrix
from lsdo_geo.cython.curve_projection_py import compute_curve_projection


class BSplineCurve:
    def __init__(self, name, control_points, order_u=4, knots_u=None):
        
        self.name = name
        self.order_u = order_u
        self.shape = control_points.shape
        self.control_points = control_points
        self.knots_u = knots_u

        if self.knots_u is None:
            self.knots_u = np.zeros(self.shape[0] + self.order_u)
            get_open_uniform(self.order_u, self.shape[0], self.knots_u)

    def compute_eval_map_points(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.control_points.shape[0], 0, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis0

    def compute_eval_map_derivs1(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.control_points.shape[0], 1, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis1

    def compute_eval_map_derivs2(self, u_vec):
        data = np.zeros(len(u_vec) * self.order_u)
        row_indices = np.zeros(len(data), np.int32)
        col_indices = np.zeros(len(data), np.int32)

        get_basis_curve_matrix(self.order_u, self.control_points.shape[0], 2, u_vec, self.knots_u, 
            len(u_vec), data, row_indices, col_indices)

        basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), self.num_control_points) )
        
        return basis2

    def evaluate_points(self, u_vec):
        basis0 = self.compute_eval_map_points(u_vec)
        points = basis0.dot(self.control_points.reshape((self.num_control_points, 3)))

        return points

    def evaluate_der1(self, u_vec):
        basis1 = self.compute_eval_map_der1(u_vec)
        derivs1 = basis1.dot(self.control_points.reshape((self.num_control_points, 3)))

        return derivs1 

    def evaluate_der2(self, u_vec):
        basis2 = self.compute_eval_map_der2(u_vec)
        derivs2 = basis2.dot(self.control_points.reshape((self.num_control_points, 3)))

        return derivs2


    def project(self, points_to_project, max_iter=100):

        num_control_points = self.control_points.shape[0] * self.control_points.shape[1]
        num_points = len(points_to_project)

        u_vec = np.zeros(num_points)

        compute_curve_projection(
            self.order_u, self.control_points.shape[0],
            num_points, max_iter,
            points_to_project, 
            self.control_points,
            u_vec, 50
        )

        return u_vec

    def compute_projection_eval_map(self, points_to_project, max_iter=100):
        u_vec = self.project(points_to_project, max_iter)

        basis0 = self.compute_eval_map(u_vec)
        
        return basis0