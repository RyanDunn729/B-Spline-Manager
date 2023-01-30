import numpy as np

from dataclasses import dataclass

# from openvsp_step_file_reader import read_stp

import scipy.sparse as sps
import matplotlib.pyplot as plt

from lsdo_geo.splines.basis_matrix_curve_py import get_basis_curve_matrix

class BSplineEntity:
    # name: str
    # shape : np.ndarray
    # control_points : np.ndarray

    def __init__(self, name, shape, control_points):
        self.name = name
        self.shape = shape
        self.control_points = control_points 


class Point(BSplineEntity):
    def __init__(self, name, shape, control_points):
        BSplineEntity.__init__(self, name, shape, control_points)

class BSplineCurve(BSplineEntity):
    def __init__(self, name, u_order, shape, control_points, u_knots):
        BSplineEntity.__init__(self, name, shape, control_points)
        self.u_order = u_order
        self.u_knots = u_knots

    def evaluate(self, u_vec):
        data = np.zeros(len(u_vec) * self.u_order)
        row_indices = np.zeros(len(u_vec) * self.u_order, np.int32)
        col_indices = np.zeros(len(u_vec) * self.u_order, np.int32)

        get_basis_curve_matrix(self.u_order, len(self.control_points), 0, u_vec, len(u_vec), data, row_indices, col_indices)
        basis0 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), len(self.control_points)))

        get_basis_curve_matrix(self.u_order, len(self.control_points), 1, u_vec, len(u_vec), data, row_indices, col_indices)
        basis1 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), len(self.control_points)))

        get_basis_curve_matrix(self.u_order, len(self.control_points), 2, u_vec, len(u_vec), data, row_indices, col_indices)
        basis2 = sps.csc_matrix((data, (row_indices, col_indices)), shape=(len(u_vec), len(self.control_points)))

        pts = basis0.dot(self.control_points)
        derivs1 = basis1.dot(self.control_points)
        derivs2 = basis2.dot(self.control_points)

        return pts, derivs1, derivs2

    def inverse_evaluate(self, pts):
        if pts.shape[-1] != 3:
            Exception('The shape of points needs to end in 3!')
        
        

class BSplineSurface(BSplineEntity):
    def __init__(self, name, u_order, v_order, shape, control_points, u_knots, v_knots):
        BSplineEntity.__init__(self, name, shape, control_points)#super
        self.u_order = u_order
        self.v_order = v_order
        self.u_knots = u_knots
        self.v_knots = v_knots

class BSplineVolume(BSplineEntity):
    def __init__(self, name, u_order, v_order, w_order, shape, control_points, u_knots, v_knots, w_knots):
        BSplineEntity.__init__(self, name, shape, control_points)
        self.u_order = u_order
        self.v_order = v_order
        self.w_order = w_order
        self.u_knots = u_knots
        self.v_knots = v_knots
        self.w_knots = w_knots


if __name__ == "__main__":
    pt1 = np.array([0., 1., 2.])
    pt2 = np.array([3., 4., 5.])
    pt3 = np.array([6., 7., 8.])
    pt4 = np.array([9., 10., 11.])
    crv1 = np.array([pt1, pt2])
    u_order = 1
    v_order = 1
    w_order = 2
    u_knots = np.array([0., 0., 1., 1.])
    v_knots = np.array([0., 0., 1., 1.])
    w_knots = np.array([0., 0., 0.5, 1., 1.])
    crv2 = np.array([pt3, pt4])
    srf1 = np.array([crv1, crv2])

    curve = BSplineCurve('crv1', u_order, np.array([2]), crv1, u_knots)
    print(curve)
    surface = BSplineSurface('srf1', u_order, v_order, np.array([2, 2]), srf1, u_knots, v_knots)
    print(surface)
    volume = BSplineVolume('vol1', u_order, v_order, w_order, np.array([3, 2, 2]), np.array([srf1, srf1, srf1]), u_knots, v_knots, w_knots)
    print(volume)
    surfs = read_stp('test_wing.stp')
    bspline_surfs = []
    for surf in surfs:
        shape = np.array([surf.nu, surf.nv])
        bspline_surfs.append(BSplineSurface(surf.name, surf.u_degree+1, surf.v_degree+1, shape, surf.cntrl_pts, surf.u_knots, surf.v_knots))

    print(bspline_surfs)
    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    for surf in bspline_surfs:
        num_control_points_u = surf.shape[0]
        num_control_points_v = surf.shape[1]
        plot_cntrl_pts = surf.control_points.reshape((num_control_points_u * num_control_points_v, 3))
        ax1.scatter(plot_cntrl_pts[:, 0], plot_cntrl_pts[:, 1], plot_cntrl_pts[:, 2], 'or')
    plt.show()


