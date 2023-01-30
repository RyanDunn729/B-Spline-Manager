from lsdo_geo.cython.get_open_uniform_py import get_open_uniform
import numpy as np
import matplotlib.pyplot as plt
from vedo import Points, Plotter, LegendBox

from lsdo_geo.Geometry.geometry import Geometry 

import lsdo_geo
from lsdo_geo.bsplines.bspline_curve import BSplineCurve
from lsdo_geo.bsplines.bspline_surface import BSplineSurface
from lsdo_geo.bsplines.bspline_volume import BSplineVolume

from lsdo_geo.splines.basis_matrix_surface_py import get_basis_surface_matrix

#TODO: Consider making nu, nv, nw attributes of the class. Talk to professor about the axis of rotation, and the anchor.
class FFD(object):
    def __init__(self, name, control_points, embedded_entities_pointers: list, xprime, yprime, zprime, origin=None, shape=None, order_u=4, order_v=4, order_w=4,
                    knots_u=None, knots_v=None, knots_w=None):
        
        self.xprime = xprime
        self.yprime = yprime
        self.zprime = zprime

        self.nxp = control_points.shape[0]
        self.nyp = control_points.shape[1]
        self.nzp = control_points.shape[2]
        
        self.embedded_entities_pointers = embedded_entities_pointers
        
        if shape is None:
            shape = control_points.shape
        else:
            control_points = control_points.reshape(shape)

        if knots_u is None:
            knots_u = np.zeros(shape[0] + order_u)
            get_open_uniform(order_u, shape[0], knots_u)
        
        if knots_v is None:
            knots_v = np.zeros(shape[1] + order_v)
            get_open_uniform(order_v, shape[1], knots_v)

        if knots_w is None:
            knots_w = np.zeros(shape[2] + order_w)
            get_open_uniform(order_w, shape[2], knots_w)

        self._generate_embedded_coordinates()

        self.BSplineVolume = BSplineVolume(name, order_u, order_v, order_w, knots_u, knots_v, knots_w, shape, control_points)

        if origin == None:
            u_vec = np.linspace(0, 1, control_points.shape[0])
            v_vec = np.ones(control_points.shape[0]) * 0.5
            w_vec = np.ones(control_points.shape[0]) * 0.5

            self.origin = self.BSplineVolume.evaluate_points(u_vec, v_vec, w_vec)
        
        elif origin.shape == (control_points.shape[0], 3):
            self.origin = origin

        else:
            Exception('Origin shape must be equal to (control_points.shape[0], 3)')


        # print('self.BSplineVolume.control_points: ', self.BSplineVolume.control_points)
        # print('self.embedded_points: ', self.embedded_points)

    def _generate_exterior_points(self, nu, nv, nw):

        v_vec_front, w_vec_front = np.mgrid[0:1:nv*1j, 0:1:nw*1j]
        u_vec_front = np.zeros(v_vec_front.shape) 

        v_vec_back, w_vec_back = np.mgrid[0:1:nv*1j, 0:1:nw*1j]
        u_vec_back = np.ones(v_vec_front.shape)

        u_vec_bot, v_vec_bot = np.mgrid[0:1:nu*1j, 0:1:nv*1j]
        w_vec_bot = np.zeros(v_vec_bot.shape)

        u_vec_top, v_vec_top = np.mgrid[0:1:nu*1j, 0:1:nv*1j]
        w_vec_top = np.ones(v_vec_bot.shape)

        u_vec_left, w_vec_left = np.mgrid[0:1:nu*1j, 0:1:nw*1j]
        v_vec_left = np.zeros(u_vec_left.shape)

        u_vec_right, w_vec_right = np.mgrid[0:1:nu*1j, 0:1:nw*1j]
        v_vec_right = np.ones(u_vec_right.shape)

        u_points = np.concatenate((u_vec_front.flatten(), u_vec_back.flatten(), u_vec_bot.flatten(), u_vec_top.flatten(), u_vec_left.flatten(), u_vec_right.flatten()))
        v_points = np.concatenate((v_vec_front.flatten(), v_vec_back.flatten(), v_vec_bot.flatten(), v_vec_top.flatten(), v_vec_left.flatten(), v_vec_right.flatten()))
        w_points = np.concatenate((w_vec_front.flatten(), w_vec_back.flatten(), w_vec_bot.flatten(), w_vec_top.flatten(), w_vec_left.flatten(), w_vec_right.flatten()))
        
        exterior_points = self.BSplineVolume.evaluate_points(u_points, v_points, w_points)

        return exterior_points

    def _generate_embedded_coordinates(self):

        self.embedded_points = [] 

        for i in self.embedded_entities_pointers:
            # print('i: ', i)
            if isinstance(i, lsdo_geo.bsplines.bspline_curve.BSplineCurve) or isinstance(i, lsdo_geo.bsplines.bspline_surface.BSplineSurface) or isinstance(i, lsdo_geo.bsplines.bspline_volume.BSplineVolume):
                self.embedded_points.append(i.control_points)  # Control points are given in Cartesian Coordinates
            else:
                self.embedded_points.append(i.coordinates)  # Here i is a PointSet, make sure that the PointSet is evalauted

    def plot(self, nu, nv, nw):

        exterior_points = self._generate_exterior_points(nu, nv, nw)

        vp_init = Plotter()
        vps = []
        vps1 = Points(exterior_points, r=8, c = 'red')
        vps.append(vps1)

        for i in self.embedded_points:
            vps2 =  Points(i, r=8, c='blue')
            vps.append(vps2)

        vp_init.show(vps, 'Bspline Volume', axes=1, viewup="z", interactive = True)

    def project_points_FFD(self):
        self.ffd_application_map = self.BSplineVolume.compute_projection_eval_map(self.embedded_points)

        return self.ffd_application_map


    # def translate_control_points(self, offset):
    #     if offset.shape == (3,):
    #         temp = np.ones(self.BSplineVolume.control_points.shape)
    #         temp[:, :, :, 0] = temp[:, :, :, 0] * offset[0]
    #         temp[:, :, :, 1] = temp[:, :, :, 1] * offset[1]
    #         temp[:, :, :, 2] = temp[:, :, :, 2] * offset[2]

    #         self.BSplineVolume.control_points = self.BSplineVolume.control_points + temp

    #     elif self.BSplineVolume.control_points.shape == offset.shape:
    #         self.BSplineVolume.control_points = self.BSplineVolume.control_points + offset

    #     else:
    #         Exception('Shape of offset must be a (3,) array OR the same shape as control_points')

    # def scale_x(self, scaling_factor_vector, origin=None):
    #     if origin == None:
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 2] = self.BSplineVolume.control_points[:, :, :, 2] * scaling_factor_vector
            
    #         else len(scaling_factor_vector) == self.BSplineVolume.control_points.shape[0]:
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector

    #     elif origin.shape == (3,):
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points - origin
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 2] = self.BSplineVolume.control_points[:, :, :, 2] * scaling_factor_vector
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points + origin
        
    #     elif origin.shape == tuple(np.append(self.BSplineVolume.control_points[0], 3)):
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points - origin
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 2] = self.BSplineVolume.control_points[:, :, :, 2] * scaling_factor_vector
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points + origin
        



    # def scale_y(self, scaling_factor_vector, origin=None):
    #     if origin == None:
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 2] = self.BSplineVolume.control_points[:, :, :, 2] * scaling_factor_vector
        
    #     elif origin.shape == (3,):
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points - origin
    #             self.BSplineVolume.control_points[:, :, :, 0] = self.BSplineVolume.control_points[:, :, :, 0] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 2] = self.BSplineVolume.control_points[:, :, :, 2] * scaling_factor_vector
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points + origin
        
    # def scale_z(self, scaling_factor_vector, origin=None):
    #     if origin == None:
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points[:, :, :, 0] = self.BSplineVolume.control_points[:, :, :, 0] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
        
    #     elif origin.shape == (3,):
    #         if len(scaling_factor_vector) == 1:
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points - origin
    #             self.BSplineVolume.control_points[:, :, :, 0] = self.BSplineVolume.control_points[:, :, :, 0] * scaling_factor_vector
    #             self.BSplineVolume.control_points[:, :, :, 1] = self.BSplineVolume.control_points[:, :, :, 1] * scaling_factor_vector
    #             self.BSplineVolume.control_points = self.BSplineVolume.control_points + origin
        




if __name__ == "__main__":

    from lsdo_geo.utils.generate_ffd import create_ffd
    nxp = 5
    nyp = 5
    nzp = 5

    point000 = np.array([170. ,0. ,100.])
    point010 = np.array([130., 230., 100.])
    point001 = np.array([170., 0., 170.])
    point011 = np.array([130., 230., 170.])
    
    point100 = np.array([240. ,0. ,100.])
    point101 = np.array([240. ,0. ,170.])
    point110 = np.array([200. ,230. ,100.])
    point111 = np.array([200. ,230. ,170.])

    control_points = np.zeros((2,2,2,3))
    
    control_points[0,0,0,:] = point000
    control_points[0,0,1,:] = point001

    control_points[0,1,0,:] = point010
    control_points[0,1,1,:] = point011
    
    control_points[1,0,0,:] = point100
    control_points[1,0,1,:] = point101
    
    control_points[1,1,0,:] = point110
    control_points[1,1,1,:] = point111

    ffd_control_points = create_ffd(control_points, nxp, nyp, nzp)
    

    # front_bot_edge = np.linspace(point000, point010, nv)
    # back_bot_edge = np.linspace(point100, point110, nv)
    
    # front_top_edge = np.linspace(point001, point011, nv)
    # back_top_edge = np.linspace(point101, point111, nv)

    # front_surface = np.linspace(front_bot_edge, front_top_edge, nw)
    # back_surface = np.linspace(back_bot_edge, back_top_edge, nw)
    
    # control_points = np.linspace(front_surface, back_surface, nu)

    ''' Camber surface creation script for this case '''
    path_name = '../Geometry/CAD/'
    file_name = 'eVTOL.stp'
    geo = Geometry(path_name + file_name)

    wing_surface_names = [
    'Surf_WFWKRQIMCA, Wing, 0, 12', 'Surf_WFWKRQIMCA, Wing, 0, 13', 
    'Surf_WFWKRQIMCA, Wing, 0, 14', 'Surf_WFWKRQIMCA, Wing, 0, 15', 
    ]

    bspline_entities = [geo.input_bspline_entity_dict[wing_surface_names[0]],
       geo.input_bspline_entity_dict[wing_surface_names[1]], 
       geo.input_bspline_entity_dict[wing_surface_names[2]],
       geo.input_bspline_entity_dict[wing_surface_names[3]]]

    xprime = np.array([1, 0, 0])
    yprime = np.array([0, 1, 0])
    zprime = np.array([0, 0, 1])

    test_ffd = FFD('test', ffd_control_points, bspline_entities, xprime=xprime, yprime=yprime, zprime=zprime)

    # print(test_ffd.BSplineVolume.control_points)

    test_ffd.plot(nxp, nyp, nzp)

    # test_ffd.translate_control_points(offset=np.array([10., 50., 100.]))

    # print(test_ffd.BSplineVolume.control_points)

    # test_ffd.plot(nu, nv, nw)


