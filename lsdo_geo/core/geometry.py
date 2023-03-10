import numpy as np
from numpy.core.fromnumeric import ndim
from numpy.core.shape_base import vstack
from scipy import sparse, interpolate
import scipy.sparse as sps
import pandas as pd
#from pandas.core.indexes.base import ensure_index
from dataclasses import dataclass
from toposort import toposort
import re
from typing import List
import copy
from pyoctree import pyoctree as ot

#import pyiges

from geomdl import fitting

# from lsdo_geo.Geometry.bspline_entities import BSplineCurve, BSplineSurface, BSplineVolume, BSplineEntity
from lsdo_geo.bsplines.bspline_curve import BSplineCurve
from lsdo_geo.bsplines.bspline_surface import BSplineSurface
from lsdo_geo.bsplines.bspline_volume import BSplineVolume
from lsdo_geo.utils.step_io import read_openvsp_stp, write_step
from lsdo_geo.utils.iges_io import read_iges, write_iges

# from lsdo_geo.splines.basis_matrix_surface_py import get_basis_surface_matrix
# from lsdo_geo.splines.surface_projection_py import compute_surface_projection
# from lsdo_geo.splines.get_open_uniform_py import get_open_uniform

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.surface_projection_py import compute_surface_projection
from lsdo_geo.cython.get_open_uniform_py import get_open_uniform

import matplotlib.pyplot as plt
from vedo import Points, Plotter, colors, LegendBox, show
import vedo # for using Mesh

class Geometry:

    def __init__(self, file_name=None, plot=False):
        self.input_bspline_entity_dict = {}
        self.initial_input_bspline_entity_dict = {}
        self.total_cntrl_pts_vector = np.array([])
        self.cntrl_pts_unique = np.array([])

        if file_name != None:
            if ((file_name[-4:].lower() == '.stp') or (file_name[-5:].lower() == '.step')):
                self.read_openvsp_stp(file_name)
            elif ((file_name[-4:].lower() == '.igs') or (file_name[-5:].lower() == '.iges')):
                print('warning, not implemented yet!!')
                self.read_iges(file_name) #TODO
            else:
                print("Please input an iges file or a stp file from openvsp.")

        self.registered_names = [] 
        self.current_id = 0
        self.parent_pointers_dict = {}
        self.pointsets_dict = {}
        self.eval_map = None
        self.output_bspline_entity_dict = self.initial_input_bspline_entity_dict
        self.evaluated_pointsets = Geometry.EvaluatedPointSets(
            pointset_id = self.current_id,
            shape = None,
            parent_pointers = [],
            absolute_map = None,
            relative_map = None,
            offset = np.array([0., 0., 0.]),            
            offset_absolute_map = None,
            physical_coordinates = None,
            )
        self.current_id += 1
        self.mesh_list = {}
        self.bspline_mesh_list = {}

        if plot == True:
            vp_init = Plotter()#TODO legend
            vps = []
            for surf, color in zip(self.input_bspline_entity_dict.values(), colors.colors.values()):
                vps.append(Points(surf.control_points, r=8, c = color).legend(surf.name))
            vp_init.show(vps, 'Control points', axes=1, viewup="z", interactive = False)
            vp_init_out = Plotter()#TODO legend
            vps = []
            for surf, color in zip(self.initial_input_bspline_entity_dict.values(), colors.colors.values()):
                vps.append(Points(surf.control_points, r=8, c = color).legend(surf.name))
            vp_init_out.show(vps, 'Control points', axes=1, viewup="z", interactive = True)

    def project_points(self, points_to_be_projected, projection_targets_names=[], projection_direction=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.]), plot=False):
        if len(np.shape(points_to_be_projected))== 1:
            shape = 1 
            num_points = 1
        elif len(np.shape(points_to_be_projected))== 2:
            shape = np.shape(points_to_be_projected)[:-1]
            num_points = np.shape(points_to_be_projected)[0]
        else:
            shape = np.shape(points_to_be_projected)[:-1]
            num_points = np.shape(points_to_be_projected)[0]*np.shape(points_to_be_projected)[1]

        if len(np.shape(projection_direction)) == 1:
            projection_direction = np.array((projection_direction),ndmin=2)
            projection_direction = np.repeat(projection_direction,num_points,axis = 0)
        
        if projection_targets_names == []:#if no input target is passed
            projection_targets = self.input_bspline_entity_dict.values()
        else:
            projection_targets = []
            for target in projection_targets_names:# create matrix of projection target identifiers
                projection_targets.append(self.input_bspline_entity_dict[target])
        
        relative_map = np.zeros((num_points,len(self.total_cntrl_pts_vector)))
        temp = 1e16 * np.ones((num_points,))
        surf_final = [None] * num_points
        cps_final = [None] * num_points
        linear_map_final = [None] * num_points

        surfs_u_order = np.empty((0,1),dtype = int)
        surfs_v_order = np.empty((0,1),dtype = int)
        surfs_num_control_points_u = np.empty((0,1),dtype = int)
        surfs_num_control_points_v = np.empty((0,1),dtype = int)
        surfs_cp = np.empty((0,projection_targets[0].shape[0]*projection_targets[0].shape[0]*3),dtype = int)
        for surf in projection_targets: 
            surfs_u_order = np.append(surfs_u_order,np.array((surf.order_u),ndmin=2),axis = 0)
            surfs_v_order = np.append(surfs_v_order,np.array((surf.order_v),ndmin=2),axis = 0)
            surfs_num_control_points_u = np.append(surfs_num_control_points_u,np.array(surf.shape[0], ndmin=2),axis = 0)
            surfs_num_control_points_v = np.append(surfs_num_control_points_v,np.array(surf.shape[1], ndmin=2),axis = 0)
            surfs_cp = np.append(surfs_cp,surf.control_points.reshape((1,projection_targets[0].shape[0]*projection_targets[0].shape[0]*3)),axis = 0)

        points = points_to_be_projected
        axis = projection_direction
        for surf in projection_targets:  
            num_control_points_u = surf.shape[0]
            num_control_points_v = surf.shape[1]
            u_order = surf.order_u
            v_order = surf.order_v
            knot_vector_u = surf.knots_u
            knot_vector_v = surf.knots_v
            cps = surf.control_points
            max_iter = 500
            u_vec = np.ones(num_points)
            v_vec = np.ones(num_points)
            surfs_index = np.zeros(num_points,dtype=int)
        compute_surface_projection(
            surfs_u_order.reshape((len(projection_targets))), surfs_num_control_points_u.reshape((len(projection_targets))),
            surfs_v_order.reshape((len(projection_targets))), surfs_num_control_points_v.reshape((len(projection_targets))),
            num_points, max_iter,
            points.reshape(num_points * 3), 
            cps.reshape((num_control_points_u * num_control_points_v * 3)),
            knot_vector_u, knot_vector_v,
            u_vec, v_vec, 50,
            axis.reshape(num_points * 3),
            surfs_index,
            surfs_cp,
        )
        #print('u_vec',u_vec)
        #print('v_vec',v_vec)
        #print('surfs_index', surfs_index)

        for ns,surf in zip(range(len(projection_targets)),projection_targets):
            num_control_points_u = surf.shape[0]
            num_control_points_v = surf.shape[1]
            u_order = surf.order_u
            v_order = surf.order_v
            knot_vector_u = surf.knots_u
            knot_vector_v = surf.knots_v
            cps = surf.control_points
            nnz = num_points * u_order * v_order
            data = np.zeros(nnz)
            row_indices = np.zeros(nnz, np.int32)
            col_indices = np.zeros(nnz, np.int32)
            get_basis_surface_matrix(
                u_order, num_control_points_u, 0, u_vec, knot_vector_u, 
                v_order, num_control_points_v, 0, v_vec, knot_vector_v,
                num_points, data, row_indices, col_indices,
            )
            basis0 = sps.csc_matrix(
                (data, (row_indices, col_indices)), 
                shape=(num_points, num_control_points_u * num_control_points_v),
            )           
            pts = basis0.dot(cps)
            pts = np.reshape(pts, (num_points, 3))
            points = np.reshape(points, (num_points, 3))
            linear_map = basis0.dot(np.identity(surf.shape[0] * surf.shape[1]))
            for i in range(np.shape(projection_direction)[0]):
                if surfs_index[i] == ns:
                    surf_final[i] = surf
                    linear_map_final[i] = linear_map[i,:]
                    cps_final[i] = cps
                    
        for i in range(np.shape(projection_direction)[0]):
            j = 0
            for surf in self.input_bspline_entity_dict.values():
                if surf == surf_final[i]:
                    relative_map[i, j:j+surf.shape[0]*surf.shape[1]] = linear_map_final[i]
                j = j + surf.shape[0]*surf.shape[1] 
        #  - Want to directly build sparse basis matrix. Must starting index of surface projection target
        #     - For now, we can loop over the shapes of the surfaces to find the starting index of the proj target
        #       - In the future, we could probably use Anugrah's Vector Class (Array Manager)
        relative_map = sps.csc_matrix(relative_map)
        pointset = Geometry.ProjectedPointSet(
            pointset_id = self.current_id,
            shape = np.append(shape,3),
            parent_pointers = [],
            absolute_map = None,
            relative_map = relative_map,
            offset = offset,            
            offset_absolute_map = None,
            physical_coordinates = None,
            )
        self.pointsets_dict[self.current_id] = pointset
        self.current_id += 1
        offset = np.array(offset,ndmin=2)
        offset = np.repeat(offset,np.shape(projection_direction)[0],axis = 0)
        point_test = relative_map.dot(self.total_cntrl_pts_vector) + offset#test
        if plot == True: 
            cps1 = [] 
            cps2 = [] 
            point_test = relative_map.dot(self.total_cntrl_pts_vector) + offset
            for surf, cps in zip(surf_final, cps_final):
                cps1.append(Points(surf.control_points, r=8, c = 'b', alpha = 0.5).legend('Contol points of projected surface'))
                cps2.append(Points(cps.reshape((num_control_points_u * num_control_points_v, 3)), r=8, c='b', alpha = 0.5).legend('Interpolated contol points of projected surface'))
            projecting_point = Points(points.reshape((num_points, 3)), r=15, c='g').legend('Projecting curve')
            projected_point = Points(point_test.reshape((num_points, 3)), r=15, c='r').legend('Projected curve')
            vp_project_curve = Plotter(N=3, axes=1) #TODO legend
            vp_project_curve.show(cps1, 'Control points of surface to be projected', at=0, viewup="z")#, lb1
            vp_project_curve.show(cps2, projecting_point, 'Surface + projecting curve', at=1, viewup="z")#, lb2
            vp_project_curve.show(cps2, projected_point, 'Surface + projected curve', at=2, viewup="z",interactive=False)#, lb3
        return pointset, point_test


    def perform_linear_combination(self, parent_pointset_list, relative_map, shape, offset=np.array([0., 0., 0.])):
        """ 
        Perform any arbitrary linear combination between PointSets and creates a new PointSet

        Parameters
        ----------
        parent_pointset_list : list
            A list of the parent poinsets in the order they are used in the relative map
        relative_map : csc_matrix     
            A sprase matrix defining the linear combination s.t. the result is a flattened shape (n,3)
        shape : np.ndarray            
            A numpy array defining the desired unflattened shape of the output
        offset : np.ndarray           
            A numpy array of shape (n,3) defining how much to shift each point

        Returns
        -------
        pointset : PointSet           
            A PointSet object that is the result of the linear combination
        """
        parent_pointers = []
        for pointset in parent_pointset_list:
            parent_pointers.append(pointset.pointset_id)
            

        pointset = Geometry.DerivedPointSet(
            pointset_id = self.current_id,
            shape = shape,
            parent_pointers = parent_pointers,
            relative_map = relative_map,
            absolute_map = None,
            offset = offset,            
            offset_absolute_map = None,
            physical_coordinates = None,
            )
        self.pointsets_dict[self.current_id] = pointset
        self.current_id += 1
        return pointset
        
    def perform_linear_interpolation(self, pointset_start, pointset_end, shape, output_parameters=np.array([0]), offset=np.array([0., 0., 0.])):       
        dims_dont_match = np.array(np.shape(pointset_start) != np.shape(pointset_end), ndmin = 1)
        if any(dims_dont_match): #pset1 and pset2 must have same number of points
            print('The sets you are trying to interpolate do not have the same dimensions.\n')
            return
        if all(output_parameters == 0):
            zeros_array = np.zeros(np.shape(pointset_start)[:-1])
            ones_array = np.ones(np.shape(pointset_start)[:-1])
            output_parameters = np.linspace(zeros_array,ones_array, shape[-1], axis = -1)
        num_parent_points = int(np.prod(np.shape(pointset_start)[:-1]))#points in single parent set
        num_interpolated_points = np.prod(shape)
        output_parameters = np.reshape(output_parameters, [num_parent_points,shape[-1]])
        relative_map = np.zeros([num_interpolated_points,2*num_parent_points])# (number of interpolation points, number of points in pointsets)
        for i in range(0,num_interpolated_points):
            relative_map[i,(i//shape[-1])] = 1 - output_parameters[(i//shape[-1]),(i%shape[-1])]
                #(i//shape[-1]) = 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6 for shape = [5,7]
                #(i%shape[-1])  = 0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6 for shape = [5,7]
            relative_map[i,(i//shape[-1])+num_parent_points] = output_parameters[(i//shape[-1]),(i%shape[-1])]
        relative_map = sps.csc_matrix(relative_map)
        
        shape = np.append(shape,3)

        parent_pointset_list = [pointset_start, pointset_end]
        
        pointset = self.perform_linear_combination(parent_pointset_list, relative_map, shape, offset)
        return pointset

    def perform_bilinear_interpolation(self, point_00, point_10, point_01, point_11, shape, output_parameters=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.])):
        if output_parameters.all() == 0:
            x = np.linspace(0,1,shape[0])
            y = np.linspace(0,1,shape[1])
            u, v = np.meshgrid(x, y, indexing='ij')
            output_parameters = np.stack((u,v),axis=2)
            shape = (np.shape(output_parameters)[0],np.shape(output_parameters)[1],3)
            relative_map = np.zeros((np.shape(output_parameters)[0]*np.shape(output_parameters)[1],4))
            u = np.reshape(output_parameters[:,:,0],(np.shape(output_parameters)[0]*np.shape(output_parameters)[1],1))
            v = np.reshape(output_parameters[:,:,1],(np.shape(output_parameters)[0]*np.shape(output_parameters)[1],1))
        else:
            if np.ndim(output_parameters) == 3:
                shape = (np.shape(output_parameters)[0],np.shape(output_parameters)[1],3)
                relative_map = np.zeros((np.shape(output_parameters)[0]*np.shape(output_parameters)[1],4))
                u = np.reshape(output_parameters[:,:,0],(np.shape(output_parameters)[0]*np.shape(output_parameters)[1],1))
                v = np.reshape(output_parameters[:,:,1],(np.shape(output_parameters)[0]*np.shape(output_parameters)[1],1))
            elif np.ndim(output_parameters) == 2: 
                shape = (np.shape(output_parameters)[0],3)
                relative_map = np.zeros((np.shape(output_parameters)[0],4))
                u = np.reshape(output_parameters[:,0],(np.shape(output_parameters)[0],1))
                v = np.reshape(output_parameters[:,1],(np.shape(output_parameters)[0],1))
            elif np.ndim(output_parameters) == 1:
                shape =(1,3)
                relative_map = np.zeros((1,4))
                u = np.reshape(output_parameters[0],(1,1))
                v = np.reshape(output_parameters[1],(1,1))
            else:
                print('Warning: Wrong dimension of output_parameters')

        for n in range(len(u)):
            relative_map[n,:] = np.array([(1-u[n])*(1-v[n]), (1-v[n])*u[n], (1-u[n])*v[n], u[n]*v[n]])[:,0]
        relative_map = sps.csc_matrix(relative_map)

        parent_pointset_list = [point_00, point_10, point_01, point_11]

        pointset = self.perform_linear_combination(parent_pointset_list, relative_map, shape, offset)
        return pointset

    def perform_trilinear_interpolation(self, point_000, point_100, point_010, point_110, point_001, point_101, point_011, point_111, 
        shape, output_parameters=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.])):        
        if output_parameters.all() == 0:
            x = np.linspace(0,1,shape)
            y = np.linspace(0,1,shape)
            z = np.linspace(0,1,shape)
            u, v, w = np.meshgrid(x, y, z, indexing='ij')
            output_parameters = np.stack((u,v,w),axis=3)
            shape = (np.shape(output_parameters)[0],np.shape(output_parameters)[1],np.shape(output_parameters)[2],3)
            sl_shape = np.shape(output_parameters)[0]*np.shape(output_parameters)[1]*np.shape(output_parameters)[2]
            relative_map = np.zeros((sl_shape,8))
            u = np.reshape(output_parameters[:,:,:,0],(sl_shape,1))
            v = np.reshape(output_parameters[:,:,:,1],(sl_shape,1))
            w = np.reshape(output_parameters[:,:,:,2],(sl_shape,1))
        else:
            if np.ndim(output_parameters) == 4:
                shape = (np.shape(output_parameters)[0],np.shape(output_parameters)[1],np.shape(output_parameters)[2],3)
                sl_shape = np.shape(output_parameters)[0]*np.shape(output_parameters)[1]*np.shape(output_parameters)[2]
                relative_map = np.zeros((sl_shape,8))
                u = np.reshape(output_parameters[:,:,:,0],(sl_shape,1))
                v = np.reshape(output_parameters[:,:,:,1],(sl_shape,1))
                w = np.reshape(output_parameters[:,:,:,2],(sl_shape,1))
            elif np.ndim(output_parameters) == 3:
                shape = (np.shape(output_parameters)[0],np.shape(output_parameters)[1],3)
                sl_shape = np.shape(output_parameters)[0]*np.shape(output_parameters)[1]*np.shape(output_parameters)[2]
                relative_map = np.zeros((sl_shape,8))
                u = np.reshape(output_parameters[:,:,:,0],(sl_shape,1))
                v = np.reshape(output_parameters[:,:,:,1],(sl_shape,1))
                w = np.reshape(output_parameters[:,:,:,2],(sl_shape,1))
            elif np.ndim(output_parameters) == 2:
                shape = (np.shape(output_parameters)[0],3)
                relative_map = np.zeros((np.shape(output_parameters)[0],8))
                u = np.reshape(output_parameters[:,0],(np.shape(output_parameters)[0],1))
                v = np.reshape(output_parameters[:,1],(np.shape(output_parameters)[0],1))
                w = np.reshape(output_parameters[:,2],(np.shape(output_parameters)[0],1))
            elif np.ndim(output_parameters) == 1:
                shape =(1,3)
                relative_map = np.zeros((1,8))
                u = np.reshape(output_parameters[0],(1,1))
                v = np.reshape(output_parameters[1],(1,1))
                w = np.reshape(output_parameters[2],(1,1))
            else:
                print('Warning: Wrong dimension of output_parameters')
        
        for n in range(len(u)):
            relative_map[n,:] = np.array([(1-u[n])*(1-v[n])*(1-w[n]), u[n]*(1-v[n])*(1-w[n]), v[n]*(1-u[n])*(1-w[n]),u[n]*v[n]*(1-w[n]), w[n]*(1-u[n])*(1-v[n]), u[n]*w[n]*(1-v[n]), v[n]*w[n]*(1-u[n]), u[n]*v[n]*w[n]])[:,0]
        relative_map = sps.csc_matrix(relative_map)

        parent_pointers = []
        parent_pointers = np.append(parent_pointers, point_000.pointset_id)
        parent_pointers = np.append(parent_pointers, point_100.pointset_id)
        parent_pointers = np.append(parent_pointers, point_010.pointset_id)
        parent_pointers = np.append(parent_pointers, point_110.pointset_id)
        parent_pointers = np.append(parent_pointers, point_001.pointset_id)
        parent_pointers = np.append(parent_pointers, point_101.pointset_id)
        parent_pointers = np.append(parent_pointers, point_011.pointset_id)
        parent_pointers = np.append(parent_pointers, point_111.pointset_id)

        parent_pointset_list = [point_000, point_100, point_010, point_110, point_001, point_101, point_011, point_111]

        pointset = self.perform_linear_combination(parent_pointset_list, relative_map, shape, offset)
        return pointset

    def perform_2d_transfinite_interpolation(self, u_curve0, u_curve1, v_curve0, v_curve1, output_parameters=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.])):
        nu = u_curve0.shape[0]
        nv = v_curve0.shape[0]
        num_input = 2*(nu+nv)
        x = np.linspace(0,1,nu)
        y = np.linspace(0,1,nv)
        u, v = np.meshgrid(x, y, indexing='ij')
        output_parameters_all = np.stack((u,v), axis =2)
        relative_map = np.zeros((nu*nv, num_input))
        u = np.reshape(output_parameters_all[:,:,0],(nu*nv,1))
        v = np.reshape(output_parameters_all[:,:,1],(nu*nv,1))
        n = 0
        for i in range(nu):
            for j in range(nv):
                relative_map[n,i] = 1-v[n]
                relative_map[n,i+nu] = v[n]
                relative_map[n,j+2*nu] = 1-u[n]
                relative_map[n,j+2*nu+nv] = u[n]
                relative_map[n,0] = relative_map[n,0] - (1-u[n])*(1-v[n])
                relative_map[n,2*nu-1] = relative_map[n,2*nu-1] - u[n]*v[n]
                relative_map[n,nu-1] = relative_map[n,nu-1] - u[n]*(1-v[n])
                relative_map[n,nu] = relative_map[n,nu] - v[n]*(1-u[n])
                n = n+1
        if output_parameters.all() == 0:
            shape = (u_curve0.shape[0], v_curve0.shape[0], 3)
        else:
            uv = np.append(u, v, axis = 1)
            b = np.where((uv[:,0] == output_parameters[0,0]) & (uv[:,1] == output_parameters[0,1]))
            relative_map_part = relative_map[b,:]
            for n in range(1, np.shape(output_parameters)[0]):
                b = np.where((uv[:,0] == output_parameters[n,0]) & (uv[:,1] == output_parameters[n,1]))
                relative_map_part = np.vstack((relative_map_part, relative_map[b,:]))
            relative_map = np.reshape(relative_map_part, (np.shape(output_parameters)[0],np.shape(relative_map)[1]))

        relative_map = sps.csc_matrix(relative_map)

        parent_pointset_list = [u_curve0, u_curve1, v_curve0, v_curve1]

        pointset = self.perform_linear_combination(parent_pointset_list, relative_map, shape, offset)
        return pointset

    def perform_3d_transfinite_interpolation(self, output_parameters=None, offset=np.array([0., 0., 0.])):
        pass

    def extract_pointset(self, parent_pointset, point_indices, shape, offset=np.array([0., 0., 0.])):
        shape = np.append(shape, 3)
        relative_map = np.zeros((len(point_indices), parent_pointset.relative_map.shape[0]))
        for index in point_indices:
            relative_map[index,index] = 1.
        relative_map = sps.csc_matrix(relative_map)
        
        parent_pointset_list = [parent_pointset]
        
        pointset = self.perform_linear_combination(parent_pointset_list, relative_map, shape, offset)
        return pointset


    def register_output(self, pointset, name=None, mesh_names=[], bspline_mesh_names=[]):
        '''
        Create/add on to the single instance of evaluated_pointsets
        '''
        if name is None:
            name = f'pointset_{pointset.id}'

        self.evaluated_pointsets.parent_pointers.append(pointset.pointset_id)
        self.registered_names.append(name)

        for mesh_name in mesh_names:
            self.mesh_list[mesh_name].register_output(pointset, name)
        for bspline_mesh_name in bspline_mesh_names:
            self.bspline_mesh_list[bspline_mesh_name].register_output(pointset, name)

    def evaluate_all_absolute_maps(self):
        '''
            construct appropriate DAG representation
            call a topological sorting algorithm to get a topological ordering
            call the pointsets in this order
            ask each to evaluate its absolute maps
        '''
        ordered_pointers = list(toposort(self.parent_pointers_dict))
        for set in ordered_pointers:
            for id in set:
                self.evaluate_my_absolute_map(self.pointsets_dict[id])

    # method in PointSet
    def evaluate_my_absolute_map(self):
        '''
        calculates absolute map using source PointSetss' abssolute maps
        '''

    def evaluate_absolute_map(self, point_set):
        if point_set.absolute_map != None:
            return
        else:
            input_map = None
            offset_input_map = None
            for parent_id in point_set.parent_pointers:
                # calc absolute map of parent pointsets if they aren't already calculated
                self.evaluate_absolute_map(self.pointsets_dict[parent_id])
                # self.evaluate_absolute_map(self.pointsets_dict[f'{parent_id}'])

                # build input map
                if input_map == None:
                    input_map = self.pointsets_dict[parent_id].absolute_map
                    offset_input_map = self.pointsets_dict[parent_id].offset_absolute_map
                    # input_map = self.pointsets_dict[f'{parent_id}'].absolute_map
                else:
                    input_map = sps.vstack((input_map, self.pointsets_dict[parent_id].absolute_map))
                    offset_input_map = np.vstack((offset_input_map, self.pointsets_dict[parent_id].offset_absolute_map))
                    # input_map = np.vstack(input_map, self.pointsets_dict[f'{parent_id}'].absolute_map)
            if input_map == None:
                input_map = sps.eye(point_set.relative_map.shape[-1])    # assuming shape is (num_outputs, num_inputs) where we want columns
                offset_input_map = np.zeros((point_set.relative_map.shape[-1], 3))
            if point_set.pointset_id == 0:
                point_set.relative_map = sps.eye(input_map.shape[0])
            if point_set.offset.shape[0] == 1:
                point_set.offset = np.tile(point_set.offset, [point_set.relative_map.shape[0], 1])
            point_set.offset = point_set.offset.reshape(np.prod(point_set.offset.shape)//3, 3)

            # print(input_map.shape)
            # print(point_set.relative_map.shape)
            # calc absolute map of this point_set
            #print(point_set.relative_map.shape)
            #print(input_map.shape)
            point_set.absolute_map = point_set.relative_map.dot(input_map)
            point_set.offset_absolute_map = np.dot(point_set.relative_map.todense(), offset_input_map) + point_set.offset
            return


    def assemble(self, pointset = None):#first evalution 
        '''
        Concatenates vertically all matrices
        '''
        if pointset == None:
            self.evaluate_absolute_map(self.evaluated_pointsets)
            self.eval_map = self.evaluated_pointsets.absolute_map
            self.offset_eval_map = self.evaluated_pointsets.offset_absolute_map
        else:
            self.evaluate_absolute_map(pointset)
        return

    def evaluate(self,pointset = None):
        '''concatenates vertically all matrices'''
        if pointset == None:
            points = self.eval_map.dot(self.total_cntrl_pts_vector)# + self.offset_eval_map 
            points += self.offset_eval_map
        else:
            points = pointset.absolute_map.dot(self.total_cntrl_pts_vector)
            points += pointset.offset_absolute_map
        return points

    def fit_bspline_entities(self, points):
        '''Least square fit b-spline surface'''
        entity_starting_point = 0
        i = 0
        for pointer_id in self.evaluated_pointsets.parent_pointers:
            component_shape = self.pointsets_dict[pointer_id].shape
            #print(component_shape)
            #print(component_shape[:-1])
            #print(entity_starting_point)
            if len(component_shape[:-1]) == 0:  # is point
                print('fitting points has not been implemented yet')
                pass        #is point
            elif len(component_shape[:-1]) == 1:  # is curve
                if i==0:
                    entity_points_be_fitted = points[0:component_shape[0],:]
                else:
                    entity_points_be_fitted = points[entity_starting_point:(entity_starting_point+component_shape[0]),:]
                entity_starting_point += component_shape[0]
                curve = fitting.approximate_curve(entity_points_be_fitted, 3)   #TODO hardcoded cubic bspline
                bspline_entity_curve = BSplineCurve(
                    name=self.registered_names[i],
                    order_u=curve.order,
                    shape=np.array[len(curve.ctrlpts), 3],
                    control_points=curve.ctrlpts,
                    knots_u=curve.knotvector)
                self.output_bspline_entity_dict[self.registered_names[i]] = bspline_entity_curve
            elif len(component_shape[:-1]) == 2:  # is surface
                if i==0:
                    num_pts = component_shape[0] * component_shape[1]
                    entity_points_be_fitted = points[0:num_pts,:]
                else:
                    num_pts = component_shape[0] * component_shape[1]
                    entity_points_be_fitted = points[entity_starting_point:(entity_starting_point+num_pts),:]
                entity_starting_point += num_pts
                order_u_fitted = 4
                order_v_fitted = 4
                num_control_points_u_fitted = component_shape[0] - 1
                num_control_points_v_fitted = component_shape[1] - 1
                num_points_u_fitted = component_shape[0]
                num_points_v_fitted = component_shape[1]

                nnz = num_points_u_fitted * num_points_v_fitted * order_u_fitted * order_v_fitted
                data = np.zeros(nnz)
                row_indices = np.zeros(nnz, np.int32)
                col_indices = np.zeros(nnz, np.int32)
                u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u_fitted), np.ones(num_points_v_fitted)).flatten()
                v_vec = np.einsum('i,j->ij', np.ones(num_points_u_fitted), np.linspace(0., 1., num_points_v_fitted)).flatten()
                knot_vector_u = np.zeros(num_control_points_u_fitted+order_u_fitted)
                knot_vector_v = np.zeros(num_control_points_v_fitted+order_v_fitted)
                get_open_uniform(order_u_fitted, num_control_points_u_fitted, knot_vector_u)
                get_open_uniform(order_v_fitted, num_control_points_v_fitted, knot_vector_v)
                get_basis_surface_matrix(
                    order_u_fitted, num_control_points_u_fitted, 0, u_vec, knot_vector_u,
                    order_v_fitted, num_control_points_v_fitted, 0, v_vec, knot_vector_v,
                    num_points_u_fitted * num_points_v_fitted, data, row_indices, col_indices,
                )
                basis0 = sps.csc_matrix(
                    (data, (row_indices, col_indices)), 
                    shape=(num_points_u_fitted * num_points_v_fitted, num_control_points_u_fitted * num_control_points_v_fitted),
                )

                a = np.matmul(basis0.toarray().T, basis0.toarray())
                if np.linalg.det(a) == 0:
                    cps_fitted,_,_,_ = np.linalg.lstsq(a, np.matmul(basis0.toarray().T, entity_points_be_fitted), rcond=None)
                else: 
                    cps_fitted = np.linalg.solve(a, np.matmul(basis0.toarray().T, entity_points_be_fitted))            
            
                bspline_entity_surface = BSplineSurface(
                    name=self.registered_names[i],
                    order_u=order_u_fitted,
                    order_v=order_v_fitted,
                    shape=np.array([num_control_points_u_fitted, num_control_points_v_fitted, 3]),
                    control_points=np.array(cps_fitted).reshape((num_control_points_u_fitted*num_control_points_v_fitted,3)),
                    knots_u=np.array(knot_vector_u),
                    knots_v=np.array(knot_vector_v))
                self.output_bspline_entity_dict[self.registered_names[i]] = bspline_entity_surface

            elif len(component_shape[:-1]) == 3:  # is volume
                print('fitting BSplineVolume has not been implemented yet')
                pass
            i += 1
        pass 


    def remove_multiplicity(self, bspline_entity):
        # TODO allow it to be curves or volumes too
        component_shape = np.array(bspline_entity.shape)
        if len(component_shape[:-1]) == 0:  # is point
            print('fitting points has not been implemented yet')
            pass        #is point
        elif len(component_shape[:-1]) == 1:  # is curve
            print('fitting curves has not been implemented yet')
            pass 
        elif len(component_shape[:-1]) == 2: 
            order_u = bspline_entity.order_u
            order_v = bspline_entity.order_v
            num_control_points_u = bspline_entity.shape[0]
            num_control_points_v = bspline_entity.shape[1]
            num_points_u = 20   # TODO might want to pass these in as input
            num_points_v = 20

            nnz = num_points_u * num_points_v * order_u * order_v
            data = np.zeros(nnz)
            row_indices = np.zeros(nnz, np.int32)
            col_indices = np.zeros(nnz, np.int32)

            knot_vector_u = bspline_entity.knots_u
            knot_vector_v = bspline_entity.knots_v
            u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
            v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()

            get_basis_surface_matrix(
                order_u, num_control_points_u, 0, u_vec, knot_vector_u,
                order_v, num_control_points_v, 0, v_vec, knot_vector_v,
                num_points_u * num_points_v, data, row_indices, col_indices,
            )

            basis0 = sps.csc_matrix(
                (data, (row_indices, col_indices)),
                shape=(num_points_u * num_points_v, num_control_points_u * num_control_points_v),
            )

            pts = basis0.dot(bspline_entity.control_points)
            #print(pts.shape)
            order_u_fitted = 4
            order_v_fitted = 4
            num_control_points_u_fitted = 15
            num_control_points_v_fitted = 15
            num_points_u_fitted = num_points_u
            num_points_v_fitted = num_points_v

            nnz = num_points_u_fitted * num_points_v_fitted * order_u_fitted * order_v_fitted
            data = np.zeros(nnz)
            row_indices = np.zeros(nnz, np.int32)
            col_indices = np.zeros(nnz, np.int32)
            u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u_fitted), np.ones(num_points_v_fitted)).flatten()
            v_vec = np.einsum('i,j->ij', np.ones(num_points_u_fitted), np.linspace(0., 1., num_points_v_fitted)).flatten()
            knot_vector_u = np.zeros(num_control_points_u_fitted+order_u_fitted)
            knot_vector_v = np.zeros(num_control_points_v_fitted+order_v_fitted)
            get_open_uniform(order_u_fitted, num_control_points_u_fitted, knot_vector_u)
            get_open_uniform(order_v_fitted, num_control_points_v_fitted, knot_vector_v)
            get_basis_surface_matrix(
                order_u_fitted, num_control_points_u_fitted, 0, u_vec, knot_vector_u,
                order_v_fitted, num_control_points_v_fitted, 0, v_vec, knot_vector_v,
                num_points_u_fitted * num_points_v_fitted, data, row_indices, col_indices,
            )
            basis0 = sps.csc_matrix(
                (data, (row_indices, col_indices)), 
                shape=(num_points_u_fitted * num_points_v_fitted, num_control_points_u_fitted * num_control_points_v_fitted),
            )

            a = np.matmul(basis0.toarray().T, basis0.toarray())
            if np.linalg.det(a) == 0:
                cps_fitted,_,_,_ = np.linalg.lstsq(a, np.matmul(basis0.toarray().T, pts), rcond=None)
            else: 
                cps_fitted = np.linalg.solve(a, np.matmul(basis0.toarray().T, pts))            
        
            bspline_entity_surface = BSplineSurface(
                name=bspline_entity.name,
                order_u=order_u_fitted,
                order_v=order_v_fitted,
                shape=np.array([num_control_points_u_fitted, num_control_points_v_fitted, 3]),
                control_points=np.array(cps_fitted).reshape((num_control_points_u_fitted*num_control_points_v_fitted,3)),
                knots_u=np.array(knot_vector_u),
                knots_v=np.array(knot_vector_v))
            #print(bspline_entity.name)           
            self.input_bspline_entity_dict[bspline_entity.name] = bspline_entity_surface
        elif len(component_shape[:-1]) == 3:  # is volume
            print('fitting BSplineVolume has not been implemented yet')
            pass
        # return self.fit_bspline_entities(pts)
        return

    @dataclass
    class PointSet:
        pointset_id: int
        shape : np.ndarray # shape=(n,3) or (nu, nv 3), or (nu, nv, nw, 3)
        parent_pointers : List  # list of IDs to parent pointsets or BSplineEntities
        relative_map : np.ndarray  
        absolute_map : np.ndarray
        offset : np.ndarray
        offset_absolute_map : np.ndarray
        physical_coordinates : np.ndarray
        
    @dataclass
    class ProjectedPointSet(PointSet):
        pass

    @dataclass
    class DerivedPointSet(PointSet):
        pass

    @dataclass
    class EvaluatedPointSets(PointSet):     
        pass

    def read_openvsp_stp(self, file_name):
        read_openvsp_stp(self, file_name)

    def read_iges(self,file_name):
        read_iges(self, file_name)

    def write_step(self, file_name, plot=False):
        write_step(self, file_name, plot)

    def write_iges(self, file_name, plot = False):
        write_iges(self, file_name, plot)


if __name__ == "__main__": 
    geo = Geometry('CAD/eVTOL.stp')
