import numpy as np
import scipy.sparse as sps
from geometry import Geometry



# def extract_pointset_from_bspline(parametric_coordinates):



def define_linear_combination(geo, parent_pointset_list, coefficients_matrix, shape, offset=np.array([0., 0., 0.])):
    """ 
    Perform any arbitrary linear combination between PointSets and creates a new PointSet

    Parameters
    ----------
    parent_pointset_list : list
        A list of the parent poinsets in the order they are used in the relative map
    coefficients_matrix : csc_matrix     
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
    _define_linear_combination(geo, parent_pointset_list, coefficients_matrix, shape, offset=np.array([0., 0., 0.]))

def _define_linear_combination(geo, parent_pointset_list, relative_map, shape, offset=np.array([0., 0., 0.])):
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
        pointset_id = geo.current_id,
        shape = shape,
        parent_pointers = parent_pointers,
        relative_map = relative_map,
        absolute_map = None,
        offset = offset,            
        offset_absolute_map = None,
        physical_coordinates = None,
        )
    geo.pointsets_dict[geo.current_id] = pointset
    geo.current_id += 1
    return pointset


def define_linear_interpolation(geo, pointset0, pointset1, shape, output_parameters=np.array([0]), offset=np.array([0., 0., 0.])):       
    dims_dont_match = np.array(np.shape(pointset0) != np.shape(pointset1), ndmin = 1)
    if any(dims_dont_match): #pset1 and pset2 must have same number of points
        print('The sets you are trying to interpolate do not have the same dimensions.\n')
        return
    if output_parameters.all() == 0:
        zeros_array = np.zeros(np.shape(pointset0)[:-1])
        ones_array = np.ones(np.shape(pointset0)[:-1])
        output_parameters = np.linspace(zeros_array,ones_array, shape[-1], axis = -1)
    num_parent_points = int(np.prod(np.shape(pointset0)[:-1]))#points in single parent set
    num_interpolated_points = np.prod(shape)
    output_parameters = np.reshape(output_parameters, [num_parent_points,shape[-1]])
    relative_map = np.zeros([num_interpolated_points,2*num_parent_points])# (number of interpolation points, number of points in pointsets)
    floor = np.zeros([1,num_interpolated_points])
    for i in range(0,num_interpolated_points):
        relative_map[i, (i//shape[-1])] = 1 - output_parameters[(i//shape[-1]),(i%shape[-1])]
        relative_map[i, (i//shape[-1])+num_parent_points] = output_parameters[(i//shape[-1]),(i%shape[-1])]
    relative_map = sps.csc_matrix(relative_map)
    
    shape = np.append(shape,3)

    parent_pointset_list = [pointset0, pointset1]
    
    pointset = geo._define_linear_combination(parent_pointset_list, relative_map, shape, offset)
    return pointset

def define_bilinear_interpolation(geo, point_00, point_10, point_01, point_11, shape, output_parameters=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.])):
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

    pointset = geo._define_linear_combination(parent_pointset_list, relative_map, shape, offset)
    return pointset

def define_trilinear_interpolation(geo, point_000, point_100, point_010, point_110, point_001, point_101, point_011, point_111, 
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

    pointset = geo._define_linear_combination(parent_pointset_list, relative_map, shape, offset)
    return pointset

def define_2d_transfinite_interpolation(geo, u_curve0, u_curve1, v_curve0, v_curve1, output_parameters=np.array([0., 0., 0.]), offset=np.array([0., 0., 0.])):
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

    pointset = geo._define_linear_combination(parent_pointset_list, relative_map, shape, offset)
    return pointset

def define_3d_transfinite_interpolation(geo, output_parameters=None, offset=np.array([0., 0., 0.])):
    '''
    NOT IMPLEMENTED YET
    Performs 3D transfinite interpolation to generate a volume from 6 bounding surfaces.
    '''
    print("WARNING: 3D TFI not implemented yet!")
    pass

def extract_pointset(geo, parent_pointset, point_indices, shape, offset=np.array([0., 0., 0.])):
    shape = np.append(shape, 3)
    relative_map = np.zeros((len(point_indices), parent_pointset.relative_map.shape[0]))
    for index in point_indices:
        relative_map[index,index] = 1.
    relative_map = sps.csc_matrix(relative_map)
    
    parent_pointset_list = [parent_pointset]
    
    pointset = geo._define_linear_combination(parent_pointset_list, relative_map, shape, offset)
    return pointset