import numpy as np
import pyiges

from geometry import Geometry 
from bspline_entities import BSplineCurve, BSplineSurface, BSplineVolume

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
import scipy.sparse as sps

import matplotlib.pyplot as plt

from vedo import Points, Plotter, LegendBox

''' Rib creation script '''
path_name = 'CAD/'
file_name = 'eVTOL.stp'
geo = Geometry(path_name + file_name)

wing_surface_names = [
    'Surf_WFWKRQIMCA, Wing, 0, 12',
    'Surf_WFWKRQIMCA, Wing, 0, 13',
    ]
top_wing_surface_names = [
    'Surf_WFWKRQIMCA, Wing, 0, 13',
    ]
bot_wing_surface_names = [
    'Surf_WFWKRQIMCA, Wing, 0, 12',
    ]
down_direction = np.array([0., 0., -1.])
up_direction = np.array([0., 0., 1.])

# Project top and bottom curve onto the surface
num_pts1 = 5

rib_top_curve, test_top_curve = geo.project_curve(np.linspace(np.array([180.,63.,145.]),np.array([210.,63.,145.]), num_pts1),
    projection_targets_names=top_wing_surface_names, offset=np.array([0., 0., 0.5]), plot=True, projection_direction = down_direction)#top_wing_surface_names, 

rib_bot_curve, test_bot_curve = geo.project_curve(np.linspace(np.array([180.,63.,105.]),np.array([210.,63.,105.]), num_pts1),
    projection_targets_names = bot_wing_surface_names, offset=np.array([0., 0., -0.5]), plot=True, projection_direction = up_direction)#

top_projected_point1 = geo.extract_pointset(rib_top_curve, np.array([0]), np.array([1]))
top_projected_point2 = geo.extract_pointset(rib_top_curve, np.array([-1]), np.array([1]))
bot_projected_point1 = geo.extract_pointset(rib_bot_curve, np.array([0]), np.array([1]))
bot_projected_point2 = geo.extract_pointset(rib_bot_curve, np.array([-1]), np.array([1]))

# Create rib side curves
num_pts2 = [15]
rib_side_curve1 = geo.perform_linear_interpolation(top_projected_point1, bot_projected_point1, num_pts2)
rib_side_curve2 = geo.perform_linear_interpolation(top_projected_point2, bot_projected_point2, num_pts2)

# Define rib surface mesh
rib_surface_mesh = geo.perform_2d_transfinite_interpolation(rib_top_curve, rib_bot_curve,  rib_side_curve1, rib_side_curve2)

geo.register_output(rib_surface_mesh, name = 'Rib')
# Concatenates vertically all the linear matrices 
geo.assemble()
# Evaluate the physical coornadites of points to be fitted
points_to_be_fitted = geo.evaluate()
geo.fit_bspline_entities(points_to_be_fitted)

path_name = 'CAD/'
file_name = 'eVTOL_rib.igs'
geo.write_iges(path_name + file_name)

#---plot
corner_projecting = []
corner_projected = []
vp_test = Plotter(N=7, axes=1)
top_bot = []
side = []
cps = []
surface_points = []
corner_projecting.append(Points(np.vstack(([180.,60.,145.],[210.,60.,145.],[180.,60.,100.],[210.,60.,100.])), r=15, c='crimson').legend('Corner projecting'))
corner_projected.append(Points(np.vstack((test_top_curve[0,:],test_top_curve[-1,:],test_bot_curve[0,:],test_bot_curve[-1,:])), r=15, c='darkgoldenrod').legend('Corner projected'))
top_bot.append(Points(test_top_curve, r=15, c='brown').legend('Top curve'))
top_bot.append(Points(test_bot_curve, r=15, c='brown').legend('Bot curve'))
side.append(Points(np.linspace(test_top_curve[0,:],test_bot_curve[0,:],num_pts2[0]), r=15, c='chartreuse').legend('Side curve'))
side.append(Points(np.linspace(test_top_curve[-1,:],test_bot_curve[-1,:],num_pts2[0]), r=15, c='chartreuse').legend('Side curve'))
TFI = Points(points_to_be_fitted, r=10, c='slategray')
for target in wing_surface_names:
    cps.append(Points(geo.input_bspline_entity_dict[target].control_points, r=8, c='cyan',alpha=0.3).legend('Control points'))
    bspline_entity = geo.input_bspline_entity_dict[target]
    order_u = bspline_entity.u_order
    order_v = bspline_entity.v_order
    num_control_points_u = bspline_entity.shape[0]
    num_control_points_v = bspline_entity.shape[1]
    num_points_u = 50   # TODO might want to pass these in as input
    num_points_v = 50

    nnz = num_points_u * num_points_v * order_u * order_v
    data = np.zeros(nnz)
    row_indices = np.zeros(nnz, np.int32)
    col_indices = np.zeros(nnz, np.int32)

    knot_vector_u = bspline_entity.u_knots
    knot_vector_v = bspline_entity.v_knots
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
    surface_points.append(Points(pts, r=5, c='seagreen',alpha=0.3).legend('Surface points'))
    cps.append(Points(geo.input_bspline_entity_dict[target].control_points, r=8, c='cyan',alpha=0.3).legend('Control points'))        
for surf in geo.output_bspline_entity_dict.values():
    if surf.name == 'Rib':
        bspline_fitted_cps = Points(surf.control_points, r=9, c='plum').legend('Fitted bspline')
vp_test.show(cps, surface_points, 'Control points of surface to be projected', at=0, viewup="z")#, lb1
vp_test.show(cps, surface_points, corner_projecting, 'Surface + projecting points', at=1, viewup="z")#, lb2
vp_test.show(cps, surface_points, corner_projected, 'Surface + projected points', at=2, viewup="z")
vp_test.show(cps, surface_points, top_bot, 'Surface + projected curves', at=3, viewup="z")
vp_test.show(cps, surface_points,  top_bot, side, 'Surface + projected curves + interpolated curves', at=4, viewup="z")
vp_test.show(cps, surface_points, top_bot, side, TFI, 'Surface + transfinite interpolated points', at=5, viewup="z")
vp_test.show(cps, surface_points, bspline_fitted_cps, 'Surface + control points of fitted b-spline surface', at=6, viewup="z", interactive=True)
       


