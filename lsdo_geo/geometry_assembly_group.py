import numpy as np
import math
import copy

from lsdo_avd.analysis_group import AnalysisGroup

from lsdo_geo.mesh.projection_preprocess_group import ProjectionPreprocessGroup
from lsdo_geo.mesh.projection_preprocess_group import ProjectionVectors
# from lsdo_geo.drep.finalproject import GetEdgeV
# from lsdo_geo.thomastest import FinalProjTest
from lsdo_geo.functionals.mean_aerodynamic_comp import MeanAerodynamicComp
from lsdo_geo.functionals.projected_area_comp import ProjectedAreaComp

from lsdo_geo.functionals.projected_area_group import ProjectedAreaGroup

from lsdo_geo.functionals.wetted_area_comp import WettedAreaComp
from lsdo_geo.functionals.lifting_surface_functionals_comp import LiftingSurfaceFunctionalsComp
from lsdo_geo.functionals.body_functionals_comp import BodyFunctionalsComp

from lsdo_geo.splines.einsum_reorder_comp import EinsumReorderComp
from lsdo_geo.splines.reshape_comp import ReshapeComp

from lsdo_comps.arithmetic_comps.average_comp import AverageComp
from lsdo_comps.arithmetic_comps.linear_combination_comp import LinearCombinationComp
from lsdo_comps.arithmetic_comps.summation_comp import SummationComp
from lsdo_comps.arithmetic_comps.multiplication_comp import MultiplicationComp

from openmdao.api import Problem, Group, view_model, IndepVarComp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

class GeometryAssembly(AnalysisGroup):

    def initialize(self):
        super(GeometryAssembly, self).initialize()

        self.options.declare('n_cpu', types=int)
        self.options.declare('n_cpv', types=int)
        self.options.declare('n_ptu', types=int)
        self.options.declare('n_ptv', types=int)
        # self.options.declare('has_fuselage', default=True, types=bool)
        self.geo_list = []
        self.prop_blade_list = []
        self.sub_prob = Problem()
        self.n_times = 1

    def set_n_times(self, n_times):
        self.n_times = n_times

    def retrieve_n_cp(self):
        n_cpu = self.options['n_cpu']
        n_cpv = self.options['n_cpv']

        return [n_cpu, n_cpv]

    def setup(self):
        n_times = self.n_times
        n_cpu = self.options['n_cpu']
        n_cpv = self.options['n_cpv']
        n_ptu = self.options['n_ptu']
        n_ptv = self.options['n_ptv']
        try:
            lifting_surfaces = self.lifting_surfaces
            projections = True
        except:
            projections = False

        if 'body_group' in self.geo_list:
            group = Group()
            self.add_subsystem('body_functionals_group', group, promotes=['*'])
            for surface in ['upper', 'lower']:
                comp = EinsumReorderComp(subscript_input='ijk', subscript_output='jik', shape_input=[n_ptu*n_ptv, n_times, 3], shape_output=[n_times, n_ptu*n_ptv, 3])
                group.add_subsystem('body_aero_comp_pre1_{}'.format(surface), comp)
                self.connect('body_group.{}_reshape4.array_out'.format(surface), 'body_aero_comp_pre1_{}.array_in'.format(surface))

                comp = ProjectedAreaGroup(
                    n_times=n_times,
                    n_pts_u=n_ptu,
                    n_pts_v=n_ptv,
                    in_name='body_wetted_area_in_array_{}'.format(surface),
                    out_name='body_wetted_area_{}'.format(surface)
                )
                group.add_subsystem('body_wetted_area_{}_comp'.format(surface), comp, promotes=['*'])

                # comp = WettedAreaComp(
                #     n_times=n_times,
                #     n_cpu=n_cpu,
                #     n_cpv=n_cpv,
                #     in_name='body_wetted_area_in_array_{}'.format(surface),
                #     out_name='body_wetted_area_{}'.format(surface)
                # )
                # group.add_subsystem('body_wetted_area_{}_comp'.format(surface), comp, promotes=['*'])

                comp = AverageComp(
                    in_shape=(n_times, n_ptu*n_ptv, 3),
                    out_shape=(n_times, 3),
                    average_axis=1,
                    in_name='body_average_{}_in'.format(surface),
                    out_name='body_average_{}_out'.format(surface),
                )
                group.add_subsystem('body_average_{}_comp'.format(surface), comp, promotes=['*'])
                self.connect('body_aero_comp_pre1_{}.array_out'.format(surface), ['body_wetted_area_in_array_{}'.format(surface), 'body_average_{}_in'.format(surface)])

            comp = SummationComp(
                shape = (n_times,),
                in_names=['body_wetted_area_upper', 'body_wetted_area_lower'],
                out_name='body_wetted_area_m2'
            )
            group.add_subsystem('body_wetted_area_sum', comp, promotes=['*'])

            comp = LinearCombinationComp(
                shape = (n_times, 3),
                in1_name = 'body_average_upper_out',
                in2_name = 'body_average_lower_out',
                out_name = 'body_average',
                c1 = 0.5,
                c2 = 0.5,
            )
            group.add_subsystem('body_average_comp', comp, promotes=['*'])

            comp = BodyFunctionalsComp(n_cpv=n_cpv, body_name='body', n_times=n_times)
            group.add_subsystem('body_functionals_comp', comp, promotes_outputs=['*'])
            for var in ['width', 'x_origin']:
                self.connect('body_group.{}_einsum_reorder2.array_out'.format(var), 'body_functionals_comp.{}'.format(var))

        temp_ivc = IndepVarComp()
        temp_ivc.add_output('lifting_surface_average_y', val=np.einsum('i,j->ij', np.ones(n_times), [1,0,1]))
        temp_ivc.add_output('lifting_surface_average_z', val=np.einsum('ik,j->ikj', np.ones((n_times, n_ptu*n_ptv)), [1,1,0]))
        temp_ivc.add_output('zero_placeholders_x', 0., shape=(n_times))
        temp_ivc.add_output('zero_placeholders_y', 0., shape=(n_times))
        self.add_subsystem('lifting_surface_average_ivc', temp_ivc, promotes=['*'])

        if projections == True:
            for lifting_surface_name, _ in lifting_surfaces:
                group = Group()
                self.add_subsystem('{}_functionals_group'.format(lifting_surface_name), group, promotes=['*'])
                for surface in ['upper', 'lower']:
                    comp = EinsumReorderComp(subscript_input='ijk', subscript_output='jik', shape_input=[n_ptu*n_ptv, n_times, 3], shape_output=[n_times, n_ptu*n_ptv, 3])
                    group.add_subsystem('{}_aero_comp_pre1_{}'.format(lifting_surface_name, surface), comp)
                    self.connect('{}_group.{}_reshape4.array_out'.format(lifting_surface_name, surface), '{}_aero_comp_pre1_{}.array_in'.format(lifting_surface_name, surface))

                    comp = ProjectedAreaGroup(
                        n_times=n_times,
                        n_pts_u=n_ptu,
                        n_pts_v=n_ptv,
                        in_name='{}_wetted_area_in_array_{}'.format(lifting_surface_name, surface),
                        out_name='{}_wetted_area_half_{}'.format(lifting_surface_name, surface)
                    )
                    group.add_subsystem('{}_wetted_area_half_{}_comp'.format(lifting_surface_name, surface), comp, promotes=['*'])

                    # comp = WettedAreaComp(
                    #     n_times=n_times,
                    #     n_cpu=n_cpu,
                    #     n_cpv=n_cpv,
                    #     in_name='{}_wetted_area_in_array_{}'.format(lifting_surface_name, surface),
                    #     out_name='{}_wetted_area_half_{}'.format(lifting_surface_name,surface)
                    # )
                    # group.add_subsystem('{}_wetted_area_half_{}_comp'.format(lifting_surface_name, surface), comp, promotes=['*'])

                    comp = AverageComp(
                        in_shape=(n_times, n_ptu*n_ptv, 3),
                        out_shape=(n_times, 3),
                        average_axis=1,
                        in_name='{}_average_{}_in'.format(lifting_surface_name, surface),
                        out_name='{}_average_{}_out'.format(lifting_surface_name, surface),
                    )
                    group.add_subsystem('{}_average_{}_comp'.format(lifting_surface_name,surface), comp, promotes=['*'])
                    self.connect('{}_aero_comp_pre1_{}.array_out'.format(lifting_surface_name,surface), ['{}_wetted_area_in_array_{}'.format(lifting_surface_name, surface), '{}_average_{}_in'.format(lifting_surface_name,surface)])

                comp = SummationComp(
                    shape = (n_times,),
                    in_names=['{}_wetted_area_half_upper'.format(lifting_surface_name), '{}_wetted_area_half_lower'.format(lifting_surface_name)],
                    out_name='{}_wetted_area_half_m2'.format(lifting_surface_name)
                )
                group.add_subsystem('{}_wetted_area_sum'.format(lifting_surface_name), comp, promotes=['*'])

                comp = LinearCombinationComp(
                    shape = (n_times, 3),
                    in1_name = '{}_average_upper_out'.format(lifting_surface_name),
                    in2_name = '{}_average_lower_out'.format(lifting_surface_name),
                    out_name = '{}_average_pre'.format(lifting_surface_name),
                    c1 = 0.5,
                    c2 = 0.5,
                )
                group.add_subsystem('{}_average_pre_comp'.format(lifting_surface_name), comp, promotes=['*'])

                comp = MultiplicationComp(
                    shape = (n_times, 3),
                    in1_name = '{}_average_pre'.format(lifting_surface_name),
                    in2_name = 'lifting_surface_average_y',
                    out_name = '{}_average'.format(lifting_surface_name)
                )
                group.add_subsystem('{}_average_comp'.format(lifting_surface_name), comp, promotes=['*'])

                comp = MultiplicationComp(
                    shape = (n_times, n_ptu*n_ptv, 3),
                    in1_name = '{}_projected_area_in_array'.format(lifting_surface_name),
                    in2_name = 'lifting_surface_average_z',
                    out_name = '{}_projected_area_in_array_2'.format(lifting_surface_name),
                )
                group.add_subsystem('{}_projected_area_in_array_2_comp'.format(lifting_surface_name), comp, promotes=['*'])

                comp = ProjectedAreaGroup(
                    n_times=n_times,
                    n_pts_u=n_ptu,
                    n_pts_v=n_ptv,
                    in_name='{}_projected_area_in_array_2'.format(lifting_surface_name),
                    out_name='{}_projected_area_half'.format(lifting_surface_name)
                )
                group.add_subsystem('{}_projected_area_half_comp'.format(lifting_surface_name), comp, promotes=['*'])

                # comp = ProjectedAreaComp(
                #     projection_normal_axis='z',
                #     n_times=n_times, 
                #     n_cpu=n_cpu, 
                #     n_cpv=n_cpv,
                #     in_name='{}_projected_area_in_array'.format(lifting_surface_name),
                #     out_name='{}_projected_area_half'.format(lifting_surface_name))
                # group.add_subsystem('{}_projected_area_half_comp'.format(lifting_surface_name), comp, promotes=['*'])
                self.connect('{}_aero_comp_pre1_lower.array_out'.format(lifting_surface_name), '{}_projected_area_in_array'.format(lifting_surface_name))

                comp = MeanAerodynamicComp(
                    n_times=n_times, 
                    n_pts_u=n_cpu, 
                    n_pts_v=n_cpv,
                    mesh_in_name='{}_aero_comps_in_array'.format(lifting_surface_name),
                    projected_area_name='{}_projected_area_half'.format(lifting_surface_name),
                    aero_center_out_name='{}_aerodynamic_center_pre'.format(lifting_surface_name),
                    aero_chord_out_name='{}_aerodynamic_chord'.format(lifting_surface_name),
                )
                group.add_subsystem('{}_aero_comps'.format(lifting_surface_name), comp, promotes=['*'])
                self.connect('{}_group.upper_reorder2.array_out'.format(lifting_surface_name), '{}_aero_comps_in_array'.format(lifting_surface_name))

                comp = MultiplicationComp(
                    shape = (n_times, 3),
                    in1_name = '{}_aerodynamic_center_pre'.format(lifting_surface_name),
                    in2_name = 'lifting_surface_average_y',
                    out_name = '{}_aerodynamic_center'.format(lifting_surface_name)
                )
                group.add_subsystem('{}_aerodynamic_center_comp'.format(lifting_surface_name), comp, promotes=['*'])

                geo_prop = LiftingSurfaceFunctionalsComp(n_cpv = n_cpv, lifting_surface_name=lifting_surface_name, n_times=n_times)
                group.add_subsystem('{}_functionals_comp'.format(lifting_surface_name), geo_prop, promotes_outputs=['*'])
                for var in ['twist', 'chord', 'x_origin', 'y_origin', 'z_origin']:
                    self.connect('{}_group.{}_einsum_reorder2.array_out'.format(lifting_surface_name, var), '{}_functionals_comp.{}'.format(lifting_surface_name, var))
                self.connect('{}_projected_area_half'.format(lifting_surface_name), '{}_functionals_comp.projected_area'.format(lifting_surface_name))
    
    def add_component(self, name, component, propeller_blade_num = None, rotation_center = None, var_names=[], propeller_var_names=[], nacelle_var_names=[]):
        n_times = self.n_times
        n_cpu = self.options['n_cpu']
        n_cpv = self.options['n_cpv']
        n_ptu = self.options['n_ptu']
        n_ptv = self.options['n_ptv']

        component.set_n_times(n_times=n_times)
        component.set_num_ctrl_pts(n_cpu=n_cpu, n_cpv=n_cpv)
        try:
            component.set_num_visual_pts(n_pts_u=n_ptu, n_pts_v=n_ptv)
        except:
            pass

        component.set_operating_conditions(condition_parameters=self.condition_parameters)

        if propeller_blade_num == None:
            self.geo_list.append('{}_group'.format(name))
            if 'rotor' in name:
                component.set_var_names(propeller_var_names=propeller_var_names, nacelle_var_names=nacelle_var_names)
                # component.set_operating_conditions(condition_parameters=self.condition_parameters)
                self.prop_blade_list.append(component.return_propeller_num())
            else:
                self.sub_prob.model.add_subsystem('{}_proj_group'.format(name), copy.deepcopy(component), promotes_inputs=var_names)
            self.add_subsystem('{}_group'.format(name), component, promotes_inputs=var_names+propeller_var_names+nacelle_var_names)
            
        else:
            for i in np.arange(propeller_blade_num):
                temp_copy = copy.deepcopy(component)
                temp_copy.set_rot('x', angle=i * 2 * np.pi / propeller_blade_num, rotation_center=rotation_center)
                self.geo_list.append('{}_blade{}_group'.format(name, i+1))
                self.add_subsystem('{}_blade{}_group'.format(name, i+1), temp_copy, promotes_inputs=var_names)

    def return_vlm_mesh_vectors(self, lifting_surfaces, order=4):
        n_times = self.n_times
        n_cpu = self.options['n_cpu']
        n_cpv = self.options['n_cpv']

        self.lifting_surfaces = lifting_surfaces
        
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            vlm_mesh_u = lifting_surface_data['num_points_x']
            if lifting_surface_data['centered'] == False:
                # vpoints = FinalProjTest()
                vlm_mesh_v = lifting_surface_data['num_points_z_half'] - 1
            else:
                vlm_mesh_v = lifting_surface_data['num_points_z_half']
                vpoints={
                'leading': np.linspace(0., 1., vlm_mesh_v) ,
                'trailing': np.linspace(0., 1., vlm_mesh_v)
                }
                
            projection_preprocess = ProjectionPreprocessGroup(
                n_times=n_times, n_cpu=n_cpu, n_cpv=n_cpv, n_pts_u=vlm_mesh_u, n_pts_v=vlm_mesh_v, vpoints=vpoints
            )

            self.sub_prob.model.add_subsystem('{}_projection'.format(lifting_surface_name), projection_preprocess)
            self.sub_prob.model.connect('{}_proj_group.upper_ctrl_pts_surface'.format(lifting_surface_name), ['{}_projection.leading_edge.ctrl_pts'.format(lifting_surface_name), '{}_projection.upper_bspline.ctrl_pts'.format(lifting_surface_name)] )
            self.sub_prob.model.connect('{}_proj_group.lower_ctrl_pts_surface'.format(lifting_surface_name), ['{}_projection.trailing_edge.ctrl_pts'.format(lifting_surface_name), '{}_projection.lower_bspline.ctrl_pts'.format(lifting_surface_name)] )
            
        self.sub_prob.setup()
        self.sub_prob.run_model()
        
        # view_model(self.sub_prob)

        proj_vectors = {}
        for lifting_surface_name, lifting_surface_data in lifting_surfaces:
            if lifting_surface_data['centered'] == False:
                vlm_mesh_v = lifting_surface_data['num_points_z_half'] - 1
            else:
                vlm_mesh_v = lifting_surface_data['num_points_z_half']
            projection_dict = {
                'interp_pts': self.sub_prob['{}_projection.interp_reorder2.array_out'.format(lifting_surface_name)],
                'fine_cps': {
                    'upper': self.sub_prob['{}_projection.upper_bspline.pts'.format(lifting_surface_name)],
                    'lower': self.sub_prob['{}_projection.lower_bspline.pts'.format(lifting_surface_name)]
                    },
                'cps': {
                    'upper': self.sub_prob['{}_proj_group.upper_ctrl_pts_surface'.format(lifting_surface_name)].reshape(n_cpu*n_cpv*n_times*3),
                    'lower': self.sub_prob['{}_proj_group.lower_ctrl_pts_surface'.format(lifting_surface_name)].reshape(n_cpu*n_cpv*n_times*3)
                    },
                }
                
            proj_vectors[lifting_surface_name] = ProjectionVectors(
                n_times=n_times, n_cpu=n_cpu, n_cpv=n_cpv, mesh_u=vlm_mesh_u, mesh_v=vlm_mesh_v, projection_dict=projection_dict, max_iter=49, order=order)
        
        print('Preprocess projection group complete.')

        #----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- 
        # # Debugging

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # import matplotlib.tri as mtri

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        # for lifting_surface_name, _ in lifting_surfaces:
        #     interp = prob['camber_meshes.{}_camber_mesh_mirror.array_out'.format(lifting_surface_name)]
        #     ax.scatter(interp[..., 0], interp[..., 1], interp[..., 2], label = 'interp points')
            
        # # cp = prob['wing_group.upper_3D_CP.array_out']
        # # cp = prob['projection.leading_edge.ctrl_pts']
        # # leading_points = prob['projection.leading_edge.pts']
        # # ax.scatter(leading_points[:, 0],leading_points[:, 1],leading_points[:, 2])
        # # ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], color='orange', label = 'cp')
        # plt.show()
        #----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- 

        return proj_vectors

    def set_operating_conditions(self, model, condition_parameters):
        self.condition_parameters = condition_parameters

    def plot_mpl(self, prob):
        
        def set_axes_equal(ax):
            '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
            cubes as cubes, etc..  This is one possible solution to Matplotlib's
            ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

            Input
            ax: a matplotlib axis, e.g., as output from plt.gca().
            '''

            x_limits = ax.get_xlim3d()
            y_limits = ax.get_ylim3d()
            z_limits = ax.get_zlim3d()

            x_range = abs(x_limits[1] - x_limits[0])
            x_middle = np.mean(x_limits)
            y_range = abs(y_limits[1] - y_limits[0])
            y_middle = np.mean(y_limits)
            z_range = abs(z_limits[1] - z_limits[0])
            z_middle = np.mean(z_limits)

            # The plot bounding box is a sphere in the sense of the infinity
            # norm, hence I call half the max range the plot radius.
            plot_radius = 0.5*max([x_range, y_range, z_range])

            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.set_aspect('equal')

        prop_count=-1
        for group in self.geo_list:
            if 'rotor' in group:
                upper_nacelle=prob['{}.nacelle_group.upper_ctrl_pts_surface'.format(group)]
                lower_nacelle=prob['{}.nacelle_group.lower_ctrl_pts_surface'.format(group)]
                ax.scatter(upper_nacelle[:, 0], upper_nacelle[:, 1], upper_nacelle[:, 2], color='orange')
                ax.scatter(lower_nacelle[:, 0], lower_nacelle[:, 1], lower_nacelle[:, 2], color='orange')
                prop_count += 1

                for i in np.arange(self.prop_blade_list[prop_count]):
                    upper_prop=prob['{}.blade{}_group.upper_ctrl_pts_surface'.format(group, i+1)]
                    lower_prop=prob['{}.blade{}_group.lower_ctrl_pts_surface'.format(group, i+1)]
                    ax.scatter(upper_prop[:, 0], upper_prop[:, 1], upper_prop[:, 2], color='red')
                    ax.scatter(lower_prop[:, 0], lower_prop[:, 1], lower_prop[:, 2], color='red')
            else:
                upper = prob['{}.upper_ctrl_pts_surface'.format(group)]
                lower = prob['{}.lower_ctrl_pts_surface'.format(group)]

                if ('wing' in group) or ('tail' in group):
                    color = 'blue'
                elif 'blade' in group:
                    color = 'red'
                elif 'body' in group:
                    color = 'g'
                elif 'nacelle' in group:
                    color = 'orange'

                ax.scatter(upper[:, 0], upper[:, 1], upper[:, 2], color=color)
                ax.scatter(lower[:, 0], lower[:, 1], lower[:, 2], color=color)

        # test = prob['camber_meshes.wing_camber_mesh_mirror.array_out'][0]

        set_axes_equal(ax)
        plt.show()

    def get_ctrl_faces_geo(self, prob):
        n_times = self.n_times
        n_cpu = self.options['n_ptu']
        n_cpv = self.options['n_ptv']
        
        def get_triangle_faces(n_cpu, n_cpv):
            faces = np.arange((n_cpu-1)*(n_cpv-1)*2*3).reshape((n_cpu-1) * (n_cpv-1) * 2, 3)
            ind=0
            for i in np.arange(n_cpu-1):
                for j in np.arange(n_cpv-1):
                    faces[ind] = [i*n_cpv + j, i*n_cpv + j + 1, (i+1)*n_cpv + j]
                    ind += 1
                    faces[ind] = [(i+1)*n_cpv + j, (i+1)*n_cpv + j + 1, i*n_cpv + j + 1]
                    ind += 1

            return faces
        
        faces_orig = get_triangle_faces(n_cpu, n_cpv)

        faces = list(np.zeros((n_times,1,3)))
        ctrl_pts = list(np.zeros((n_times,1,3)))

        group_count = -1
        prop_count = -0.5
        
        for group in self.geo_list:
            for surface in ['upper', 'lower']:
                group_count += 1
                if 'rotor' in group:
                    prop_count += 0.5
                    for time_ind in np.arange(n_times):
                        # ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.nacelle_group.{}_ctrl_pts_surface'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                        ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.nacelle_group.{}_bspline_surface.pts'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                        faces[time_ind] = np.concatenate([faces[time_ind], faces_orig + group_count * (n_cpu*n_cpv)])
                    for i in np.arange(self.prop_blade_list[math.floor(prop_count)]):
                        group_count += 1
                        for time_ind in np.arange(n_times):
                            # ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.blade{}_group.{}_ctrl_pts_surface'.format(group, i+1, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                            ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.blade{}_group.{}_bspline_surface.pts'.format(group, i+1, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                            faces[time_ind] = np.concatenate([faces[time_ind], faces_orig + group_count * (n_cpu*n_cpv)])
                else:
                    for time_ind in np.arange(n_times):
                        # ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.{}_ctrl_pts_surface'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                        ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], prob['{}.{}_bspline_surface.pts'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :]])
                        faces[time_ind] = np.concatenate([faces[time_ind], faces_orig + group_count * (n_cpu*n_cpv)])
                    if ('wing' in group) or ('tail' in group):
                        group_count += 1
                        for time_ind in np.arange(n_times):
                            # mirror = prob['{}.{}_ctrl_pts_surface'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :].copy()
                            mirror = prob['{}.{}_bspline_surface.pts'.format(group, surface)].reshape(n_cpu*n_cpv, n_times, 3)[:, time_ind, :].copy()
                            mirror[:, 1] = -mirror[:, 1]
                            ctrl_pts[time_ind] = np.concatenate([ctrl_pts[time_ind], mirror])
                            faces[time_ind] = np.concatenate([faces[time_ind], faces_orig + group_count * (n_cpu*n_cpv)])

        for i in np.arange(n_times):
            ctrl_pts[i] = np.delete(ctrl_pts[i], 0, 0)
            faces[i] = np.delete(faces[i], 0, 0)

        return faces, ctrl_pts

    def export_stl(self, prob, name=None):
        from stl import mesh

        faces, ctrl_pts = self.get_ctrl_faces_geo(prob)

        for ind, faces_each in enumerate(faces):
            ctrl_pts_each = ctrl_pts[ind]

            total_mesh = mesh.Mesh(np.zeros(faces_each.shape[0], dtype=mesh.Mesh.dtype))

            for i, f in enumerate(faces_each.astype(int)):
                for j in range(3):
                    total_mesh.vectors[i][j] = ctrl_pts_each[f[j],:]
            
            total_mesh.save(name+'_mesh_{}.stl'.format(ind))

    def get_3D_viz(self, prob, plot_times = True, grid_on = False):
        import pyqtgraph.opengl as gl

        faces, ctrl_pts = self.get_ctrl_faces_geo(prob)

        if plot_times == True:
            num_plots = self.n_times
        else:
            num_plots = len(plot_times)

        w = gl.GLViewWidget()
        w.setWindowTitle('3D Visualization')
        w.setCameraPosition(distance=40)

        if grid_on == True:
            g = gl.GLGridItem()
            g.scale(1, 1, 1)
            w.addItem(g)
        
        count = 0

        for ind, faces_each in enumerate(faces):
            ctrl_pts_each = ctrl_pts[ind]
            if plot_times != True:
                if prob.model.missions.parameters[ind]['name'] not in plot_times:
                    continue
            m1 = gl.GLMeshItem(vertexes=ctrl_pts_each, faces=faces_each.astype(int), smooth=True, drawEdges=True)
            m1.translate(-3.7 + -7.4*((num_plots-1)//3)/2, -13*((num_plots-1)%3)/2, 0)
            m1.translate(7.4*(count//3), 13*(count%3), 0)
            w.addItem(m1)
            count += 1

        return w