import lsdo_geo.bsplines as bsp

import csdl


'''
wing_component.add_shape_parameter('twist', num=2, order=2)
'''

class SectionPropertiesModel(csdl.Model):

    def initialize(self):
        self.parameters.declare('ffd_blocks', types=list)

    def define(self):
        '''
        TOP LEVEL
        parameter_names = dict()
            - property_names are the keys of this top level dictionary
            - The property_names keys contain a dictionary that has parameter_name = string, and parameter_dict = dict()
            - parameter_dict has two keys order and num_cp
        
        '''

        ffd_blocks = self.parameters['ffd_blocks']

        # TODO: Should the keys in ffd_blocks contain an ffd object instead of a dictionary?
        for ffd_block_name, ffd_block_dict in ffd_blocks:
            num_sections = ffd_block_dict['num_sections']

            for property_name in property_names_list:

                property_var = self.create_input('{}'.format(property_name))

                for parameter_name, parameter_dict in parameter_names[property_name]:
                    
                    # Where do initial values of parameter var get set?
                    parameter_var = self.declare_variable(parameter_name)

                    # TODO: Implement way to create initial values of Control Points. Create default control points for shape 
                    if parameter_dict['num'] != None: 
                        control_points = np.zeros(parameter_dict['num'])                       
                    
                    bsp_curve = bsp.BSplineCurve(order_u=parameter_dict['order'], control_points=control_points)
                    sp_mtx = bsp_curve.evaluate_basis_matrix(np.linspace(0., 1., num_sections))

                    # This is the summation of all the orders of the properties
                    property_var = property_var + csdl.matvec(sp_mtx, parameter_var)


                ''' For each property and each section in the FFD we need to setup the matrix '''

                if property_name == 'twist':
                    rot_mat_x = np.zeros((num_sections, 3, 3))
                    for i in range(num_sections):
                        rot_mat_x[i,:,:] = [[1, 0, 0], [0, np.cos()]

                    pass

                elif property_name == 'rot_nyp':
                    pass
                
                elif property_name == 'rot_nzp':
                    pass
                    
                elif property_name == 'chord':
                    ''' 
                    Scales in the y-direction
                    '''

                    # TODO: Need to obtain initial chord somehow to divide scale_y by the initial chord

                    scale_y = self.create_output('scale_y', val=np.ones((nxp, nyp, nzp, 3)))

                    chord_scale = np.ones((nxp, nyp, nzp, 1))
                    chord_scale = chord_scale * property_var

                    scale_y[:, :, :, 1] = scale_y[:, :, :, 1] * chord_scale       
                    

                elif property_name == 'thickness':
                    
                    pass

                elif property_name == 'span':
                    trans_x = self.create_output('trans_x', val=np.zeros((nxp, nyp, nzp, 3)))

                    span = np.ones((nxp, nyp, nzp, 1))
                    span = span * property_var

                    trans_x[:, :, :, 0] = span       

                elif property_name == 'dihedral':
                    trans_z = self.create_output('trans_z', val=np.zeros((nxp, nyp, nzp, 3)))

                    dihedral = np.ones((nxp, nyp, nzp, 1))
                    dihedral = dihedral * property_var

                    trans_z[:, :, :, 2] = dihedral 

                elif property_name == 'sweep':
                    trans_y = self.create_output('trans_y', val=np.zeros((nxp, nyp, nzp, 3)))

                    sweep = np.ones((nxp, nyp, nzp, 1))
                    sweep = sweep * property_var

                    trans_y[:, :, :, 1] = sweep        

                elif property_name == 'x':
                    pass

                elif property_name == 'y':
                    pass

                elif property_name == 'z':
                    pass

                else:
                    Exception('Property Name does not exist!')






                self.register_output(
                    '{}_{}'.format(ffd_block_name, property_name), 
                    property_var,
                )


        

                
