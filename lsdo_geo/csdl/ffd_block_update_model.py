import csdl

class FfdBlockUpdateModel(csdl.model):
    '''
    Maps from the OML control points to the mesh coordinates.
    '''

    def initialize(self):
        geometry = self.parameters.declare('geometry')
        ffd_object = self.parameters.declare('ffd_object')

        # geometry = self.parameters['geometry']
        geometry.assemble()
        self.map = geometry.eval_map
        # self.offset = geometry.offset_eval_map

        initial_ffd = ffd_object.control_points
        
        origin = ffd_object.origin

        xprime = ffd_object.xprime
        yprime = ffd_object.yprime 
        zprime = ffd_object.zprime

    def define(self):
        
        # geometry = self.parameters['geometry']

        # These rotations, translations, and scalings all come from the section properties model.
        rot_x = self.declare_input('rot_x')
        rot_y = self.declare_input('rot_y')
        rot_z = self.declare_input('rot_z')

        trans_x = self.declare_input('trans_x')
        trans_y = self.declare_input('trans_y')
        trans_z = self.declare_input('trans_z')

        scale_y = self.declare_input('scale_y')
        scale_z = self.declare_input('scale_z')
        
        shape = self.declare_input('shape')

        # for i in 




        # points = self.map.dot(control_points)
        # final_points = points + self.offset

        self.register_output('pointset_points', final_points)
