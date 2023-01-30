import csdl

class FFDapplicationModel(csdl.model):
    '''
    Maps from the FFD control points to the OML geometry control points.
    '''

    def initialize(self):
        self.parameters.declare('ffd_block')

    def define(self):

        ffd_block = self.parameters['ffd_block']

        self.map = ffd_block.project_points_FFD() 

        # The ffd_control_points are received from ffdblockupdatemodel
        ffd_control_points = self.declare_input('ffd_control_points')

        geometry_control_points = csdl.matvec(self.map, ffd_control_points)

        self.register_output('geometry_control_points', geometry_control_points)
