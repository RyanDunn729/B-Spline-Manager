import csdl

class MeshEvaluationModel(csdl.model):
    '''
    Maps from the OML control points to the mesh coordinates.
    '''

    def initialize(self):
        geometry = self.parameters.declare('geometry')

        # Geometry assemble generates the absolute map that maps the geometry control points to the OML

    def define(self):

        geometry = self.parameters['geometry']

        geometry.assemble()
        self.map = geometry.eval_map
        self.offset = geometry.offset_eval_map

        control_points = self.declare_input('geometry_control_points')

        points = csdl.matvec(self.map, control_points)
        final_points = points + self.offset

        self.register_output('pointset_points', final_points)
