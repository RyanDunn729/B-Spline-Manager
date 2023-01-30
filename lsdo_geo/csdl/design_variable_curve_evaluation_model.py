import csdl
 
class DesignVariableCurveEvaluationModel(csdl.model):
    '''
    Maps from the FFD control points to the OML geometry control points.
    '''
    
    def initialize(self):
        self.parameters.declare('bsplinecurve')

        # User needs to specify such that the nxp from corresponding FFD is the same here. 
        self.parameters.declare('u_vec')
        
    
    def define(self):    

        bsplinecurve = self.parameters['bsplinecurve']
        u_vec = self.parameters['u_vec']

        self.map = bsplinecurve.compute_eval_map_points(u_vec)
                
        # The curve control points refer to the design variable control points that have been created 
        # else where. 
        curve_control_points = self.declare_input('curve_control_points')

        curve_points = csdl.matvec(self.map, curve_control_points)
    
        self.register_output('curve_points', curve_points)