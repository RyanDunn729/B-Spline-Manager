cdef get_standard_uniform(int order, int num_control_points, double* knot_vector):
    cdef int i
    cdef double den = num_control_points + order - 1

    for i in range(num_control_points + order):
        knot_vector[i] = i / den