# F(u,v) = \sum_k 0.5 * [ \sum_i \sum_j B_i(u) B_j(v) C_ijk - P_k ] ** 2
# dF/du = [ \sum_i \sum_j B_i(u) B_j(v)  C_ijk - P_k ]
#         [ \sum_i \sum_j B_i'(u) B_j(v) C_ijk ]
# dF/dv = [ \sum_i \sum_j B_i(u) B_j(v)  C_ijk - P_k ]
#         [ \sum_i \sum_j B_i(u) B_j'(v) C_ijk ]

# F(u,v) = \sum_k 0.5 * [ P_k(u,v) - Q_k ] ** 2
# dF/du = [ P_k(u,v) - Q_k ] dP_k/du
# dF/dv = [ P_k(u,v) - Q_k ] dP_k/dv
# d2F/du2 = dP_k/du dP_k/du + [ P_k(u,v) - Q_k ] d2P_k/du2
# d2F/dudv = dP_k/du dP_k/dv + [ P_k(u,v) - Q_k ] d2P_k/dudv
# d2F/dv2 = dP_k/dv dP_k/dv + [ P_k(u,v) - Q_k ] d2P_k/dv2


cdef compute_surface_projection(
    int order_u, int num_control_points_u,
    int order_v, int num_control_points_v,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v,
    double* u_vec, double* v_vec,
    int guess_grid_n,
):
    cdef int i_pt, i_order_u, i_order_v, i_start_u, i_start_v
    cdef int i_iter, k, index
    cdef double u, v
    cdef double P00[2]
    cdef double P10[2], P01[2]
    cdef double P20[2], P02[2]
    cdef double P11[2],
    cdef double C[2], D[2], P[2]
    cdef double x[2], dx[2], G[2], H[4], N[4], det

    cdef double *basis_u0 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u1 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u2 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v0 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v1 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v2 = <double *> malloc(order_v * sizeof(double))

    # cdef double *knot_vector_u = <double *> malloc(
    #    (order_u + num_control_points_u) * sizeof(double))
    # cdef double *knot_vector_v = <double *> malloc(
    #    (order_v + num_control_points_v) * sizeof(double))

    #tom's initial guess vars
    cdef double temp_distance
    cdef double distance
    cdef double u_closest
    cdef double v_closest

    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)

    #tom's initial guess implementation
    if not (guess_grid_n == 0):
        for i in range(num_points):
            u_closest = .5
            v_closest = .5
            distance = 1000.

            for k in range(2):
                P[k] = pts[2 * i + k]

            for a in range(guess_grid_n) :
                for b in range(guess_grid_n) :
                    x[0] = a/guess_grid_n
                    x[1] = b/guess_grid_n

                    i_start_u = get_basis0(
                        order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
                    i_start_v = get_basis0(
                        order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)

                    for k in range(2): 
                        P00[k] = 0. 
                        for i_order_u in range(order_u):
                            for i_order_v in range(order_v):
                                index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                                    + 3 * (i_start_v + i_order_v) + k
                                C[k] = cps[index]
                                P00[k] = P00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]
                    for k in range(2):
                        D[k] = P[k]-P00[k]

                    temp_distance = norm(2,D)

                    if temp_distance < distance :
                        u_closest = x[0]
                        v_closest = x[1]
                        distance = temp_distance

            u_vec[i] = u_closest
            v_vec[i] = v_closest

    # print(u_vec)
    # print(v_vec)

    for i_pt in range(num_points):
        x[0] = u_vec[i_pt]
        x[1] = v_vec[i_pt]

        for k in range(2):
            P[k] = pts[2 * i_pt + k]

        for i_iter in range(max_iter):
            i_start_u = get_basis0(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
            i_start_u = get_basis1(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u1)
            i_start_u = get_basis2(
                order_u, num_control_points_u, x[0], knot_vector_u, basis_u2)
            
            i_start_v = get_basis0(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
            i_start_v = get_basis1(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v1)
            i_start_v = get_basis2(
                order_v, num_control_points_v, x[1], knot_vector_v, basis_v2)

            for k in range(2):
                P00[k] = 0.
                P10[k] = 0.
                P01[k] = 0.
                P20[k] = 0.
                P02[k] = 0.
                P11[k] = 0.

                for i_order_u in range(order_u):
                    for i_order_v in range(order_v):
                        index = 3 * num_control_points_v * (i_start_u + i_order_u) \
                            + 3  * (i_start_v + i_order_v) + k
                        C[k] = cps[index]

                        P00[k] = P00[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * C[k]
                        P10[k] = P10[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * C[k]
                        P01[k] = P01[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * C[k]
                        P20[k] = P20[k] + basis_u2[i_order_u] * basis_v0[i_order_v] * C[k]
                        P11[k] = P11[k] + basis_u1[i_order_u] * basis_v1[i_order_v] * C[k]
                        P02[k] = P02[k] + basis_u0[i_order_u] * basis_v2[i_order_v] * C[k]

                D[k] = P00[k] - P[k]

            # f(u,v,w) = ||P(u,v) - P0||_2^2
            # f(u,v) = || d ||_2^2
            # f(u,v) = dx^2 + dy^2 + dz^2
            # df/du = 2*D*dD/du
            # df/dv = 2*D*dD/dv
            # d2f/dudv = 2*D*d2D/dudv + 2*dD/du*dD/dv
            # d2f/du2 = 2*dD/du*dD/du + 2*D*d2D/du2

            G[0] = 2 * dot(2, D, P10) # dx
            G[1] = 2 * dot(2, D, P01) # dy
            H[0] = 2 * dot(2, P10, P10) + 2 * dot(2, D, P20) # dxx
            H[1] = 2 * dot(2, P10, P01) + 2 * dot(2, D, P11) # dxy
            H[2] = H[1] # dxy
            H[3] = 2 * dot(2, P01, P01) + 2 * dot(2, D, P02) # dyy

            for k in range(2):
                if (
                    (abs(x[k] - 0) < 1e-14) and (G[k] > 0) or
                    (abs(x[k] - 1) < 1e-14) and (G[k] < 0)
                ):
                    G[k] = 0.
                    H[1] = 0.
                    H[2] = 0.
                    H[k * 3] = 1.
            
            det = H[0]* H[3] - H[1]*H[2]

            N[0] = H[3]     / det
            N[1] = -H[1]    / det
            N[2] = -H[2]    / det
            N[3] = H[0]     / det

            matvec(2, 2, N, G, dx)
            for k in range(2):
                dx[k] = -dx[k]

            for k in range(2):
                if x[k] + dx[k] < 0:
                    dx[k] = -x[k]
                elif x[k] + dx[k] > 1:
                    dx[k] = 1 - x[k]

            norm_G = norm(2, G)
            norm_dx = norm(2, dx)
            # print(i_iter, norm_G)

            if norm_G < 1e-14 or norm_dx < 1e-14:
                # print("solution found")
                break

            for k in range(2):
                x[k] = x[k] + dx[k]

        u_vec[i_pt] = x[0]
        v_vec[i_pt] = x[1]

    free(basis_u0)
    free(basis_u1)
    free(basis_u2)
    free(basis_v0)
    free(basis_v1)
    free(basis_v2)

cdef double dot(int size, double* a, double* b):
    cdef int i
    cdef double result = 0

    for i in range(size):
        result = result + a[i] * b[i]

    return result


cdef double norm(int size, double* a):
    return dot(size, a, a) ** 0.5

cdef matvec(int size_i, int size_j, double* A, double* x, double *y):
    cdef int i, j

    for i in range(size_i):
        y[i] = 0.
        for j in range(size_j):
            y[i] = y[i] + A[i * size_j + j] * x[j]