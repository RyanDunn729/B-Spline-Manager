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


cdef compute_volume_projection(
    int order_u, int num_control_points_u,
    int order_v, int num_control_points_v,
    int order_w, int num_control_points_w,
    int num_points, int max_iter,
    double* pts, double* cps,
    double* knot_vector_u, double* knot_vector_v, double* knot_vector_w, 
    double* u_vec, double* v_vec, double* w_vec,
    int guess_grid_n,
):
    cdef int i_pt, i_order_u, i_order_v, i_order_w, i_start_u, i_start_v, i_start_w
    cdef int i_iter, k, index
    cdef double u, v, w
    cdef double P000[3]
    cdef double P100[3], P010[3], P001[3]
    cdef double P200[3], P020[3], P002[3]
    cdef double P110[3], P101[3], P011[3]
    cdef double C[3], D[3], P[3]
    cdef double x[3], dx[3], G[3], H[9], N[9], det

    cdef double *basis_u0 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u1 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_u2 = <double *> malloc(order_u * sizeof(double))
    cdef double *basis_v0 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v1 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_v2 = <double *> malloc(order_v * sizeof(double))
    cdef double *basis_w0 = <double *> malloc(order_w * sizeof(double))
    cdef double *basis_w1 = <double *> malloc(order_w * sizeof(double))
    cdef double *basis_w2 = <double *> malloc(order_w * sizeof(double))

    # cdef double *knot_vector_u = <double *> malloc(
    #    (order_u + num_control_points_u) * sizeof(double))
    # cdef double *knot_vector_v = <double *> malloc(
    #    (order_v + num_control_points_v) * sizeof(double))
    # cdef double *knot_vector_w = <double *> malloc(
    #    (order_w + num_control_points_w) * sizeof(double))

    #tom's initial guess vars
    cdef double temp_distance
    cdef double distance
    cdef double u_closest
    cdef double v_closest
    cdef double w_closest

    # get_standard_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_standard_uniform(order_v, num_control_points_v, knot_vector_v)
    # get_standard_uniform(order_w, num_control_points_w, knot_vector_w)

    # get_open_uniform(order_u, num_control_points_u, knot_vector_u)
    # get_open_uniform(order_v, num_control_points_v, knot_vector_v)
    # get_open_uniform(order_w, num_control_points_w, knot_vector_w)

    #tom's initial guess implementation
    if not (guess_grid_n == 0):
        for i in range(num_points):
            u_closest = .5
            v_closest = .5
            w_closest = .5
            distance = 1000.

            for k in range(3):
                P[k] = pts[3 * i + k]

            for a in range(guess_grid_n) :
                for b in range(guess_grid_n) :
                    for c in range(guess_grid_n) :
                        x[0] = a/guess_grid_n
                        x[1] = b/guess_grid_n
                        x[2] = c/guess_grid_n

                        i_start_u = get_basis0(
                            order_u, num_control_points_u, x[0], knot_vector_u, basis_u0)
                        i_start_v = get_basis0(
                            order_v, num_control_points_v, x[1], knot_vector_v, basis_v0)
                        i_start_w = get_basis0(
                            order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)

                        for k in range(3): 
                            P000[k] = 0. 
                            for i_order_u in range(order_u):
                                for i_order_v in range(order_v):
                                    for i_order_w in range(order_w):
                                        index = 4 * num_control_points_v * num_control_points_w * (i_start_u + i_order_u) \
                                            + 4 * num_control_points_w * (i_start_v + i_order_v) \
                                            + 4 * (i_start_w + i_order_w) + k
                                        C[k] = cps[index]
                                        P000[k] = P000[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                        for k in range(3) :
                            D[k] = P[k]-P000[k]

                        temp_distance = norm(3,D)

                        if temp_distance < distance :
                            u_closest = x[0]
                            v_closest = x[1]
                            w_closest = x[2]
                            distance = temp_distance

            u_vec[i] = u_closest
            v_vec[i] = v_closest
            w_vec[i] = w_closest

    # print(u_vec)
    # print(v_vec)
    # print(w_vec)

    for i_pt in range(num_points):
        x[0] = u_vec[i_pt]
        x[1] = v_vec[i_pt]
        x[2] = w_vec[i_pt]

        for k in range(3):
            P[k] = pts[3 * i_pt + k]

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

            i_start_w = get_basis0(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w0)
            i_start_w = get_basis1(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w1)
            i_start_w = get_basis2(
                order_w, num_control_points_w, x[2], knot_vector_w, basis_w2)

            for k in range(3):
                P000[k] = 0.
                P100[k] = 0.
                P010[k] = 0.
                P001[k] = 0.
                P200[k] = 0.
                P020[k] = 0.
                P002[k] = 0.
                P110[k] = 0.
                P101[k] = 0. 
                P011[k] = 0.

                for i_order_u in range(order_u):
                    for i_order_v in range(order_v):
                        for i_order_w in range(order_w):
                            index = 4 * num_control_points_w * num_control_points_v * (i_start_u + i_order_u) \
                                + 4 * num_control_points_w * (i_start_v + i_order_v) \
                                + 4 * (i_start_w + i_order_w) + k
                            C[k] = cps[index]

                            P000[k] = P000[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                            P100[k] = P100[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                            P010[k] = P010[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * basis_w0[i_order_w] * C[k]
                            P001[k] = P001[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w1[i_order_w] * C[k]
                            P200[k] = P200[k] + basis_u2[i_order_u] * basis_v0[i_order_v] * basis_w0[i_order_w] * C[k]
                            P020[k] = P020[k] + basis_u0[i_order_u] * basis_v2[i_order_v] * basis_w0[i_order_w] * C[k]
                            P002[k] = P002[k] + basis_u0[i_order_u] * basis_v0[i_order_v] * basis_w2[i_order_w] * C[k]
                            P110[k] = P110[k] + basis_u1[i_order_u] * basis_v1[i_order_v] * basis_w0[i_order_w] * C[k]
                            P101[k] = P101[k] + basis_u1[i_order_u] * basis_v0[i_order_v] * basis_w1[i_order_w] * C[k]
                            P011[k] = P011[k] + basis_u0[i_order_u] * basis_v1[i_order_v] * basis_w1[i_order_w] * C[k]

                D[k] = P000[k] - P[k]

            # f(u,v,w) = ||P(u,v) - P0||_2^2
            # f(u,v) = || d ||_2^2
            # f(u,v) = dx^2 + dy^2 + dz^2
            # df/du = 2*D*dD/du
            # df/dv = 2*D*dD/dv
            # d2f/dudv = 2*D*d2D/dudv + 2*dD/du*dD/dv
            # d2f/du2 = 2*dD/du*dD/du + 2*D*d2D/du2

            G[0] = 2 * dot(3, D, P100) # dx
            G[1] = 2 * dot(3, D, P010) # dy
            G[2] = 2 * dot(3, D, P001) # dz
            H[0] = 2 * dot(3, P100, P100) + 2 * dot(3, D, P200) # dxx
            H[1] = 2 * dot(3, P100, P010) + 2 * dot(3, D, P110) # dxy
            H[2] = 2 * dot(3, P100, P001) + 2 * dot(3, D, P101) # dxz
            H[3] = H[1]
            H[4] = 2 * dot(3, P010, P010) + 2 * dot(3, D, P020) # dyy
            H[5] = 2 * dot(3, P010, P001) + 2 * dot(3, D, P011) # dyz
            H[6] = H[2]
            H[7] = H[5]
            H[8] = 2 * dot(3, P001, P001) + 2 * dot(3, D, P002) # dzz

            for k in range(3):
                if (
                    (abs(x[k] - 0) < 1e-14) and (G[k] > 0) or
                    (abs(x[k] - 1) < 1e-14) and (G[k] < 0)
                ):
                    G[k] = 0.
                    H[1] = 0.
                    H[2] = 0.
                    H[3] = 0.
                    H[5] = 0.
                    H[6] = 0.
                    H[7] = 0.
                    H[k * 4] = 1.
            
            det = H[0] * (H[4]*H[8] - H[5]*H[7]) - H[3] * (H[1]*H[8] - H[2]*H[7]) + H[6] * (H[1]*H[5] - H[2]*H[4])

            N[0] = (H[4]*H[8] - H[5]*H[7])      / det
            N[1] = - (H[1]*H[8] - H[2]*H[7])    / det
            N[2] = (H[1]*H[5] - H[2]*H[4])      / det
            N[3] = - (H[3]*H[8] - H[5]*H[6])    / det
            N[4] = (H[0]*H[8] - H[2]*H[6])      / det
            N[5] = - (H[0]*H[5] - H[2]*H[3])    / det
            N[6] = (H[3]*H[7] - H[4]*H[6])      / det
            N[7] = - (H[0]*H[7] - H[1]*H[6])    / det
            N[8] = (H[0]*H[4] - H[1]*H[3])      / det


            matvec(3, 3, N, G, dx)
            for k in range(3):
                dx[k] = -dx[k]

            for k in range(3):
                if x[k] + dx[k] < 0:
                    dx[k] = -x[k]
                elif x[k] + dx[k] > 1:
                    dx[k] = 1 - x[k]

            norm_G = norm(3, G)
            norm_dx = norm(3, dx)
            # print(i_iter, norm_G)

            if norm_G < 1e-14 or norm_dx < 1e-14:
                # print("solution found")
                break

            for k in range(3):
                x[k] = x[k] + dx[k]

        u_vec[i_pt] = x[0]
        v_vec[i_pt] = x[1]
        w_vec[i_pt] = x[2]

    free(basis_u0)
    free(basis_u1)
    free(basis_u2)
    free(basis_v0)
    free(basis_v1)
    free(basis_v2)
    free(basis_w0)
    free(basis_w1)
    free(basis_w2)

    # free(knot_vector_u)
    # free(knot_vector_v)
    # free(knot_vector_w)

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