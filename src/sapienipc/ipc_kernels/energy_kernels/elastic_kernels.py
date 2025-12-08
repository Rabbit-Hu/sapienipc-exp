import warp as wp
from ...ipc_utils.warp_types import mat32


@wp.kernel
def elastic_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat33),
    rest_volumes: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    energy: wp.array(dtype=float),
):
    t_id = wp.tid()  # tet id

    b_id = tet_b_ids[t_id]  # block id
    
    if b_status[b_id] == 0:
        return

    i = b_indices[b_id, 0]  # vertex id
    j = b_indices[b_id, 1]
    k = b_indices[b_id, 2]
    l = b_indices[b_id, 3]

    scene_id = particle_scene[i]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    k_mu = materials[t_id, 0]
    k_la = materials[t_id, 1]

    v = rest_volumes[t_id]
    inv_Dm = inv_Dms[t_id]

    Ds = wp.mat33(x1 - x0, x2 - x0, x3 - x0)
    F = Ds * inv_Dm
    J = wp.determinant(F)
    
    if J < 0:
        wp.printf("Tet inversion! negative J: %f\n", J)
    
    log_J = wp.log(J)
    F_T = wp.transpose(F)
    F_T_F = F_T * F
    trace_F_T_F = wp.trace(F_T_F)

    energy_upd = (
        (trace_F_T_F - 3.0) * (k_mu / 2.0) - k_mu * log_J + k_la / 2.0 * log_J * log_J
    ) * (v * h * h)

    wp.atomic_add(energy, scene_id, energy_upd)


@wp.func
def _elas_hess_func(  # Elasetic Energy Hessian block
    i: int,
    j: int,
    k: int,
    l: int,
    k_mu: float,
    k_la: float,
    log_J: float,
    F_inv_T_inv_Dm_T: wp.mat33,
    inv_Dm_inv_Dm_T: wp.mat33,
):
    return (
        wp.select(i == k, 0.0, k_mu * inv_Dm_inv_Dm_T[l][j])
        + (k_mu - log_J) * F_inv_T_inv_Dm_T[k][j] * F_inv_T_inv_Dm_T[i][l]
        + k_la * F_inv_T_inv_Dm_T[k][l] * F_inv_T_inv_Dm_T[i][j]
    )


@wp.kernel
def elastic_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat33),
    rest_volumes: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    t_id = wp.tid()  # tet id

    b_id = tet_b_ids[t_id]  # block id
    
    if b_status[b_id] == 0:
        return

    block = blocks[b_id]  # there was a fatal bug...

    i = b_indices[b_id, 0]  # vertex id
    j = b_indices[b_id, 1]
    k = b_indices[b_id, 2]
    l = b_indices[b_id, 3]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    k_mu = materials[t_id, 0]
    k_la = materials[t_id, 1]

    v = rest_volumes[t_id]
    inv_Dm = inv_Dms[t_id]
    inv_Dm_T = wp.transpose(inv_Dm)

    Ds = wp.mat33(x1 - x0, x2 - x0, x3 - x0)
    F = Ds * inv_Dm
    J = wp.determinant(F)
    log_J = wp.log(J)
    F_T = wp.transpose(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)
    mat1 = F_inv_T * inv_Dm_T  # F_inv_T_inv_Dm_T
    mat2 = inv_Dm * inv_Dm_T  # inv_Dm_inv_Dm_T

    P = (F - F_inv_T) * k_mu + F_inv_T * (log_J * k_la)
    H = P * wp.transpose(inv_Dm) * v

    H = H * h * h * coeff

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])
    f0 = (f1 + f2 + f3) * -1.0

    wp.atomic_add(energy_grad, i, f0 * mask[i])
    wp.atomic_add(energy_grad, j, f1 * mask[j])
    wp.atomic_add(energy_grad, k, f2 * mask[k])
    wp.atomic_add(energy_grad, l, f3 * mask[l])

    for j in range(0, 3):
        for l in range(0, 3):
            mat_update = wp.mat33(
                _elas_hess_func(0, j, 0, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(0, j, 1, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(0, j, 2, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(1, j, 0, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(1, j, 1, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(1, j, 2, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(2, j, 0, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(2, j, 1, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_func(2, j, 2, l, k_mu, k_la, log_J, mat1, mat2),
            ) * (v * h * h)
            block[j + 1][l + 1] = block[j + 1][l + 1] + mat_update
            block[0][l + 1] = block[0][l + 1] - mat_update
            block[j + 1][0] = block[j + 1][0] - mat_update
            block[0][0] = block[0][0] + mat_update


@wp.kernel
def elastic_energy_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat22),
    rest_areas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    energy: wp.array(dtype=float),
):
    """temporary kernel for 2d elastic membrane
    Neo-Hookean elastic energy
    """

    t_id = wp.tid()

    b_id = tri_b_ids[t_id]
    if b_status[b_id] == 0:
        return
    
    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]

    scene_id = particle_scene[i0]

    k_mu = materials[t_id, 0]
    k_la = materials[t_id, 1]

    s = rest_areas[t_id]
    thickness = thicknesses[t_id]
    inv_Dm = inv_Dms[t_id]

    e1 = x1 - x0
    e2 = x2 - x0
    basis1 = wp.normalize(e1)
    basis2 = wp.normalize(e2 - wp.dot(e2, basis1) * basis1)

    basis_mat = mat32(basis1, basis2)
    Ds = wp.transpose(basis_mat) * mat32(e1, e2)

    F = Ds * inv_Dm
    F_T = wp.transpose(F)
    F_T_F = F_T * F
    trace_F_T_F = wp.trace(F_T_F)

    J = wp.determinant(F)
    log_J = wp.log(J)

    energy_upd = (
        (trace_F_T_F - 2.0) * (k_mu / 2.0) - k_mu * log_J + k_la / 2.0 * log_J * log_J
    ) * (s * thickness * h * h)

    wp.atomic_add(energy, scene_id, energy_upd)


@wp.func
def _elas_hess_2d_func(  # Elasetic Energy Hessian block
    i: int,
    j: int,
    k: int,
    l: int,
    k_mu: float,
    k_la: float,
    log_J: float,
    F_inv_T_inv_Dm_T: wp.mat22,
    inv_Dm_inv_Dm_T: wp.mat22,
):
    return (
        wp.select(i == k, 0.0, k_mu * inv_Dm_inv_Dm_T[l][j])
        + (k_mu - log_J) * F_inv_T_inv_Dm_T[k][j] * F_inv_T_inv_Dm_T[i][l]
        + k_la * F_inv_T_inv_Dm_T[k][l] * F_inv_T_inv_Dm_T[i][j]
    )


@wp.kernel
def elastic_diff_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat22),
    rest_areas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    t_id = wp.tid()

    b_id = tri_b_ids[t_id]
    if b_status[b_id] == 0:
        return
    
    block = blocks[b_id]
    
    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]

    k_mu = materials[t_id, 0]
    k_la = materials[t_id, 1]

    s = rest_areas[t_id]
    thickness = thicknesses[t_id]
    inv_Dm = inv_Dms[t_id]
    inv_Dm_T = wp.transpose(inv_Dm)

    e1 = x1 - x0
    e2 = x2 - x0
    basis1 = wp.normalize(e1)
    basis2 = wp.normalize(e2 - basis1 * wp.dot(e2, basis1))

    basis_mat = mat32(basis1, basis2)
    basis_mat_T = wp.transpose(basis_mat)
    Ds = basis_mat_T * mat32(e1, e2)

    F = Ds * inv_Dm
    J = wp.determinant(F)
    log_J = wp.log(J)
    F_T = wp.transpose(F)
    F_inv = wp.inverse(F)
    F_inv_T = wp.transpose(F_inv)
    mat1 = F_inv_T * inv_Dm_T
    mat2 = inv_Dm * inv_Dm_T

    P = (F - F_inv_T) * k_mu + F_inv_T * (log_J * k_la)
    H = (P * inv_Dm_T) * (s * thickness)

    H = H * (h * h * coeff)

    f1 = wp.vec2(H[0, 0], H[1, 0])
    f2 = wp.vec2(H[0, 1], H[1, 1])
    f0 = (f1 + f2) * -1.0

    wp.atomic_add(energy_grad, i0, basis_mat * f0 * mask[i0])
    wp.atomic_add(energy_grad, i1, basis_mat * f1 * mask[i1])
    wp.atomic_add(energy_grad, i2, basis_mat * f2 * mask[i2])

    for j in range(0, 2):
        for l in range(0, 2):
            mat_update = wp.mat22(
                _elas_hess_2d_func(0, j, 0, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_2d_func(0, j, 1, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_2d_func(1, j, 0, l, k_mu, k_la, log_J, mat1, mat2),
                _elas_hess_2d_func(1, j, 1, l, k_mu, k_la, log_J, mat1, mat2),
            ) * (s * thickness * h * h)
            mat_update_33 = basis_mat * mat_update * basis_mat_T
            block[j + 1][l + 1] = block[j + 1][l + 1] + mat_update_33
            block[0][l + 1] = block[0][l + 1] - mat_update_33
            block[j + 1][0] = block[j + 1][0] - mat_update_33
            block[0][0] = block[0][0] + mat_update_33


@wp.kernel
def bending_energy_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    hinge_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    rest_angles: wp.array(dtype=float),
    materials: wp.array(dtype=float),  # k_bend
    h: float,
    energy: wp.array(dtype=float),
):
    tid = wp.tid()

    b_id = hinge_b_ids[tid]
    if b_status[b_id] == 0:
        return
    
    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]
    i3 = b_indices[b_id, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    scene_id = particle_scene[i0]

    k_bend = materials[tid]
    rest_angle = rest_angles[tid]
    
    e0 = x2 - x1
    e0_len = wp.length(e0)
    e0_len_rest = wp.length(x_rest[i2] - x_rest[i1])
    n = wp.normalize(wp.cross(x1 - x0, x2 - x0))  # triangle 012
    n_ = wp.normalize(wp.cross(x2 - x3, x1 - x3))  # triangle 132
    cos_theta = wp.dot(n, n_)
    sin_theta = wp.dot(wp.cross(n, n_), e0) / e0_len
    theta = wp.atan2(sin_theta, cos_theta)  # range (-pi, pi)
    
    delta_theta = theta - rest_angle
    energy_upd = 0.5 * k_bend * delta_theta * delta_theta * h * h * e0_len_rest

    wp.atomic_add(energy, scene_id, energy_upd)


@wp.func
def _symmetric(A: wp.mat33):
    return A + wp.transpose(A)


@wp.kernel
def bending_diff_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    hinge_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    rest_angles: wp.array(dtype=float),
    materials: wp.array(dtype=float),  # k_bend
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    tid = wp.tid()

    b_id = hinge_b_ids[tid]
    if b_status[b_id] == 0:
        return
    
    block = blocks[b_id]
    
    k_bend = materials[tid]
    rest_angle = rest_angles[tid]
    
    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]
    i3 = b_indices[b_id, 3]

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    x3 = x[i3]

    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_ = x3 - x2
    e2_ = x3 - x1

    e0_lensq = wp.length_sq(e0)
    e1_lensq = wp.length_sq(e1)
    e2_lensq = wp.length_sq(e2)
    e1_lensq_ = wp.length_sq(e1_)
    e2_lensq_ = wp.length_sq(e2_)

    e0_len = wp.sqrt(e0_lensq)
    e1_len = wp.sqrt(e1_lensq)
    e2_len = wp.sqrt(e2_lensq)
    e1_len_ = wp.sqrt(e1_lensq_)
    e2_len_ = wp.sqrt(e2_lensq_)
    
    e0_unit = wp.normalize(e0)
    e1_unit = wp.normalize(e1)
    e2_unit = wp.normalize(e2)
    e1_unit_ = wp.normalize(e1_)
    e2_unit_ = wp.normalize(e2_)

    cos_alpha1 = wp.dot(e2_unit, e0_unit)
    cos_alpha2 = -wp.dot(e1_unit, e0_unit)
    cos_alpha1_ = wp.dot(e2_unit_, e0_unit)
    cos_alpha2_ = -wp.dot(e1_unit_, e0_unit)
    
    h0 = wp.length(wp.cross(e1, e2)) / e0_len
    h1 = wp.length(wp.cross(e0, e2)) / e1_len
    h2 = wp.length(wp.cross(e0, e1)) / e2_len
    h0_ = wp.length(wp.cross(e1_, e2_)) / e0_len
    h1_ = wp.length(wp.cross(e0, e2_)) / e1_len_
    h2_ = wp.length(wp.cross(e0, e1_)) / e2_len_

    n = wp.normalize(wp.cross(e0, e2))
    n_ = wp.normalize(wp.cross(e2_, e0))

    cos_theta = wp.dot(n, n_)
    sin_theta = wp.dot(wp.cross(n, n_), e0) / e0_len
    theta = wp.atan2(sin_theta, cos_theta)  # range (-pi, pi)
    
    delta_theta = theta - rest_angle
    e0_len_rest = wp.length(x_rest[i2] - x_rest[i1])
    dE_dtheta = k_bend * delta_theta * h * h * e0_len_rest
    d2E_dtheta2 = k_bend * h * h * e0_len_rest

    dtheta_dx1 = n * (cos_alpha2 / h1) + n_ * (cos_alpha2_ / h1_)
    dtheta_dx2 = n * (cos_alpha1 / h2) + n_ * (cos_alpha1_ / h2_)
    dtheta_dx0 = n * (-1.0 / h0)
    dtheta_dx3 = n_ * (-1.0 / h0_)

    # sum_grad = dtheta_dx0 + dtheta_dx1 + dtheta_dx2 + dtheta_dx3
    # wp.printf("sum_grad: %f %f %f\n", sum_grad[0], sum_grad[1], sum_grad[2])

    wp.atomic_add(energy_grad, i0, dE_dtheta * dtheta_dx0 * coeff * mask[i0])
    wp.atomic_add(energy_grad, i1, dE_dtheta * dtheta_dx1 * coeff * mask[i1])
    wp.atomic_add(energy_grad, i2, dE_dtheta * dtheta_dx2 * coeff * mask[i2])
    wp.atomic_add(energy_grad, i3, dE_dtheta * dtheta_dx3 * coeff * mask[i3])

    m0 = wp.normalize(wp.cross(e0, n))
    m1 = wp.normalize(wp.cross(e1, n))
    m2 = wp.normalize(wp.cross(n, e2))
    m0_ = wp.normalize(wp.cross(n_, e0))
    m1_ = wp.normalize(wp.cross(n_, e1_))
    m2_ = wp.normalize(wp.cross(e2_, n_))

    M0 = wp.outer(n, m0)
    M1 = wp.outer(n, m1)
    M2 = wp.outer(n, m2)
    M0_ = wp.outer(n_, m0_)
    M1_ = wp.outer(n_, m1_)
    M2_ = wp.outer(n_, m2_)

    N0 = M0 / e0_lensq
    N0_ = M0_ / e0_lensq

    P10 = cos_alpha1 * wp.transpose(M0) / (h1 * h0)
    P11 = cos_alpha1 * wp.transpose(M1) / (h1 * h1)
    P12 = cos_alpha1 * wp.transpose(M2) / (h1 * h2)
    P20 = cos_alpha2 * wp.transpose(M0) / (h2 * h0)
    P21 = cos_alpha2 * wp.transpose(M1) / (h2 * h1)
    P22 = cos_alpha2 * wp.transpose(M2) / (h2 * h2)
    P10_ = cos_alpha1_ * wp.transpose(M0_) / (h1_ * h0_)
    P11_ = cos_alpha1_ * wp.transpose(M1_) / (h1_ * h1_)
    P12_ = cos_alpha1_ * wp.transpose(M2_) / (h1_ * h2_)
    P20_ = cos_alpha2_ * wp.transpose(M0_) / (h2_ * h0_)
    P21_ = cos_alpha2_ * wp.transpose(M1_) / (h2_ * h1_)
    P22_ = cos_alpha2_ * wp.transpose(M2_) / (h2_ * h2_)

    Q0 = M0 / (h0 * h0)
    Q1 = M1 / (h0 * h1)
    Q2 = M2 / (h0 * h2)
    Q0_ = M0_ / (h0_ * h0_)
    Q1_ = M1_ / (h0_ * h1_)
    Q2_ = M2_ / (h0_ * h2_)

    block[0, 0] = -_symmetric(Q0) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx0, dtheta_dx0)
    block[3, 3] = -_symmetric(Q0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx3, dtheta_dx3)
    block[1, 1] = (_symmetric(P11) - N0 + _symmetric(P11_) - N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx1)
    block[2, 2] = (_symmetric(P22) - N0 + _symmetric(P22_) - N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx2)
    block[1, 0] = (P10 - Q1) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx0)
    block[2, 0] = (P20 - Q2) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx0)
    block[1, 3] = (P10_ - Q1_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx3)
    block[2, 3] = (P20_ - Q2_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx3)
    block[1, 2] = (P12 + wp.transpose(P21) + N0 + P12_ + wp.transpose(P21_) + N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx2)
    block[0, 3] = d2E_dtheta2 * wp.outer(dtheta_dx0, dtheta_dx3)

    block[0, 1] = wp.transpose(block[1, 0])
    block[0, 2] = wp.transpose(block[2, 0])
    block[3, 1] = wp.transpose(block[1, 3])
    block[3, 2] = wp.transpose(block[2, 3])
    block[2, 1] = wp.transpose(block[1, 2])
    block[3, 0] = wp.transpose(block[0, 3])

    # for i in range(0, 4):
    #     sum_row = wp.mat33(0.0)
    #     for j in range(0, 4):
    #         sum_row = sum_row + block[i, j]
    #     wp.printf("sum_row[%d]: ", i)
    #     for a in range(0, 3):
    #         for b in range(0, 3):
    #             wp.printf("%f ", sum_row[a, b])
    #     wp.printf("\n")
