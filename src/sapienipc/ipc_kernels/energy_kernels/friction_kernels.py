import warp as wp

from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ...ipc_utils.global_defs import *
from ...ipc_utils.warp_types import *
from .collision_kernels import compute_db_dd, compute_e


@wp.func
def compute_f0(x: float, epsv_h: float):
    return wp.select(
        x < epsv_h,
        x,
        (x * x * x) / (-3.0 * epsv_h * epsv_h) + x * x / epsv_h + epsv_h / 3.0,
    )


@wp.func
def compute_f1(x: float, epsv_h: float):
    return wp.select(
        x < epsv_h, 1.0, -1.0 * (x * x) / (epsv_h * epsv_h) + 2.0 * x / epsv_h
    )


@wp.func
def compute_f1_over_x(x: float, epsv_h: float):
    return wp.select(x < epsv_h, 1.0 / x, -1.0 * x / (epsv_h * epsv_h) + 2.0 / epsv_h)


@wp.func
def compute_f2(x: float, epsv_h: float):
    return wp.select(x < epsv_h, 0.0, -2.0 * x / (epsv_h * epsv_h) + 2.0 / epsv_h)


@wp.func
def compute_f2_x_minus_f1_over_x_sq(x: float, epsv_h: float):
    return wp.select(x < epsv_h, -1.0 / (x * x), -1.0 / (epsv_h * epsv_h))


@wp.func
def cols_to_mat32(col1: wp.vec3, col2: wp.vec3):
    return mat32(col1[0], col2[0], col1[1], col2[1], col1[2], col2[2])


@wp.func
def pp_slide_basis(
    x0: wp.vec3, x1: wp.vec3, i0: int, i1: int, T_block: wp.array(dtype=mat32)
):
    x10_norm = wp.normalize(x0 - x1)

    basis1 = wp.cross(wp.vec3(1.0, 0.0, 0.0), x10_norm)
    basis2 = wp.cross(wp.vec3(0.0, 1.0, 0.0), x10_norm)

    norm1 = wp.length(basis1)
    norm2 = wp.length(basis2)

    basis1 = wp.select(norm1 > norm2, basis2 * (1.0 / norm2), basis1 * (1.0 / norm1))
    basis2 = wp.cross(x10_norm, basis1)

    T_block[i0] = cols_to_mat32(basis1, basis2)
    T_block[i1] = cols_to_mat32(-basis1, -basis2)


@wp.func
def pe_slide_basis(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    i0: int,
    i1: int,
    i2: int,
    T_block: wp.array(dtype=mat32),
):
    x10 = x0 - x1
    x12 = x2 - x1

    beta = wp.dot(x10, x12) / wp.dot(x12, x12)
    alpha = 1.0 - beta

    basis1 = wp.normalize(x12)
    basis2 = wp.normalize(wp.cross(x10, x12))

    T_block[i0] = cols_to_mat32(basis1, basis2)
    T_block[i1] = cols_to_mat32(basis1 * (-alpha), basis2 * (-alpha))
    T_block[i2] = cols_to_mat32(basis1 * (-beta), basis2 * (-beta))


@wp.func
def pt_slide_basis(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, T_block: wp.array(dtype=mat32)
):
    x10 = x0 - x1
    x12 = x2 - x1
    x13 = x3 - x1
    n = wp.normalize(wp.cross(x12, x13))
    x0_proj = x0 - wp.dot(n, x10) * n
    x10_proj = x0_proj - x1

    a00 = wp.dot(x12, x12)
    a01 = wp.dot(x12, x13)
    a11 = wp.dot(x13, x13)
    b0 = wp.dot(x10_proj, x12)
    b1 = wp.dot(x10_proj, x13)
    det_A = a00 * a11 - a01 * a01

    beta = (b0 * a11 - b1 * a01) / det_A
    gamma = (a00 * b1 - b0 * a01) / det_A
    alpha = 1.0 - beta - gamma

    basis1 = wp.normalize(wp.cross(n, x12))
    basis2 = wp.normalize(wp.cross(n, basis1))

    T_block[0] = cols_to_mat32(basis1, basis2)
    T_block[1] = cols_to_mat32(basis1 * (-alpha), basis2 * (-alpha))
    T_block[2] = cols_to_mat32(basis1 * (-beta), basis2 * (-beta))
    T_block[3] = cols_to_mat32(basis1 * (-gamma), basis2 * (-gamma))


@wp.func
def ee_slide_basis(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, T_block: wp.array(dtype=mat32)
):
    e0 = x0 - x1
    e1 = x2 - x3

    n = wp.normalize(wp.cross(e0, e1))
    basis1 = wp.normalize(wp.cross(n, e0))
    basis2 = wp.normalize(wp.cross(n, basis1))

    a00 = wp.dot(e0, e0)
    a01 = -wp.dot(e0, e1)
    a10 = wp.dot(e0, e1)
    a11 = -wp.dot(e1, e1)
    b0 = wp.dot(x3 - x1, e0)
    b1 = wp.dot(x3 - x1, e1)

    detA = a00 * a11 - a01 * a10

    mu0 = (b0 * a11 - b1 * a01) / detA
    mu1 = (a00 * b1 - a10 * b0) / detA

    T_block[0] = cols_to_mat32(basis1 * mu0, basis2 * mu0)
    T_block[1] = cols_to_mat32(basis1 * (1.0 - mu0), basis2 * (1.0 - mu0))
    T_block[2] = cols_to_mat32(basis1 * (-mu1), basis2 * (-mu1))
    T_block[3] = cols_to_mat32(basis1 * (mu1 - 1.0), basis2 * (mu1 - 1.0))


@wp.kernel
def friction_preprocess_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    kappa: float,
    d_hat: float,
    h: float,
    contact_d: wp.array(dtype=float),
    contact_c: wp.array(dtype=float),
    contact_eps_cross: wp.array(dtype=float),
    block_status: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.float32),
    contact_T: wp.array(dtype=mat32, ndim=2),
):
    """Can be called only AFTER launching the collision energy kernel on x!"""

    tid = wp.tid()
    if tid >= contact_counter[0]:
        return
    block_id = tid + n_static_blocks

    if block_status[block_id] == 0:
        return

    type1 = contact_type[block_id, 0]
    type2 = contact_type[block_id, 1]
    d = contact_d[block_id]
    contact_force[block_id] = -1.0 * kappa / (h * h) * compute_db_dd(d, d_hat)
    T_block = contact_T[block_id]  # 4 mat32

    T_block[0] = mat32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T_block[1] = mat32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T_block[2] = mat32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T_block[3] = mat32(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if type1 == PG_CONTACT_BLOCK:
        # TODO: fix this
        basis1 = wp.vec3(1.0, 0.0, 0.0)
        basis2 = wp.vec3(0.0, 1.0, 0.0)

        T_block[0] = cols_to_mat32(basis1, basis2)
        return

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    x0 = abd_apply(x_prev, x_rest, b_indices, abd_b_ids, p_affine, i0)
    x1 = abd_apply(x_prev, x_rest, b_indices, abd_b_ids, p_affine, i1)
    x2 = abd_apply(x_prev, x_rest, b_indices, abd_b_ids, p_affine, i2)
    x3 = abd_apply(x_prev, x_rest, b_indices, abd_b_ids, p_affine, i3)

    if type1 == PT_CONTACT_BLOCK:
        if type2 == PT_CONTACT_PT:
            pt_slide_basis(x0, x1, x2, x3, T_block)
        if type2 == PT_CONTACT_PP01:  # Equals x1
            pp_slide_basis(x0, x1, 0, 1, T_block)
        if type2 == PT_CONTACT_PP02:  # Equals x2
            pp_slide_basis(x0, x2, 0, 2, T_block)
        if type2 == PT_CONTACT_PE012:  # On edge x1x2
            pe_slide_basis(x0, x1, x2, 0, 1, 2, T_block)
        if type2 == PT_CONTACT_PP03:  # Equals x3
            pp_slide_basis(x0, x3, 0, 3, T_block)
        if type2 == PT_CONTACT_PE031:  # On edge x3x1
            pe_slide_basis(x0, x3, x1, 0, 3, 1, T_block)
        if type2 == PT_CONTACT_PE023:  # On edge x2x3
            pe_slide_basis(x0, x2, x3, 0, 2, 3, T_block)

    if type1 == EE_CONTACT_BLOCK:
        c = contact_c[block_id]
        eps = contact_eps_cross[block_id]
        e = compute_e(c, eps)
        contact_force[block_id] = contact_force[block_id] * e  # TODO: is this correct?

        if type2 == EE_CONTACT_PP02:  # x0x2
            pp_slide_basis(x0, x2, 0, 2, T_block)
        if type2 == EE_CONTACT_PP03:  # x0x3
            pp_slide_basis(x0, x3, 0, 3, T_block)
        if type2 == EE_CONTACT_PP12:  # x1x2
            pp_slide_basis(x1, x2, 1, 2, T_block)
        if type2 == EE_CONTACT_PP13:  # x1x3
            pp_slide_basis(x1, x3, 1, 3, T_block)
        if type2 == EE_CONTACT_PE023:  # x0e1
            pe_slide_basis(x0, x2, x3, 0, 2, 3, T_block)
        if type2 == EE_CONTACT_PE123:  # x1e1
            pe_slide_basis(x1, x2, x3, 1, 2, 3, T_block)
        if type2 == EE_CONTACT_PE201:  # x2e0
            pe_slide_basis(x2, x0, x1, 2, 0, 1, T_block)
        if type2 == EE_CONTACT_PE301:  # x3e0
            pe_slide_basis(x3, x0, x1, 3, 0, 1, T_block)
        if type2 == EE_CONTACT_EE:  # e0e1
            ee_slide_basis(x0, x1, x2, x3, T_block)


@wp.kernel
def friction_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    x_prev: wp.array(dtype=wp.vec3),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    eps_v: float,
    h: float,
    block_status: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.float32),
    contact_T: wp.array(dtype=mat32, ndim=2),
    contact_friction: wp.array(dtype=float),
    energy: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= contact_counter[0]:
        return
    block_id = tid + n_static_blocks

    if block_status[block_id] == 0:
        return

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    scene_id = particle_scene[i0]

    force = contact_force[block_id]
    T_block = contact_T[block_id]  # 4 mat32
    mu = contact_friction[block_id]

    x0 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i0) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i0
    )
    x1 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i1) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i1
    )
    x2 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i2) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i2
    )
    x3 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i3) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i3
    )

    u = (
        wp.transpose(T_block[0]) * x0
        + wp.transpose(T_block[1]) * x1
        + wp.transpose(T_block[2]) * x2
        + wp.transpose(T_block[3]) * x3
    )

    u_length = wp.length(u)

    friction_energy = compute_f0(u_length, eps_v * h) * force * mu * h * h

    wp.atomic_add(energy, scene_id, friction_energy)


@wp.kernel
def friction_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    eps_v: float,
    h: float,
    block_status: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.float32),
    contact_T: wp.array(dtype=mat32, ndim=2),
    contact_friction: wp.array(dtype=float),
    coeff: float,
    grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    particle_friction_force: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= contact_counter[0]:
        return
    block_id = tid + n_static_blocks

    if block_status[block_id] == 0:
        return

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    force = contact_force[block_id]
    T_block = contact_T[block_id]  # 4 mat32
    mu = contact_friction[block_id]

    x0 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i0) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i0
    )
    x1 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i1) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i1
    )
    x2 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i2) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i2
    )
    x3 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i3) - abd_apply(
        x_prev, x_rest, b_indices, abd_b_ids, p_affine, i3
    )

    u = (
        wp.transpose(T_block[0]) * x0
        + wp.transpose(T_block[1]) * x1
        + wp.transpose(T_block[2]) * x2
        + wp.transpose(T_block[3]) * x3
    )

    u_length = wp.length(u)
    eps_v_h = eps_v * h

    h_coeff = force * mu * h * h
    g_coeff = h_coeff * compute_f1_over_x(u_length, eps_v_h) * coeff

    u_new = u * g_coeff

    abd_accum(
        grad, T_block[0] * u_new * mask[i0], x_rest, b_indices, abd_b_ids, p_affine, i0
    )
    abd_accum(
        grad, T_block[1] * u_new * mask[i1], x_rest, b_indices, abd_b_ids, p_affine, i1
    )
    abd_accum(
        grad, T_block[2] * u_new * mask[i2], x_rest, b_indices, abd_b_ids, p_affine, i2
    )
    abd_accum(
        grad, T_block[3] * u_new * mask[i3], x_rest, b_indices, abd_b_ids, p_affine, i3
    )

    if i0 >= 0:
        wp.atomic_add(particle_friction_force, i0, T_block[0] * u_new / (h * h))
    if i1 >= 0:
        wp.atomic_add(particle_friction_force, i1, T_block[1] * u_new / (h * h))
    if i2 >= 0:
        wp.atomic_add(particle_friction_force, i2, T_block[2] * u_new / (h * h))
    if i3 >= 0:
        wp.atomic_add(particle_friction_force, i3, T_block[3] * u_new / (h * h))

    f1_over_x = compute_f1_over_x(u_length, eps_v_h)
    f2_x_minus_f1_over_x_sq = compute_f2_x_minus_f1_over_x_sq(u_length, eps_v_h)

    mid_mat = (
        wp.outer(u, wp.normalize(u)) * f2_x_minus_f1_over_x_sq
        + mat22(1.0, 0.0, 0.0, 1.0) * f1_over_x
    ) * h_coeff

    a = int(0)
    b = int(0)
    while a < 4:
        i_a = b_indices[block_id, a]
        T_a = T_block[a]

        while b < 4:
            i_b = b_indices[block_id, b]
            T_b = T_block[b]

            if i_a >= 0 and i_b >= 0:
                wp.atomic_add(blocks, block_id, a, b, T_a * mid_mat * wp.transpose(T_b))

            b += 1

        a += 1


# @wp.kernel
# def test_kernel(
#     a: wp.array(dtype=mat32),
#     b: wp.array(dtype=wp.mat22),
# ):
#     tid = wp.tid()
#     b[tid] = wp.transpose(a[tid]) * a[tid]


# def test():
#     wp.init()
#     device = wp.get_preferred_device()

#     import numpy as np

#     a = wp.array(
#         np.arange(12).reshape(2, 3, 2),
#         dtype=mat32,
#         device=device,
#     )
#     b = wp.zeros(2, dtype=wp.mat22, device=device)

#     wp.launch(test_kernel, 2, [a, b], device=device)

#     print(a)
#     print(b)
