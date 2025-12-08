import warp as wp

from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ...ipc_utils.global_defs import *
from ..distance_kernels.distance_kernels import (
    pt_pair_classify,
    pt_pair_distance,
    pt_pair_distance_grad,
    pt_pair_distance_hessian,
    ee_pair_classify,
    ee_pair_distance,
    ee_pair_distance_grad,
    ee_pair_distance_hessian,
)
from .ee_mollifier_funcs import (
    ee_pair_mollifier_grad,
    ee_pair_mollifier_hessian,
)


@wp.func
def compute_b(d: float, d_hat: float):
    return wp.select(d >= d_hat, -(d - d_hat) * (d - d_hat) * wp.log(d / d_hat), 0.0)


@wp.func
def compute_db_dd(d: float, d_hat: float):
    t = d - d_hat
    return wp.select(d >= d_hat, -2.0 * t * wp.log(d / d_hat) - t * t / d, 0.0)


@wp.func
def compute_d2b_dd2(d: float, d_hat: float):
    return wp.select(
        d >= d_hat,
        -2.0 * wp.log(d / d_hat)
        - 2.0 * (d - d_hat) / d
        - 1.0
        + (d_hat * d_hat) / (d * d),
        0.0,
    )


@wp.func
def compute_e(c: float, eps_cross: float):
    t = c / eps_cross
    return wp.select(c < eps_cross, 1.0, -1.0 * t * t + 2.0 * t)


@wp.func
def compute_de_dc(c: float, eps_cross: float):
    return wp.select(
        c < eps_cross, 0.0, -2.0 * c / (eps_cross * eps_cross) + 2.0 / eps_cross
    )


@wp.func
def compute_d2e_dc2(c: float, eps_cross: float):
    return wp.select(c < eps_cross, 0.0, -2.0 / (eps_cross * eps_cross))


@wp.kernel
def collision_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    kappa: float,
    d_hat: float,
    ee_classify_thres: float,
    ee_mollifier_thres: float,
    contact_d: wp.array(dtype=float),
    contact_c: wp.array(dtype=float),
    contact_eps_cross: wp.array(dtype=float),
    block_status: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= contact_counter[0]:
        return
    block_id = tid + n_static_blocks

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    scene_id = particle_scene[i0]

    x0 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i0)
    x1 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i1)
    x2 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i2)
    x3 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i3)

    d = 0.0
    energy_coeff = float(1.0)

    type1 = contact_type[block_id, 0]

    if type1 == PG_CONTACT_BLOCK:
        # d = wp.abs(x0[2] - ground_altitude)
        plane_i = -i1 - 1
        plane_normal = plane_normals[plane_i]
        plane_offset = plane_offsets[plane_i]
        d = wp.abs(wp.dot(plane_normal, x0) - plane_offset)

    elif type1 == PT_CONTACT_BLOCK:
        type2 = pt_pair_classify(x0, x1, x2, x3)
        contact_type[block_id, 1] = type2
        d = pt_pair_distance(x0, x1, x2, x3, type2)

    elif type1 == EE_CONTACT_BLOCK:
        type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
        contact_type[block_id, 1] = type2
        d = ee_pair_distance(x0, x1, x2, x3, type2)

        # Mollifier
        c = wp.length_sq(wp.cross(x1 - x0, x3 - x2))
        contact_c[block_id] = c
        eps_cross = (
            wp.length_sq(x_rest[i1] - x_rest[i0])
            * wp.length_sq(x_rest[i3] - x_rest[i2])
            * ee_mollifier_thres
        )
        contact_eps_cross[block_id] = eps_cross
        if c < eps_cross:
            energy_coeff *= compute_e(c, eps_cross)

    contact_d[block_id] = d

    if d < d_hat:
        block_status[block_id] = 1
        energy_upd = compute_b(d, d_hat)
        wp.atomic_add(energy, scene_id, energy_upd * kappa * energy_coeff)
    else:
        block_status[block_id] = 0


@wp.kernel
def collision_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    kappa: float,
    d_hat: float,
    h: float,
    contact_d: wp.array(dtype=float),
    contact_c: wp.array(dtype=float),
    contact_eps_cross: wp.array(dtype=float),
    coeff: float,
    contact_dd_dx: wp.array(dtype=wp.vec3, ndim=2),
    contact_dc_dx: wp.array(dtype=wp.vec3, ndim=2),
    grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    particle_collision_force: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid >= contact_counter[0]:
        return
    block_id = tid + n_static_blocks

    d = contact_d[block_id]
    if d >= d_hat:
        return

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    x0 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i0)
    x1 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i1)
    x2 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i2)
    x3 = abd_apply(x, x_rest, b_indices, abd_b_ids, p_affine, i3)

    block = blocks[block_id]
    dd_dx = contact_dd_dx[block_id]

    type1 = contact_type[block_id, 0]
    type2 = contact_type[block_id, 1]

    db_dd = compute_db_dd(d, d_hat)
    d2b_dd2 = compute_d2b_dd2(d, d_hat)
    dd_coeff = db_dd * kappa * coeff

    dd_dx[0] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[1] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[2] = wp.vec3(0.0, 0.0, 0.0)
    dd_dx[3] = wp.vec3(0.0, 0.0, 0.0)

    if type1 == PG_CONTACT_BLOCK:
        # Directly compute the gradient and hessian by d_signed
        # d_signed = x0[2] - ground_altitude
        plane_i = -i1 - 1
        plane_normal = plane_normals[plane_i]
        plane_offset = plane_offsets[plane_i]
        d_signed = wp.dot(plane_normal, x0) - plane_offset

        dd_dx[0] = wp.select(d_signed >= 0, -plane_normal, plane_normal)
        grad_upd = dd_dx[0] * dd_coeff * mask[i0]
        abd_accum(grad, grad_upd, x_rest, b_indices, abd_b_ids, p_affine, i0)
        wp.atomic_add(
            particle_collision_force, i0, dd_dx[0] * (-1.0 * db_dd * kappa / (h * h))
        )

        block[0, 0] = wp.outer(dd_dx[0], dd_dx[0]) * d2b_dd2 * kappa

    elif type1 == PT_CONTACT_BLOCK:
        pt_pair_distance_grad(x0, x1, x2, x3, dd_dx, type2)

        abd_accum(
            grad,
            dd_dx[0] * dd_coeff * mask[i0],
            x_rest,
            b_indices,
            abd_b_ids,
            p_affine,
            i0,
        )
        abd_accum(
            grad,
            dd_dx[1] * dd_coeff * mask[i1],
            x_rest,
            b_indices,
            abd_b_ids,
            p_affine,
            i1,
        )
        abd_accum(
            grad,
            dd_dx[2] * dd_coeff * mask[i2],
            x_rest,
            b_indices,
            abd_b_ids,
            p_affine,
            i2,
        )
        abd_accum(
            grad,
            dd_dx[3] * dd_coeff * mask[i3],
            x_rest,
            b_indices,
            abd_b_ids,
            p_affine,
            i3,
        )

        particle_force_coeff = -1.0 * db_dd * kappa / (h * h)
        wp.atomic_add(particle_collision_force, i0, dd_dx[0] * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i1, dd_dx[1] * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i2, dd_dx[2] * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i3, dd_dx[3] * particle_force_coeff)

        pt_pair_distance_hessian(
            x0, x1, x2, x3, kappa, db_dd, d2b_dd2, dd_dx, block, type2
        )

    elif type1 == EE_CONTACT_BLOCK:
        ee_pair_distance_grad(x0, x1, x2, x3, dd_dx, type2)

        c = contact_c[block_id]
        eps_cross = contact_eps_cross[block_id]

        if c >= eps_cross:
            energy_grad0 = dd_dx[0] * dd_coeff
            energy_grad1 = dd_dx[1] * dd_coeff
            energy_grad2 = dd_dx[2] * dd_coeff
            energy_grad3 = dd_dx[3] * dd_coeff
            d_grad_coeff = d2b_dd2
            d_hess_coeff = db_dd

        else:
            dc_dx = contact_dc_dx[block_id]

            b = compute_b(d, d_hat)
            e = compute_e(c, eps_cross)
            de_dc = compute_de_dc(c, eps_cross)
            d2e_dc2 = compute_d2e_dc2(c, eps_cross)

            dd_coeff = db_dd * e * kappa * coeff
            dc_coeff = de_dc * b * kappa * coeff

            d_grad_coeff = d2b_dd2 * e + db_dd * de_dc
            d_hess_coeff = db_dd * e

            c_grad_coeff = d2e_dc2 * b + de_dc * db_dd
            c_hess_coeff = de_dc * b

            ee_pair_mollifier_grad(x0, x1, x2, x3, dc_dx)

            energy_grad0 = dd_dx[0] * dd_coeff + dc_dx[0] * dc_coeff
            energy_grad1 = dd_dx[1] * dd_coeff + dc_dx[1] * dc_coeff
            energy_grad2 = dd_dx[2] * dd_coeff + dc_dx[2] * dc_coeff
            energy_grad3 = dd_dx[3] * dd_coeff + dc_dx[3] * dc_coeff

            ee_pair_mollifier_hessian(
                x0, x1, x2, x3, kappa, c_hess_coeff, c_grad_coeff, dc_dx, block
            )

        abd_accum(
            grad, energy_grad0 * mask[i0], x_rest, b_indices, abd_b_ids, p_affine, i0
        )
        abd_accum(
            grad, energy_grad1 * mask[i1], x_rest, b_indices, abd_b_ids, p_affine, i1
        )
        abd_accum(
            grad, energy_grad2 * mask[i2], x_rest, b_indices, abd_b_ids, p_affine, i2
        )
        abd_accum(
            grad, energy_grad3 * mask[i3], x_rest, b_indices, abd_b_ids, p_affine, i3
        )

        particle_force_coeff = -1.0 / (coeff * h * h)
        wp.atomic_add(particle_collision_force, i0, energy_grad0 * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i1, energy_grad1 * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i2, energy_grad2 * particle_force_coeff)
        wp.atomic_add(particle_collision_force, i3, energy_grad3 * particle_force_coeff)

        ee_pair_distance_hessian(
            x0, x1, x2, x3, kappa, d_hess_coeff, d_grad_coeff, dd_dx, block, type2
        )
