from sapienipc.ipc_kernels.energy_kernels.constraint_funcs import (
    angular_proportional_control_diff_func,
    angular_derivative_control_diff_func,
    area_constrant_diff_func,
)
import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_masked_func as abd_accum_m,
)
from ...ipc_utils.global_defs import *


@wp.kernel
def constraint_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    constraint_param: wp.array(dtype=wp.float32, ndim=2),
    constraint_lambda: wp.array(dtype=wp.float32),
    constraint_block_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    b_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mu: float,
    h: float,
    energy: wp.array(dtype=float),
):
    con_id = wp.tid()
    b_id = constraint_block_ids[con_id]
    if b_status[b_id] == 0:
        return

    con_type = b_type[b_id, 1]

    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]
    i3 = b_indices[b_id, 3]

    scene_id = particle_scene[i0]
    param = constraint_param[con_id]

    if con_type == CONSTRAINT_EQ2:
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)

        con_val = wp.length(x0 - x1)
        la = constraint_lambda[con_id]
        E = 0.5 * mu * con_val * con_val - la * con_val
        wp.atomic_add(energy, scene_id, E)

    elif con_type == CONSTRAINT_ANGULAR_PID_CONTROL:
        # x0: axis end 1
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        # x1: axis end 2
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        # x2: moving point on component 1, x02 \perp x01
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        # x3: moving point on component 2, x03 \perp x01
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        x0_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i0
        )
        x1_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i1
        )
        x2_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i2
        )
        x3_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i3
        )

        x01 = x1 - x0
        x02 = x2 - x0
        x03 = x3 - x0

        n2 = wp.normalize(wp.cross(x01, x02))
        n3 = wp.normalize(wp.cross(x03, x01))
        cos_theta = wp.dot(n3, n2)
        sin_theta = wp.dot(wp.cross(n3, n2), wp.normalize(x01))
        theta = wp.atan2(sin_theta, cos_theta)

        # wp.printf("theta: %f\n", theta)

        x01_prev_norm = wp.length(x1_prev - x0_prev)
        x02_prev_norm_sq = wp.length_sq(x2_prev - x0_prev)
        x03_prev_norm_sq = wp.length_sq(x3_prev - x0_prev)

        omega2 = wp.dot((x2 - x2_prev) - (x0 - x0_prev), wp.cross(x01, x02)) / (
            x01_prev_norm * x02_prev_norm_sq * h
        )
        omega3 = wp.dot((x3 - x3_prev) - (x0 - x0_prev), wp.cross(x01, x03)) / (
            x01_prev_norm * x03_prev_norm_sq * h
        )

        kp = param[0]  # proportional gain
        ki = param[1]  # integral gain
        kd = param[2]  # derivative gain
        theta_target = param[3]  # target angular position
        omega_target = param[4]  # target angular velocity

        delta_theta = theta - (wp.pi - theta_target)
        # project delta_theta to [-pi, pi]
        while delta_theta > wp.pi:
            delta_theta -= 2.0 * wp.pi
        while delta_theta < -wp.pi:
            delta_theta += 2.0 * wp.pi
        delta_omega = omega2 - omega3 - omega_target

        E = 0.5 * (kp * delta_theta * delta_theta + kd * delta_omega * delta_omega)
        wp.atomic_add(energy, scene_id, E)

    elif con_type == CONSTRAINT_AREA:
        # constraint: 2 area = ||cross(x1 - x0, x3 - x2)|| = 2.0 * area_target

        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        x01 = x1 - x0
        x23 = x3 - x2

        target_cross = constraint_param[con_id][0] * 2.0

        con_val = wp.length(wp.cross(x01, x23)) - target_cross

        la = constraint_lambda[con_id]
        E = 0.5 * mu * con_val * con_val - la * con_val
        wp.atomic_add(energy, scene_id, E)

    elif con_type == CONSTRAINT_DIST_SYM:
        # constraint: ||x1 - x0 + x3 - x2|| = 0
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        con_val = wp.length(x1 - x0 + x3 - x2)

        la = constraint_lambda[con_id]
        E = 0.5 * mu * con_val * con_val - la * con_val
        wp.atomic_add(energy, scene_id, E)

    elif con_type == CONSTRAINT_DIST_PID_CONTROL:
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)

        x0_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i0
        )
        x1_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i1
        )
        x2_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i2
        )

        # d = wp.length(x1 - x0)
        # v = wp.length((x1 - x1_prev) - (x0 - x0_prev)) / h

        dir = x2 - x0
        d = wp.dot(x1 - x0, dir)
        v = wp.dot((x1 - x1_prev) - (x0 - x0_prev), dir) / h

        kp = param[0]  # proportional gain
        ki = param[1]  # integral gain
        kd = param[2]  # derivative gain
        d_target = param[3]  # target angular position
        v_target = param[4]  # target angular velocity

        delta_d = d - d_target
        delta_v = v - v_target

        E = 0.5 * (kp * delta_d * delta_d + kd * delta_v * delta_v)
        wp.atomic_add(energy, scene_id, E)

        wp.printf("CONSTRAINT_DIST_PID_CONTROL E: %f, kp: %f, kd: %f, delta_d: %f, delta_v: %f\n", E, kp, kd, delta_d, delta_v)

    else:
        wp.printf("Unsupported constraint type %d\n", con_type)


@wp.kernel
def constraint_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=float),
    constraint_param: wp.array(dtype=wp.float32, ndim=2),
    constraint_lambda: wp.array(dtype=wp.float32),
    constraint_block_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    b_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mu: float,
    h: float,
    coeff: wp.float32,
    grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    con_id = wp.tid()
    b_id = constraint_block_ids[con_id]
    if b_status[b_id] == 0:
        return

    con_type = b_type[b_id, 1]

    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]
    i3 = b_indices[b_id, 3]

    scene_id = particle_scene[i0]

    param = constraint_param[con_id]
    block = blocks[constraint_block_ids[con_id]]

    if con_type == CONSTRAINT_EQ2:
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)

        con_val = wp.length(x0 - x1)
        la = constraint_lambda[con_id]
        # E = 0.5 * mu * con_val * con_val - la * con_val

        grad_i0 = (x0 - x1) * mu - wp.normalize(x0 - x1) * la
        grad_i1 = (x1 - x0) * mu - wp.normalize(x1 - x0) * la
        abd_accum_m(
            grad,
            grad_i0 * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
        )
        abd_accum_m(
            grad,
            grad_i1 * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i1,
        )

        I = wp.identity(n=3, dtype=wp.float32)
        hess_00 = I * (mu - la / con_val) + wp.outer(x0 - x1, x0 - x1) * (
            la / con_val**3.0
        )
        block[0, 0] = hess_00
        block[0, 1] = -hess_00
        block[1, 0] = -hess_00
        block[1, 1] = hess_00

    elif con_type == CONSTRAINT_ANGULAR_PID_CONTROL:
        kp = param[0]  # proportional gain
        ki = param[1]  # integral gain
        kd = param[2]  # derivative gain
        theta_target = param[3]  # target angular position
        omega_target = param[4]  # target angular velocity

        angular_proportional_control_diff_func(
            x,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
            i1,
            i2,
            i3,
            kp,
            theta_target,
            h,
            coeff,
            grad,
            block,
        )

        angular_derivative_control_diff_func(
            x,
            x_prev_step,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
            i1,
            i2,
            i3,
            kd,
            omega_target,
            h,
            coeff,
            grad,
            block,
        )

    elif con_type == CONSTRAINT_AREA:
        la = constraint_lambda[con_id]

        # wp.printf("la: %f \tmu = %f\n", la, mu)

        area_target = constraint_param[con_id, 0]

        area_constrant_diff_func(
            x,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
            i1,
            i2,
            i3,
            area_target,
            mu,
            la,
            coeff,
            grad,
            block,
        )

    elif con_type == CONSTRAINT_DIST_SYM:
        # constraint: ||x1 - x0 + x3 - x2|| = 0
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        x_diff = x1 - x0 + x3 - x2
        con_val = wp.length(x_diff)

        la = constraint_lambda[con_id]
        # E = 0.5 * mu * con_val * con_val - la * con_val

        dE_dx_diff = x_diff * mu - wp.normalize(x_diff) * la
        abd_accum_m(
            grad,
            dE_dx_diff * -coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
        )
        abd_accum_m(
            grad,
            dE_dx_diff * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i1,
        )
        abd_accum_m(
            grad,
            dE_dx_diff * -coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i2,
        )
        abd_accum_m(
            grad,
            dE_dx_diff * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i3,
        )

        I = wp.identity(n=3, dtype=wp.float32)
        hess_00 = I * (mu - la / con_val) + wp.outer(x_diff, x_diff) * (
            la / con_val**3.0
        )
        for i in range(4):
            for j in range(4):
                block[i, j] = wp.select(i % 2 == j % 2, -hess_00, hess_00)
    
    elif con_type == CONSTRAINT_DIST_PID_CONTROL:
        # Before: d = ||x1 - x0||
        # After: d = (x1 - x0).(x2 - x0)
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)

        x0_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i0
        )
        x1_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i1
        )
        x2_prev = abd_apply(
            x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i2
        )

        # d_vec = x1 - x0
        # d = wp.length(d_vec) + 1e-9
        # d_vec_norm = wp.normalize(d_vec)
        # v_vec = (x1 - x1_prev) - (x0 - x0_prev)
        # v = wp.length(v_vec) / h + 1e-9
        # v_vec_norm = wp.normalize(v_vec)

        d = wp.dot(x1 - x0, x2 - x0)
        v = wp.dot((x1 - x1_prev) - (x0 - x0_prev), x2 - x0) / h

        kp = param[0]  # proportional gain
        ki = param[1]  # integral gain
        kd = param[2]  # derivative gain
        d_target = param[3]  # target angular position
        v_target = param[4]  # target angular velocity

        delta_d = d - d_target
        delta_v = v - v_target
        
        # E = 0.5 * (kp * delta_d * delta_d + kd * delta_v * delta_v)

        dE_dd = kp * delta_d
        dE_dv = kd * delta_v
        dd_dx1 = x2 - x0
        dd_dx2 = x1 - x0
        dd_dx0 = -dd_dx1 - dd_dx2
        dv_dx1 = (x2 - x0) / h
        dv_dx2 = ((x1 - x1_prev) - (x0 - x0_prev)) / h
        dv_dx0 = -dv_dx1 - dv_dx2

        dE_dx0 = dE_dd * dd_dx0 + dE_dv * dv_dx0
        dE_dx1 = dE_dd * dd_dx1 + dE_dv * dv_dx1
        dE_dx2 = dE_dd * dd_dx2 + dE_dv * dv_dx2

        wp.printf("dE_dx0: %f %f %f\n", dE_dx0[0], dE_dx0[1], dE_dx0[2])
        wp.printf("dE_dx1: %f %f %f\n", dE_dx1[0], dE_dx1[1], dE_dx1[2])
        wp.printf("dE_dx2: %f %f %f\n", dE_dx2[0], dE_dx2[1], dE_dx2[2])

        abd_accum_m(
            grad,
            dE_dx0 * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i0,
        )
        abd_accum_m(
            grad,
            dE_dx1 * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i1,
        )
        abd_accum_m(
            grad,
            dE_dx2 * coeff,
            x_rest,
            b_indices,
            affine_b_ids,
            particle_affine,
            mask,
            i2,
        )

        d2E_dd2 = kp
        d2E_dv2 = kd
        
        I = wp.identity(n=3, dtype=wp.float32)
        # H11 = d2E_dd2 * wp.outer(dd_dx1, dd_dx1) + (dE_dd / d) * (I - wp.outer(d_vec_norm, d_vec_norm)) \
        #     + d2E_dv2 * wp.outer(dv_dx1, dv_dx1) + (dE_dv / (v * h * h)) * (I - wp.outer(v_vec_norm, v_vec_norm))
        H11 = d2E_dd2 * wp.outer(dd_dx1, dd_dx1) \
            + d2E_dv2 * wp.outer(dv_dx1, dv_dx1)
        H22 = d2E_dd2 * wp.outer(dd_dx2, dd_dx2) \
            + d2E_dv2 * wp.outer(dv_dx2, dv_dx2)
        H12 = d2E_dd2 * wp.outer(dd_dx1, dd_dx2) + dE_dd * I \
            + d2E_dv2 * wp.outer(dv_dx1, dv_dx2) + (dE_dv / h)* I
        H21 = wp.transpose(H12)
        H01 = -H11 - H21
        H10 = wp.transpose(H01)
        H02 = -H22 - H12
        H20 = wp.transpose(H02)
        H00 = -H10 - H20

        block[0, 0] = H00
        block[0, 1] = H01
        block[0, 2] = H02
        block[1, 0] = H10
        block[1, 1] = H11
        block[1, 2] = H12
        block[2, 0] = H20
        block[2, 1] = H21
        block[2, 2] = H22

        wp.printf("H00: %f %f %f\n     %f %f %f\n     %f %f %f\n", H00[0, 0], H00[0, 1], H00[0, 2], H00[1, 0], H00[1, 1], H00[1, 2], H00[2, 0], H00[2, 1], H00[2, 2])

    else:
        wp.printf("Unsupported constraint type %d\n", con_type)


@wp.kernel
def constraint_update_lambda_kernel(
    x: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=float),
    constraint_param: wp.array(dtype=wp.float32, ndim=2),
    constraint_lambda: wp.array(dtype=wp.float32),
    constraint_block_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    b_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mu: float,
):
    con_id = wp.tid()
    b_id = constraint_block_ids[con_id]
    if b_status[b_id] == 0:
        return

    con_type = b_type[b_id, 1]

    i0 = b_indices[b_id, 0]
    i1 = b_indices[b_id, 1]
    i2 = b_indices[b_id, 2]
    i3 = b_indices[b_id, 3]

    scene_id = particle_scene[i0]

    if con_type == CONSTRAINT_EQ2:
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)

        con_val = wp.length(x0 - x1)
        la = constraint_lambda[con_id]
        # E = 0.5 * mu * con_val * con_val - la * con_val

        constraint_lambda[con_id] = la - mu * con_val

    elif con_type == CONSTRAINT_ANGULAR_PID_CONTROL:
        pass

    elif con_type == CONSTRAINT_AREA:
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        x01 = x1 - x0
        x23 = x3 - x2

        target_cross = constraint_param[con_id][0] * 2.0

        con_val = wp.length(wp.cross(x01, x23)) - target_cross

        la = constraint_lambda[con_id]

        constraint_lambda[con_id] = la - mu * con_val

    elif con_type == CONSTRAINT_DIST_SYM:
        # constraint: ||x1 - x0 + x3 - x2|| = 0
        x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

        x_diff = x1 - x0 + x3 - x2
        con_val = wp.length(x_diff)

        la = constraint_lambda[con_id]

        constraint_lambda[con_id] = la - mu * con_val

    elif con_type == CONSTRAINT_DIST_PID_CONTROL:
        pass

    else:
        wp.printf("Unsupported constraint type %d\n", con_type)
