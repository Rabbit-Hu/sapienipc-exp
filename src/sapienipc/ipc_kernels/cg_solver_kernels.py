import warp as wp
from .abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)


@wp.kernel
def update_p_r_z_compute_r_2_zr_kernel(
    v: wp.array(dtype=wp.vec3),
    A_v: wp.array(dtype=wp.vec3),
    v_A_v: wp.array(dtype=float),
    zr: wp.array(dtype=float),
    diag: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=int),
    p: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
    r_2: wp.array(dtype=float),
    zr_new: wp.array(dtype=float),
):
    tid = wp.tid()
    scene_id = particle_scene[tid]

    if v_A_v[scene_id] > 1e-16:
        alpha = zr[scene_id] / v_A_v[scene_id]
        p[tid] = p[tid] + alpha * v[tid]
        r[tid] = r[tid] - alpha * A_v[tid]
    z[tid] = wp.cw_mul(diag[tid], r[tid])
    wp.atomic_add(r_2, scene_id, wp.dot(r[tid], r[tid]))
    wp.atomic_add(zr_new, scene_id, wp.dot(z[tid], r[tid]))


@wp.kernel
def update_v_p_best_kernel(
    z: wp.array(dtype=wp.vec3),
    zr: wp.array(dtype=float),
    zr_new: wp.array(dtype=float),
    p: wp.array(dtype=wp.vec3),
    r_2: wp.array(dtype=float),
    particle_scene: wp.array(dtype=int),
    v: wp.array(dtype=wp.vec3),
    p_best: wp.array(dtype=wp.vec3),
    r_2_best: wp.array(dtype=float),
):
    tid = wp.tid()
    scene_id = particle_scene[tid]

    s = zr_new[scene_id] / zr[scene_id]
    v[tid] = z[tid] + s * v[tid]

    if r_2[scene_id] < r_2_best[scene_id]:
        p_best[tid] = p[tid]


@wp.kernel
def update_r_2_best_kernel(
    r_2: wp.array(dtype=float),
    r_2_best: wp.array(dtype=float),
):
    tid = wp.tid()
    if r_2[tid] < r_2_best[tid]:
        r_2_best[tid] = r_2[tid]


@wp.kernel
def compute_dot_kernel(
    x: wp.array(dtype=wp.vec3), 
    y: wp.array(dtype=wp.vec3), 
    particle_scene: wp.array(dtype=int),
    ret: wp.array(dtype=float)
):
    """dim: n_particles
    input: x, y
    output: ret += x * y"""
    tid = wp.tid()
    scene_id = particle_scene[tid]

    wp.atomic_add(ret, scene_id, wp.dot(x[tid], y[tid]))


@wp.kernel
def a_plus_k_b_kernel(
    a: wp.array(dtype=wp.vec3),
    k: float,
    b: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3),
):
    """dim: n_particles
    input: a, k, b
    output: ret = a + k * b"""
    tid = wp.tid()

    ret[tid] = a[tid] + k * b[tid]


@wp.kernel
def compute_velocity_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    x_next: wp.array(dtype=wp.vec3),
    dt: float,
    v_next: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v_next[tid] = (x_next[tid] - x_prev[tid]) / dt


@wp.kernel
def compute_block_diag_kernel(
    n_particles: int,
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    m: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    dbc_mask: wp.array(dtype=float),
    kappa_con: float,
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    block_status: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    diag: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if tid >= n_particles and tid >= n_static_blocks + contact_counter[0]:
        return

    if tid < n_particles:
        affine_id = p_affine[tid]
        m_i = wp.select(affine_id == -2, m[tid], 0.0)
        m_i = mask[tid] * m_i + kappa_con * dbc_mask[tid] * m[tid]
        if affine_id < 0:
            wp.atomic_add(diag, tid, wp.vec3(m_i))
        if affine_id >= 0:
            x_r = x_rest[tid]
            x_r_2 = wp.cw_mul(x_r, x_r)
            block_id = abd_b_ids[affine_id]
            wp.atomic_add(diag, b_indices[block_id, 3], wp.vec3(m_i))
            wp.atomic_add(diag, b_indices[block_id, 0], x_r_2 * m_i)
            wp.atomic_add(diag, b_indices[block_id, 1], x_r_2 * m_i)
            wp.atomic_add(diag, b_indices[block_id, 2], x_r_2 * m_i)

    if tid < n_static_blocks + contact_counter[0]:
        if block_status[tid] == 1:
            # for i_local in range(4):
            #     i_global = b_indices[tid, i_local]
            #     if i_global >= 0:
            #         if mask[i_global] == 0.0:
            #             continue
            #         block = blocks[tid, i_local, i_local]
            #         affine_id = p_affine[i_global]
            #         if affine_id < 0:  # Not on an affine body, or is a affine proxy
            #             wp.atomic_add(
            #                 diag,
            #                 i_global,
            #                 wp.vec3(block[0, 0], block[1, 1], block[2, 2]),
            #             )
            #         else:
            #             x_r = x_rest[i_global]
            #             x_r_2 = wp.cw_mul(x_r, x_r)
            #             block_id = abd_b_ids[affine_id]
            #             wp.atomic_add(
            #                 diag,
            #                 b_indices[block_id, 3],
            #                 wp.vec3(block[0, 0], block[1, 1], block[2, 2]),
            #             )
            #             wp.atomic_add(diag, b_indices[block_id, 0], x_r_2 * block[0, 0])
            #             wp.atomic_add(diag, b_indices[block_id, 1], x_r_2 * block[1, 1])
            #             wp.atomic_add(diag, b_indices[block_id, 2], x_r_2 * block[2, 2])

            for i_local in range(4):
                i_global = b_indices[tid, i_local]
                if i_global < 0:
                    continue
                if mask[i_global] == 0.0:
                    continue

                affine_i = p_affine[i_global]

                if affine_i < 0:  # Not on an affine body, or is a affine proxy
                    # only involve diagonal elements about [i_global, i_global]
                    block = blocks[tid, i_local, i_local]
                    wp.atomic_add(
                        diag,
                        i_global,
                        wp.vec3(block[0, 0], block[1, 1], block[2, 2]),
                    )
                else:
                    # may involve diagonal elements about [i_global, j_global]
                    # if affine_i == affine_j >= 0
                    for j_local in range(4):
                        j_global = b_indices[tid, j_local]
                        if j_global < 0:
                            continue
                        if mask[j_global] == 0.0:
                            continue
                        affine_j = p_affine[j_global]

                        if affine_i == affine_j:
                            block = blocks[tid, i_local, j_local]
                            x_r_dot = wp.cw_mul(x_rest[i_global], x_rest[j_global])
                            block_id = abd_b_ids[affine_i]
                            wp.atomic_add(
                                diag,
                                b_indices[block_id, 3],
                                wp.vec3(block[0, 0], block[1, 1], block[2, 2]),
                            )
                            wp.atomic_add(
                                diag, b_indices[block_id, 0], x_r_dot * block[0, 0]
                            )
                            wp.atomic_add(
                                diag, b_indices[block_id, 1], x_r_dot * block[1, 1]
                            )
                            wp.atomic_add(
                                diag, b_indices[block_id, 2], x_r_dot * block[2, 2]
                            )


@wp.kernel
def compute_block_diag_inv_kernel(
    mask: wp.array(dtype=float),
    particle_affine: wp.array(dtype=int),
    diag: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    affine_id = particle_affine[tid]

    if mask[tid] != 0.0 and affine_id < 0:  # FEM vertices or affine proxies
        diag[tid] = wp.cw_div(wp.vec3(1.0), diag[tid])
    else:  # mask == 0.0 or affine_id >= 0 (ABD vertices)
        diag[tid] = wp.vec3(1.0)
    # diag[tid] = wp.vec3(1.0)

    z[tid] = wp.cw_mul(diag[tid], r[tid])


@wp.func
def get_mask(
    mask: wp.array(dtype=float),
    i: int,
):
    if i >= 0:
        return mask[i]
    return 0.0


@wp.kernel
def hess_block_mul_dx_kernel(
    n_particles: int,
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    m: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    dbc_mask: wp.array(dtype=float),
    kappa_con: float,
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    block_status: wp.array(dtype=wp.int32),
    dx: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    prod: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if tid >= n_particles and tid >= n_static_blocks + contact_counter[0]:
        return

    if tid < n_particles:
        dx_i = abd_apply(dx, x_rest, b_indices, abd_b_ids, p_affine, tid)
        # update = m[tid] * dx_i * mask[tid]
        m_i = wp.select(p_affine[tid] == -2, m[tid], 0.0)
        update = dx_i * (mask[tid] * m_i + kappa_con * dbc_mask[tid] * m[tid])
        # for affine proxies, only use mass (sum of vertex mass) for dbc
        abd_accum(
            prod,
            update,
            x_rest,
            b_indices,
            abd_b_ids,
            p_affine,
            tid,
        )

    if tid < n_static_blocks + contact_counter[0]:
        if block_status[tid] == 1:
            block = blocks[tid]

            i0 = b_indices[tid, 0]
            i1 = b_indices[tid, 1]
            i2 = b_indices[tid, 2]
            i3 = b_indices[tid, 3]

            dx0 = abd_apply(dx, x_rest, b_indices, abd_b_ids, p_affine, i0) * get_mask(
                mask, i0
            )
            dx1 = abd_apply(dx, x_rest, b_indices, abd_b_ids, p_affine, i1) * get_mask(
                mask, i1
            )
            dx2 = abd_apply(dx, x_rest, b_indices, abd_b_ids, p_affine, i2) * get_mask(
                mask, i2
            )
            dx3 = abd_apply(dx, x_rest, b_indices, abd_b_ids, p_affine, i3) * get_mask(
                mask, i3
            )

            if i0 >= 0:
                update = (
                    block[0, 0] * dx0
                    + block[0, 1] * dx1
                    + block[0, 2] * dx2
                    + block[0, 3] * dx3
                ) * mask[i0]
                abd_accum(
                    prod,
                    update,
                    x_rest,
                    b_indices,
                    abd_b_ids,
                    p_affine,
                    i0,
                )
            if i1 >= 0:
                update = (
                    block[1, 0] * dx0
                    + block[1, 1] * dx1
                    + block[1, 2] * dx2
                    + block[1, 3] * dx3
                ) * mask[i1]
                abd_accum(
                    prod,
                    update,
                    x_rest,
                    b_indices,
                    abd_b_ids,
                    p_affine,
                    i1,
                )
            if i2 >= 0:
                update = (
                    block[2, 0] * dx0
                    + block[2, 1] * dx1
                    + block[2, 2] * dx2
                    + block[2, 3] * dx3
                ) * mask[i2]
                abd_accum(
                    prod,
                    update,
                    x_rest,
                    b_indices,
                    abd_b_ids,
                    p_affine,
                    i2,
                )
            if i3 >= 0:
                update = (
                    block[3, 0] * dx0
                    + block[3, 1] * dx1
                    + block[3, 2] * dx2
                    + block[3, 3] * dx3
                ) * mask[i3]
                abd_accum(
                    prod,
                    update,
                    x_rest,
                    b_indices,
                    abd_b_ids,
                    p_affine,
                    i3,
                )
