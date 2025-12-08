import warp as wp
from ..ipc_utils.global_defs import *
from ..ipc_kernels.abd_utils_kernels import apply_parent_affine_func


@wp.kernel
def init_random_states(states: wp.array(dtype=wp.uint32), seed: int):
    tid = wp.tid()
    states[tid] = wp.rand_init(seed, tid)


@wp.kernel
def set_to_ones_at_indices_int_kernel(
    arr: wp.array(dtype=int),
    indices: wp.array(dtype=int),
):
    tid = wp.tid()
    arr[indices[tid]] = 1


@wp.kernel
def set_values_at_indices_vec3_kernel(
    arr: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    values: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    arr[indices[tid]] = values[tid]


@wp.kernel
def add_random_noise_kernel(
    x: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.uint32),
    mean: wp.vec3,
    std: float,
    mask: wp.array(dtype=float),
):
    tid = wp.tid()
    state = states[tid]
    x[tid] = (
        x[tid]
        + wp.vec3(wp.randn(state), wp.randn(state),
                  wp.randn(state)) * std * mask[tid]
        + mean * mask[tid]
    )
    states[tid] = state


@wp.kernel
def a_plus_k_b_kernel(
    a: wp.array(dtype=wp.vec3),
    k: float,
    b: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3),
):
    """
    input: a, k, b
    output: ret = a + k * b
    """
    tid = wp.tid()

    ret[tid] = a[tid] + k * b[tid]


@wp.kernel
def masked_diff_kernel(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=int),
    ret: wp.array(dtype=wp.vec3),
):
    """
    input: a, k, b
    output: ret = (a - b) * mask
    """
    tid = wp.tid()

    ret[tid] = wp.select(mask[tid], wp.vec3(0.), a[tid] - b[tid])


@wp.kernel
def compute_velocity_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    x_next: wp.array(dtype=wp.vec3),
    h: float,
    v_next: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v_next[tid] = (x_next[tid] - x_prev[tid]) / h


# @wp.kernel
# def apply_affine_compute_velocity_kernel(
#     x: wp.array(dtype=wp.vec3),
#     x_prev: wp.array(dtype=wp.vec3),
#     x_rest: wp.array(dtype=wp.vec3),
#     block_indices: wp.array(dtype=int, ndim=2),
#     affine_block_ids: wp.array(dtype=int),
#     particle_affine: wp.array(dtype=int),
#     h: float,
#     v_max: float,
#     v: wp.array(dtype=wp.vec3),
# ):
#     i = wp.tid()

#     x[i] = apply_parent_affine_func(
#         x, x_rest, block_indices, affine_block_ids, particle_affine, i
#     )

#     v[i] = (x[i] - x_prev[i]) / h

#     if particle_affine[i] != -2 and wp.length(v[i]) > v_max:

#         wp.print("Warning: velocity v[%d] = %.3e exceeds v_max (particle_is_ABD = %d).\n", v, wp.length(
#             v[i]), particle_affine[i] >= 0)


@wp.kernel
def block_spd_project_kernel(
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    block_type: wp.array(dtype=int, ndim=2),
    block_status: wp.array(dtype=int),
    it_max: int,
):
    tid = wp.tid()

    if block_status[tid] != 1:
        return

    if block_type[tid, 0] == ABD_AFFINE_BLOCK:
        return  # affine coordinates is not translational invariant

    block = blocks[tid]

    wp.spd_project_blocks(3, block, it_max)

    block[3][0] = -block[0][0] - block[1][0] - block[2][0]
    block[3][1] = -block[0][1] - block[1][1] - block[2][1]
    block[3][2] = -block[0][2] - block[1][2] - block[2][2]

    block[0][3] = wp.transpose(block[3][0])
    block[1][3] = wp.transpose(block[3][1])
    block[2][3] = wp.transpose(block[3][2])

    block[3][3] = -block[0][3] - block[1][3] - block[2][3]


@wp.kernel
def process_dbc_kernel(
    ccd_step: wp.array(dtype=float),
    particle_scene: wp.array(dtype=int),
    block_indices: wp.array(dtype=int, ndim=2),
    affine_block_ids: wp.array(dtype=int),
    particle_affine: wp.array(dtype=int),
    dbc_delta_q: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    dbc_tag: wp.array(dtype=int),
    dbc_mask: wp.array(dtype=float),
    particle_mask: wp.array(dtype=float),
):
    tid = wp.tid()

    scene_id = particle_scene[tid]

    ccd_thres = 0.999

    affine_id = particle_affine[tid]
    if affine_id >= 0:  # controlled by affine
        block_id = affine_block_ids[affine_id]
        ids = block_indices[block_id]
        if dbc_tag[ids[0]] != 0:
            if dbc_tag[ids[0]] == 1 and ccd_step[scene_id] < ccd_thres:
                particle_mask[tid] = 1.0
                # do not compute constraint energy directly on the controlled vertices
                dbc_mask[tid] = 0.0
            else:
                particle_mask[tid] = 0.0
                dbc_mask[tid] = 0.0
        else:
            particle_mask[tid] = 1.0
            dbc_mask[tid] = 0.0
    else:
        if dbc_tag[tid] != 0:
            if wp.length(dbc_delta_q[tid]) > 1e-6 and ccd_step[scene_id] < ccd_thres:
                particle_q[tid] = particle_q[tid] + \
                    dbc_delta_q[tid] * ccd_step[scene_id]
                particle_mask[tid] = 1.0
                dbc_mask[tid] = 1.0
            else:
                particle_q[tid] = particle_q[tid] + dbc_delta_q[tid]
                particle_mask[tid] = 0.0
                dbc_mask[tid] = 0.0
        else:
            particle_mask[tid] = 1.0
            dbc_mask[tid] = 0.0

        # if affine_id == -2:  # ABD proxy
        #     wp.printf("mask[%d] = %.3e\n", tid, particle_mask[tid])


@wp.kernel
def clip_velocity_kernel(
    particle_affine: wp.array(dtype=int),
    p: wp.array(dtype=wp.vec3),
    x_prev_it: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    r: float,
):
    """Clip p such that ||p + x_prev_it - x_prev_step|| <= r^2 """

    tid = wp.tid()

    if particle_affine[tid] != -1:  # only clip FEM particles
        return
    
    if wp.length_sq(p[tid]) == 0.0:
        return

    dxi = x_prev_it[tid] - x_prev_step[tid]
    pi = p[tid]

    A = wp.dot(pi, pi)
    B = 2.0 * wp.dot(pi, dxi)
    C = wp.dot(dxi, dxi) - r * r

    if C > 0.0:
        wp.printf("[clip_velocity_kernel] Error: v_prev_it > v_max, C = ||dx||^2 - ||p||^2 = %.3e\n", C)
        return
    
    delta = B * B - 4.0 * A * C\
    
    if delta < 0.0:
        wp.printf("[clip_velocity_kernel] Error: delta = %.3e\n", delta)
        return

    delta = delta / wp.sqrt(delta)
    eps = 1e-6 / wp.length(pi)
    t2 = (-B + delta) / (2.0 * A)

    # wp.printf("[clip_velocity_kernel] t2 = %.3e\n", t2)
    
    p[tid] = pi * wp.min(wp.max(t2 - eps, 0.0), 1.0)
    
