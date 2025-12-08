import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)


@wp.kernel
def kinetic_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    particle_component: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    x_prev: wp.array(dtype=wp.vec3),
    v_prev: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    gravity: wp.vec3,
    h: float,
    energy: wp.array(dtype=float),
):
    tid = wp.tid()
    scene_id = particle_scene[tid]
    
    if mask[tid] == 0.0 or component_removed[particle_component[tid]] == 1:
        return
    
    if particle_affine[tid] == -2:
        return
    # Do not directly compute the kinetic energy of affine proxies.
    # Instead, compute the kinetic energy of the ABD vertices,
    # and accumulate the result to the proxies.
    
    x_i = abd_apply(
        x, x_rest, block_indices, affine_block_ids, particle_affine, tid
    )
    x_hat_i = x_prev[tid] + v_prev[tid] * h + gravity * h * h
    delta_x_i = x_i - x_hat_i
    # wp.printf("tid=%d, x_i: (%f %f %f), x_prev_i: (%f %f %f)\n", tid, x_i[0], x_i[1], x_i[2], x_prev[tid][0], x_prev[tid][1], x_prev[tid][2])
    wp.atomic_add(energy, scene_id, 0.5 * m[tid] * wp.dot(delta_x_i, delta_x_i))
    

@wp.kernel
def kinetic_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_component: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    x_prev: wp.array(dtype=wp.vec3),
    v_prev: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    mask: wp.array(dtype=float),
    gravity: wp.vec3,
    h: float,
    grad_coeff: float,
    grad: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    if mask[tid] == 0.0 or component_removed[particle_component[tid]] == 1:
        return

    if particle_affine[tid] == -2:
        return
    # Do not directly compute the kinetic energy of affine proxies.
    # Instead, compute the kinetic energy of the ABD vertices,
    # and accumulate the result to the proxies.
    
    x_i = abd_apply(
        x, x_rest, block_indices, affine_block_ids, particle_affine, tid
    )
    x_hat_i = x_prev[tid] + v_prev[tid] * h + gravity * h * h
    delta_x_i = x_i - x_hat_i
    grad_i = delta_x_i * (m[tid] * grad_coeff)
    abd_accum(
        grad, grad_i, x_rest, block_indices, affine_block_ids, particle_affine, tid
    )
    
    