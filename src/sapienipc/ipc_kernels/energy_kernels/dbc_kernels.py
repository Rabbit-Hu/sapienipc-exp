import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)


@wp.kernel
def dbc_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    particle_component: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    m: wp.array(dtype=float),
    con_mask: wp.array(dtype=float),
    con_q: wp.array(dtype=wp.vec3),
    kappa_con: float,
    energy: wp.array(dtype=float),
):
    pass


@wp.kernel
def dbc_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_component: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    m: wp.array(dtype=float),
    con_mask: wp.array(dtype=float),
    con_q: wp.array(dtype=wp.vec3),
    kappa_con: float,
    grad_coeff: float,
    grad: wp.array(dtype=wp.vec3),
):
    pass


