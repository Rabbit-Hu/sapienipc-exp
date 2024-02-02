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
    pass


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
    pass


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
    pass


