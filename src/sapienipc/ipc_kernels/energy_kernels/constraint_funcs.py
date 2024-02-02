import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_masked_func as abd_accum_m
)


@wp.func
def angular_derivative_control_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    kd: float,
    omega_target: float,
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    pass


@wp.func
def _symmetric(A: wp.mat33):
    pass


@wp.func
def angular_proportional_control_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    kp: float,
    theta_target: float,
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    pass


@wp.func
def area_constrant_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    area_target: float,
    mu: float,
    la: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    pass


