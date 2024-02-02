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
    pass


@wp.func
def compute_f1(x: float, epsv_h: float):
    pass


@wp.func
def compute_f1_over_x(x: float, epsv_h: float):
    pass


@wp.func
def compute_f2(x: float, epsv_h: float):
    pass


@wp.func
def compute_f2_x_minus_f1_over_x_sq(x: float, epsv_h: float):
    pass


@wp.func
def cols_to_mat32(col1: wp.vec3, col2: wp.vec3):
    pass


@wp.func
def pp_slide_basis(
    x0: wp.vec3, x1: wp.vec3, i0: int, i1: int, T_block: wp.array(dtype=mat32)
):
    pass


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
    pass


@wp.func
def pt_slide_basis(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, T_block: wp.array(dtype=mat32)
):
    pass


@wp.func
def ee_slide_basis(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, T_block: wp.array(dtype=mat32)
):
    pass


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
    pass


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
    pass


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
    pass


# @wp.kernel
# def test_kernel(
#     a: wp.array(dtype=mat32),
#     b: wp.array(dtype=wp.mat22),
# ):
    pass


#     import numpy as np


