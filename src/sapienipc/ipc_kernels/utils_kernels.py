import warp as wp
from ..ipc_utils.global_defs import *
from ..ipc_kernels.abd_utils_kernels import apply_parent_affine_func


@wp.kernel
def init_random_states(states: wp.array(dtype=wp.uint32), seed: int):
    pass


@wp.kernel
def set_to_ones_at_indices_int_kernel(
    arr: wp.array(dtype=int),
    indices: wp.array(dtype=int),
):
    pass


@wp.kernel
def set_values_at_indices_vec3_kernel(
    arr: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    values: wp.array(dtype=wp.vec3),
):
    pass


@wp.kernel
def add_random_noise_kernel(
    x: wp.array(dtype=wp.vec3),
    states: wp.array(dtype=wp.uint32),
    mean: wp.vec3,
    std: float,
    mask: wp.array(dtype=float),
):
    pass


@wp.kernel
def a_plus_k_b_kernel(
    a: wp.array(dtype=wp.vec3),
    k: float,
    b: wp.array(dtype=wp.vec3),
    ret: wp.array(dtype=wp.vec3),
):
    pass


@wp.kernel
def masked_diff_kernel(
    a: wp.array(dtype=wp.vec3),
    b: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=int),
    ret: wp.array(dtype=wp.vec3),
):
    pass


@wp.kernel
def compute_velocity_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    x_next: wp.array(dtype=wp.vec3),
    h: float,
    v_next: wp.array(dtype=wp.vec3),
):
    pass


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
    pass


@wp.kernel
def block_spd_project_kernel(
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    block_type: wp.array(dtype=int, ndim=2),
    block_status: wp.array(dtype=int),
    it_max: int,
):
    pass


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
    pass


@wp.kernel
def clip_velocity_kernel(
    particle_affine: wp.array(dtype=int),
    p: wp.array(dtype=wp.vec3),
    x_prev_it: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    r: float,
):
    pass


