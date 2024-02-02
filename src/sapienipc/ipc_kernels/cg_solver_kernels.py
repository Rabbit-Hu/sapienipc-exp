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
    pass


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
    pass


@wp.kernel
def update_r_2_best_kernel(
    r_2: wp.array(dtype=float),
    r_2_best: wp.array(dtype=float),
):
    pass


@wp.kernel
def compute_dot_kernel(
    x: wp.array(dtype=wp.vec3), 
    y: wp.array(dtype=wp.vec3), 
    particle_scene: wp.array(dtype=int),
    ret: wp.array(dtype=float)
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
def compute_velocity_kernel(
    x_prev: wp.array(dtype=wp.vec3),
    x_next: wp.array(dtype=wp.vec3),
    dt: float,
    v_next: wp.array(dtype=wp.vec3),
):
    pass


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
    pass


@wp.kernel
def compute_block_diag_inv_kernel(
    mask: wp.array(dtype=float),
    particle_affine: wp.array(dtype=int),
    diag: wp.array(dtype=wp.vec3),
    r: wp.array(dtype=wp.vec3),
    z: wp.array(dtype=wp.vec3),
):
    pass


@wp.func
def get_mask(
    mask: wp.array(dtype=float),
    i: int,
):
    pass


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
    pass


