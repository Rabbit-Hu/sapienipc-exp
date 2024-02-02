import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ...ipc_utils.global_defs import *
from ..distance_kernels.distance_kernels import (
    pt_pair_classify,
    pt_pair_distance,
    pt_pair_distance_grad,
    pt_pair_distance_hessian,
    ee_pair_classify,
    ee_pair_distance,
    ee_pair_distance_grad,
    ee_pair_distance_hessian,
)
from .ee_mollifier_funcs import (
    ee_pair_mollifier_grad,
    ee_pair_mollifier_hessian,
)


@wp.func
def compute_b(d: float, d_hat: float):
    pass


@wp.func
def compute_db_dd(d: float, d_hat: float):
    pass


@wp.func
def compute_d2b_dd2(d: float, d_hat: float):
    pass


@wp.func
def compute_e(c: float, eps_cross: float):
    pass


@wp.func
def compute_de_dc(c: float, eps_cross: float):
    pass


@wp.func
def compute_d2e_dc2(c: float, eps_cross: float):
    pass


@wp.kernel
def collision_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    kappa: float,
    d_hat: float,
    ee_classify_thres: float,
    ee_mollifier_thres: float,
    contact_d: wp.array(dtype=float),
    contact_c: wp.array(dtype=float),
    contact_eps_cross: wp.array(dtype=float),
    block_status: wp.array(dtype=wp.int32),
    energy: wp.array(dtype=float),
):
    pass


@wp.kernel
def collision_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    n_static_blocks: int,
    contact_counter: wp.array(dtype=wp.int32),
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    abd_b_ids: wp.array(dtype=wp.int32),
    p_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    kappa: float,
    d_hat: float,
    h: float,
    contact_d: wp.array(dtype=float),
    contact_c: wp.array(dtype=float),
    contact_eps_cross: wp.array(dtype=float),
    coeff: float,
    contact_dd_dx: wp.array(dtype=wp.vec3, ndim=2),
    contact_dc_dx: wp.array(dtype=wp.vec3, ndim=2),
    grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
    particle_collision_force: wp.array(dtype=wp.vec3),
):
    pass


