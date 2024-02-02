import warp as wp
from .abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ..ipc_utils.global_defs import *
from .distance_kernels.distance_kernels import (
    pt_pair_classify,
    pt_pair_distance,
    ee_pair_classify,
    ee_pair_distance,
)


@wp.func
def cubic_ccd_check_solution(
    a: float, b: float, c: float, d: float, s: float, x: float
):
    pass


@wp.func
def cubic_ccd(
    a: float,
    b: float,
    c: float,
    d: float,
):
    pass


@wp.func
def tet_ccd(  # prevent inversion
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    v3: wp.vec3,
    tet_inversion_thres: float,
):
    pass


@wp.kernel
def ccd_kernel(  # One kernel for all contact types
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    n_static_blocks: int,
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    slackness: float,
    thickness: float,
    max_iters: int,
    tet_inversion_thres: float,
    ee_classify_thres: float,
    ccd_step: wp.array(dtype=float),
):
    pass


