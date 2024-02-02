import warp as wp
from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ...ipc_utils.global_defs import *
from .grad_funcs import *
from .hessian_funcs import *


@wp.func
def pt_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    pass


@wp.func
def ee_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    pass


@wp.func
def pe_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3):
    pass


@wp.func
def pp_distance(x0: wp.vec3, x1: wp.vec3):
    pass


@wp.func
def pt_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    pass


@wp.func
def pe_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3):
    pass


@wp.func
def ee_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    pass


@wp.func
def pt_pair_classify(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3):
    pass


@wp.func
def pt_pair_closest_point(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, pt_type: int):
    pass


@wp.func
def pt_pair_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, pt_type: int):
    pass


@wp.func
def pt_pair_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    dd_dx: wp.array(dtype=wp.vec3),
    pt_type: int,
):
    pass


@wp.func
def pt_pair_distance_hessian(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    kappa: float,
    db_dd: float,
    d2b_dd2: float,
    dd_dx: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=2),
    pt_type: int,
):
    pass


@wp.func
def ee_pair_classify(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, sin_thres: float
):
    pass


@wp.func
def ee_pair_distance(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, ee_type: int):
    pass


@wp.func
def ee_pair_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    dd_dx: wp.array(dtype=wp.vec3),
    ee_type: int,
):
    pass


@wp.func
def ee_pair_distance_hessian(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    kappa: float,
    d_hess_coeff: float,
    d_grad_coeff: float,
    dd_dx: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=2),
    ee_type: int,
):
    pass


