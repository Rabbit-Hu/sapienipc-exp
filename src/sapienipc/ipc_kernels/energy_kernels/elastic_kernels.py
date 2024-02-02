import warp as wp
from ...ipc_utils.warp_types import mat32


@wp.kernel
def elastic_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat33),
    rest_volumes: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    energy: wp.array(dtype=float),
):
    pass


@wp.func
def _elas_hess_func(  # Elasetic Energy Hessian block
    i: int,
    j: int,
    k: int,
    l: int,
    k_mu: float,
    k_la: float,
    log_J: float,
    F_inv_T_inv_Dm_T: wp.mat33,
    inv_Dm_inv_Dm_T: wp.mat33,
):
    pass


@wp.kernel
def elastic_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tet_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat33),
    rest_volumes: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    pass


@wp.kernel
def elastic_energy_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat22),
    rest_areas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    energy: wp.array(dtype=float),
):
    pass


@wp.func
def _elas_hess_2d_func(  # Elasetic Energy Hessian block
    i: int,
    j: int,
    k: int,
    l: int,
    k_mu: float,
    k_la: float,
    log_J: float,
    F_inv_T_inv_Dm_T: wp.mat22,
    inv_Dm_inv_Dm_T: wp.mat22,
):
    pass


@wp.kernel
def elastic_diff_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    inv_Dms: wp.array(dtype=wp.mat22),
    rest_areas: wp.array(dtype=float),
    thicknesses: wp.array(dtype=float),
    materials: wp.array(dtype=float, ndim=2),
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    pass


@wp.kernel
def bending_energy_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    hinge_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    rest_angles: wp.array(dtype=float),
    materials: wp.array(dtype=float),  # k_bend
    h: float,
    energy: wp.array(dtype=float),
):
    pass


@wp.func
def _symmetric(A: wp.mat33):
    pass


@wp.kernel
def bending_diff_2d_kernel(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    hinge_b_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    rest_angles: wp.array(dtype=float),
    materials: wp.array(dtype=float),  # k_bend
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    pass


