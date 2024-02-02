import warp as wp
from .abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ..ipc_kernels.distance_kernels.distance_kernels import *
from ..ipc_utils.global_defs import *


@wp.kernel
def pg_filter_kernel(
    contact_counter: wp.array(dtype=int),
    n_static_blocks: int,
    max_blocks: int,
    max_pairs_per_scene: int,
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    n_surface_particles: wp.array(dtype=wp.int32),
    n_surface_planes: wp.array(dtype=wp.int32),
    surface_particles: wp.array(dtype=wp.int32, ndim=2),
    surface_planes: wp.array(dtype=wp.int32, ndim=2),
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    particle_component: wp.array(dtype=wp.int32),
    plane_component: wp.array(dtype=wp.int32),
    particle_scene: wp.array(dtype=wp.int32),
    plane_scene: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    component_friction: wp.array(dtype=float),
    component_group: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    d_hat: float,
    h: float,
    v_max: float,
    contact_type: wp.array(dtype=int, ndim=2),
    contact_friction: wp.array(dtype=float),
):
    pass


@wp.kernel
def pt_filter_kernel(
    contact_counter: wp.array(dtype=int),
    n_static_blocks: int,
    max_blocks: int,
    max_pairs_per_scene: int,
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    n_surface_particles: wp.array(dtype=wp.int32),
    n_surface_triangles: wp.array(dtype=wp.int32),
    surface_particles: wp.array(dtype=wp.int32, ndim=2),
    surface_triangles: wp.array(dtype=wp.int32, ndim=3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    particle_component: wp.array(dtype=wp.int32),
    particle_scene: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    component_friction: wp.array(dtype=float),
    component_group: wp.array(dtype=wp.int32),
    d_hat: float,
    h: float,
    v_max: float,
    allow_self_collision: bool,
    contact_type: wp.array(dtype=int, ndim=2),
    contact_friction: wp.array(dtype=float),
):
    pass


@wp.kernel
def ee_filter_kernel(
    contact_counter: wp.array(dtype=int),
    n_static_blocks: int,
    max_blocks: int,
    max_pairs_per_scene: int,
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    n_surface_edges: wp.array(dtype=wp.int32),
    surface_edges: wp.array(dtype=wp.int32, ndim=3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    particle_component: wp.array(dtype=wp.int32),
    particle_scene: wp.array(dtype=wp.int32),
    component_removed: wp.array(dtype=wp.int32),
    component_friction: wp.array(dtype=float),
    component_group: wp.array(dtype=wp.int32),
    d_hat: float,
    h: float,
    v_max: float,
    ee_classify_thres: float,
    allow_self_collision: bool,
    contact_type: wp.array(dtype=int, ndim=2),
    contact_friction: wp.array(dtype=float),
):
    pass


