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
    # tid = wp.tid()
    # sid = tid / max_pairs_per_scene
    # tid = tid % max_pairs_per_scene
    # sur_particle_i = tid / n_surface_planes[sid]
    # sur_plane_i = tid % n_surface_planes[sid]

    sid, sur_particle_i, sur_plane_i = wp.tid()

    if sur_particle_i >= n_surface_particles[sid] or sur_plane_i >= n_surface_planes[sid]:
        return

    particle_i = surface_particles[sid, sur_particle_i]
    plane_i = surface_planes[sid, sur_plane_i]

    particle_component_i = particle_component[particle_i]
    plane_component_i = plane_component[plane_i]
    
    group0 = component_group[particle_component_i]
    group1 = component_group[plane_component_i]
    if group0 >= 0 and group0 == group1:
        # same group >= 0, but different components
        return

    # if particle_scene[particle_i] != plane_scene[plane_i]:
    #     return

    if (
        component_removed[particle_component_i] == 1
        or component_removed[plane_component_i] == 1
    ):
        return

    x_i = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, particle_i)
    plane_normal = plane_normals[plane_i]
    plane_offset = plane_offsets[plane_i]

    # d = wp.abs(x_i[2] - ground_altitude)
    d = wp.abs(wp.dot(x_i, plane_normal) - plane_offset)

    if d < d_hat + v_max * h:
        contact_id = wp.atomic_add(contact_counter, 0, 1)
        block_id = n_static_blocks + contact_id

        if block_id >= max_blocks:
            return

        b_indices[block_id, 0] = particle_i
        b_indices[block_id, 1] = -plane_i - 1  # negative, to distinguish from particles
        b_indices[block_id, 2] = -1
        b_indices[block_id, 3] = -1

        contact_type[block_id, 0] = PG_CONTACT_BLOCK  # point-plane

        # wp.printf("contact_type[%d, 0] = %d\n", block_id, contact_type[block_id, 0])

        contact_friction[block_id] = (
            component_friction[particle_component_i]
            + component_friction[plane_component_i]
        ) / 2.0


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
    # tid = wp.tid()
    # sid = tid / max_pairs_per_scene
    # tid = tid % max_pairs_per_scene
    # sur_par_id = tid / n_surface_triangles[sid]
    # sur_tri_id = tid % n_surface_triangles[sid]

    sid, sur_par_id, sur_tri_id = wp.tid()

    if sur_par_id >= n_surface_particles[sid] or sur_tri_id >= n_surface_triangles[sid]:
        return

    i0 = surface_particles[sid, sur_par_id]
    i1 = surface_triangles[sid, sur_tri_id, 0]
    i2 = surface_triangles[sid, sur_tri_id, 1]
    i3 = surface_triangles[sid, sur_tri_id, 2]

    # # wp.printf("i0: %d, i1: %d, i2: %d, i3: %d\n", i0, i1, i2, i3)

    # if particle_scene[i0] != particle_scene[i1]:
    #     return

    if i0 == i1 or i0 == i2 or i0 == i3:
        return

    if particle_component[i0] == particle_component[i1]:
        if not allow_self_collision or particle_affine[i0] >= 0:
            # not allow self collision, or same ABD component
            return
    else:
        group0 = component_group[particle_component[i0]]
        group1 = component_group[particle_component[i1]]
        if group0 >= 0 and group0 == group1:
            # same group >= 0, but different components
            return

    if (
        component_removed[particle_component[i0]] == 1
        or component_removed[particle_component[i1]] == 1
    ):
        return

    x0 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i0)
    x1 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i1)
    x2 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i2)
    x3 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i3)

    # # Bounding circle
    # x_c = (x1 + x2 + x3) / 3.0
    # r = wp.max(wp.max(wp.length(x1 - x_c), wp.length(x2 - x_c)), wp.length(x3 - x_c))
    # d_lower = wp.length(x0 - x_c) - r

    # d_plane = wp.abs(wp.dot(wp.normalize(wp.cross(x2 - x1, x3 - x1)), x0 - x1))
    # d_lower = wp.max(d_lower, d_plane)

    pt_type = pt_pair_classify(x0, x1, x2, x3)
    d_lower = pt_pair_distance(x0, x1, x2, x3, pt_type)

    if d_lower < d_hat + 2.0 * v_max * h:
        contact_id = wp.atomic_add(contact_counter, 0, 1)
        block_id = n_static_blocks + contact_id

        if block_id >= max_blocks:
            return

        b_indices[block_id, 0] = i0
        b_indices[block_id, 1] = i1
        b_indices[block_id, 2] = i2
        b_indices[block_id, 3] = i3

        contact_type[block_id, 0] = PT_CONTACT_BLOCK
        contact_friction[block_id] = (
            component_friction[particle_component[i0]]
            + component_friction[particle_component[i1]]
        ) / 2.0


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
    # tid = wp.tid()
    # sid = tid / max_pairs_per_scene
    # tid = tid % max_pairs_per_scene
    # edge_id0 = tid / n_surface_edges[sid]
    # edge_id1 = tid % n_surface_edges[sid]

    sid, edge_id0, edge_id1 = wp.tid()

    if edge_id0 >= n_surface_edges[sid] or edge_id1 >= n_surface_edges[sid]:
        return

    if edge_id0 >= edge_id1:
        return

    i0 = surface_edges[sid, edge_id0, 0]
    i1 = surface_edges[sid, edge_id0, 1]
    i2 = surface_edges[sid, edge_id1, 0]
    i3 = surface_edges[sid, edge_id1, 1]

    # if particle_scene[i0] != particle_scene[i2]:
    #     return

    if i0 == i2 or i0 == i3 or i1 == i2 or i1 == i3:
        return

    if particle_component[i0] == particle_component[i2]:
        if not allow_self_collision or particle_affine[i0] >= 0:
            # not allow self collision, or same ABD component
            return
    else:
        group0 = component_group[particle_component[i0]]
        group1 = component_group[particle_component[i2]]
        if group0 >= 0 and group0 == group1:
            # same group >= 0, but different components
            return

    if (
        component_removed[particle_component[i0]] == 1
        or component_removed[particle_component[i2]] == 1
    ):
        return

    x0 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i0)
    x1 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i1)
    x2 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i2)
    x3 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i3)

    # c0 = (x0 + x1) / 2.0
    # c1 = (x2 + x3) / 2.0

    # r0 = wp.length(x0 - c0)
    # r1 = wp.length(x2 - c1)

    # d_lower = wp.length(c0 - c1) - r0 - r1

    # # sphere c0 vs line x2-x3
    # d_line_0 = wp.length(wp.cross(x2 - c0, x3 - c0)) / wp.length(x3 - x2) - r0
    # # sphere c1 vs line x0-x1
    # d_line_1 = wp.length(wp.cross(x0 - c1, x1 - c1)) / wp.length(x1 - x0) - r1

    # d_lower = wp.max(d_lower, wp.max(d_line_0, d_line_1))

    ee_type = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
    d_lower = ee_pair_distance(x0, x1, x2, x3, ee_type)

    if d_lower < d_hat + 2.0 * v_max * h:
        contact_id = wp.atomic_add(contact_counter, 0, 1)
        block_id = n_static_blocks + contact_id

        if block_id >= max_blocks:
            return

        b_indices[block_id, 0] = i0
        b_indices[block_id, 1] = i1
        b_indices[block_id, 2] = i2
        b_indices[block_id, 3] = i3

        contact_type[block_id, 0] = EE_CONTACT_BLOCK
        contact_friction[block_id] = (
            component_friction[particle_component[i0]]
            + component_friction[particle_component[i2]]
        ) / 2.0
