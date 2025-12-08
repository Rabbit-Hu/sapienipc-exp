import warp as wp


@wp.func
def apply_parent_affine_func(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    i: int,
):
    """Compute x = J(x_rest) q"""
    
    if i < 0:
        return wp.vec3(0.0, 0.0, 0.0)

    affine_id = particle_affine[i]

    if affine_id < 0:  # FEM vertex or ABD proxy
        return x[i]

    block_id = affine_block_ids[affine_id]
    p_id = block_indices[block_id, 3]
    a1_id = block_indices[block_id, 0]
    a2_id = block_indices[block_id, 1]
    a3_id = block_indices[block_id, 2]

    return (
        wp.vec3(
            wp.dot(x[a1_id], x_rest[i]),
            wp.dot(x[a2_id], x_rest[i]),
            wp.dot(x[a3_id], x_rest[i]),
        )
        + x[p_id]
    )


@wp.func
def accumulate_on_proxies_func(
    target: wp.array(dtype=wp.vec3),
    update: wp.vec3,
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    i: int,
):
    """Accumulate (J^T * upd) on the 4 proxies"""
    
    if update == wp.vec3(0.0, 0.0, 0.0):
        return
    
    if i < 0:
        return

    affine_id = particle_affine[i]

    if affine_id < 0:
        wp.atomic_add(target, i, update)
        return

    block_id = affine_block_ids[affine_id]
    p_id = block_indices[block_id, 3]
    a1_id = block_indices[block_id, 0]
    a2_id = block_indices[block_id, 1]
    a3_id = block_indices[block_id, 2]

    x_rest_i = x_rest[i]

    wp.atomic_add(target, p_id, update)
    wp.atomic_add(target, a1_id, x_rest_i * update[0])
    wp.atomic_add(target, a2_id, x_rest_i * update[1])
    wp.atomic_add(target, a3_id, x_rest_i * update[2])


@wp.func
def accumulate_on_proxies_masked_func(
    target: wp.array(dtype=wp.vec3),
    update: wp.vec3,
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i: int,
):
    """Accumulate (J^T * upd) on the 4 proxies"""
    
    if update == wp.vec3(0.0, 0.0, 0.0):
        return
    
    if i < 0:
        return

    affine_id = particle_affine[i]

    if affine_id < 0:
        wp.atomic_add(target, i, update)
        return

    block_id = affine_block_ids[affine_id]
    p_id = block_indices[block_id, 3]
    a1_id = block_indices[block_id, 0]
    a2_id = block_indices[block_id, 1]
    a3_id = block_indices[block_id, 2]

    x_rest_i = x_rest[i]

    wp.atomic_add(target, p_id, update * mask[p_id])
    wp.atomic_add(target, a1_id, x_rest_i * update[0] * mask[a1_id])
    wp.atomic_add(target, a2_id, x_rest_i * update[1] * mask[a2_id])
    wp.atomic_add(target, a3_id, x_rest_i * update[2] * mask[a3_id])


@wp.kernel
def apply_parent_affine_kernel(
    begin_id: int,
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
):
    i = wp.tid() + begin_id

    x[i] = apply_parent_affine_func(
        x, x_rest, block_indices, affine_block_ids, particle_affine, i
    )
    