import warp as wp


@wp.kernel
def affine_energy_kernel(
    x: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=wp.float32),
    affine_mass: wp.array(dtype=wp.mat33, ndim=3),
    x_prev: wp.array(dtype=wp.vec3),
    v_prev: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    gravity: wp.vec3,
    h: float,
    kappa_affine: wp.float32,
    energy: wp.array(dtype=wp.float32),
):
    affine_id = wp.tid()

    volume = volumes[affine_id]
    block_id = affine_block_ids[affine_id]
    
    if b_status[block_id] == 0:
        return
    
    i0 = b_indices[block_id, 0]

    if mask[i0] == 0.0:
        return
    
    scene_id = particle_scene[i0]

    E = 0.0

    for a in range(4):
        for b in range(4):
            i_a = b_indices[block_id, a]
            i_b = b_indices[block_id, b]

            delta_x_a = x[i_a] - (x_prev[i_a] + v_prev[i_a] * h)
            delta_x_b = x[i_b] - (x_prev[i_b] + v_prev[i_b] * h)

            if a == 3:
                delta_x_a -= gravity * h * h
            if b == 3:
                delta_x_b -= gravity * h * h
            
            E += wp.dot(delta_x_a, affine_mass[affine_id, a, b] * delta_x_b) * 0.5

    for a in range(3):
        for b in range(3):
            i_a = b_indices[block_id, a]
            i_b = b_indices[block_id, b]

            x_a = x[i_a]
            x_b = x[i_b]

            if a == b:
                E += kappa_affine * volume * (wp.dot(x_a, x_a) - 1.0) ** 2.0
            else:
                E += kappa_affine * volume * wp.dot(x_a, x_b) ** 2.0

    wp.atomic_add(energy, scene_id, E)


@wp.kernel
def affine_diff_kernel(
    x: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    b_status: wp.array(dtype=wp.int32),
    volumes: wp.array(dtype=wp.float32),
    affine_mass: wp.array(dtype=wp.mat33, ndim=3),
    x_prev: wp.array(dtype=wp.vec3),
    v_prev: wp.array(dtype=wp.vec3),
    mask: wp.array(dtype=float),
    gravity: wp.vec3,
    h: float,
    kappa_affine: wp.float32,
    coeff: wp.float32,
    grad: wp.array(dtype=wp.vec3),
    blocks: wp.array(dtype=wp.mat33, ndim=3),
):
    affine_id = wp.tid()

    volume = volumes[affine_id]
    block_id = affine_block_ids[affine_id]
    
    if b_status[block_id] == 0:
        return
    
    if mask[b_indices[block_id, 0]] == 0.0:
        return

    eye3 = wp.identity(n=3, dtype=wp.float32)

    for a in range(4):
        for b in range(4):
            i_a = b_indices[block_id, a]
            i_b = b_indices[block_id, b]

            delta_x_b = x[i_b] - (x_prev[i_b] + v_prev[i_b] * h)
            if b == 3:
                delta_x_b -= gravity * h * h

            wp.atomic_add(
                grad,
                i_a,
                affine_mass[affine_id, a, b] * delta_x_b * coeff,
            )
            blocks[block_id, a, b] = affine_mass[affine_id, a, b]

    for a in range(3):
        for b in range(3):
            i_a = b_indices[block_id, a]
            i_b = b_indices[block_id, b]

            x_a = x[i_a]
            x_b = x[i_b]

            if a == b:
                wp.atomic_add(
                    grad,
                    i_a,
                    x_a
                    * (4.0 * kappa_affine * volume * (wp.dot(x_a, x_a) - 1.0) * coeff),
                )
                wp.atomic_add(
                    blocks,
                    block_id,
                    a,
                    a,
                    (eye3 * (wp.dot(x_a, x_a) - 1.0) + wp.outer(x_a, x_a) * 2.0)
                    * (4.0 * kappa_affine * volume),
                )
            else:
                wp.atomic_add(
                    grad,
                    i_a,
                    x_b * (4.0 * kappa_affine * volume * wp.dot(x_a, x_b) * coeff),
                )
                wp.atomic_add(
                    blocks,
                    block_id,
                    a,
                    b,
                    (eye3 * (wp.dot(x_a, x_b)) + wp.outer(x_b, x_a))
                    * (4.0 * kappa_affine * volume),
                )
                wp.atomic_add(
                    blocks,
                    block_id,
                    a,
                    a,
                    wp.outer(x_b, x_b) * (4.0 * kappa_affine * volume),
                )
