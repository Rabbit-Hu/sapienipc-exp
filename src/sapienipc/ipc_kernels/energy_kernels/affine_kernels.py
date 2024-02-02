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
    pass


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
    pass


