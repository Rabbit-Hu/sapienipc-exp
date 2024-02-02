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
    pass


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
    pass


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
    pass


@wp.kernel
def apply_parent_affine_kernel(
    begin_id: int,
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    block_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
):
    pass


