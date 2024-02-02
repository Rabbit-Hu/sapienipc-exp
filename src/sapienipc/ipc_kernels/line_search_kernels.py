import warp as wp


@wp.kernel
def line_search_iteration_kernel(
    x: wp.array(dtype=wp.vec3),
    p: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=float),
    n_halves: float,
    particle_scene: wp.array(dtype=int),
    energy_prev: wp.array(dtype=float),
    energy: wp.array(dtype=float),
    x_new: wp.array(dtype=wp.vec3),
):
    pass


