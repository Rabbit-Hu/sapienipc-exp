import warp as wp


@wp.func
def pp_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    i0: int,
    i1: int,  # i0, i1 in {0, 1, 2, 3}
    dd_dx: wp.array(dtype=wp.vec3),
):
    pass


@wp.func
def ee_mollifier_grad(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, dc_dx: wp.array(dtype=wp.vec3)
):
    pass


@wp.func
def ee_distance_grad(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, dd_dx: wp.array(dtype=wp.vec3)
):
    pass


@wp.func
def pt_distance_grad(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, dd_dx: wp.array(dtype=wp.vec3)
):
    pass


@wp.func
def pe_distance_grad(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    i0: int,
    i1: int,
    i2: int,
    dd_dx: wp.array(dtype=wp.vec3),
):
    pass


