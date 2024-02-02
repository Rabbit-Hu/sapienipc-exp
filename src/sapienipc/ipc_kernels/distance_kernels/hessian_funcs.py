import warp as wp


@wp.func
def pp_distance_hessian(
        x0: wp.vec3, x1: wp.vec3,
        i0: int, i1: int,  # in {0, 1, 2, 3}
        kappa: float,
        db_dd: float,
        d2b_dd2: float,
        dd_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    pass


@wp.func
def pe_distance_hessian(
        x0: wp.vec3, x1: wp.vec3, x2: wp.vec3,
        i0: int, i1: int, i2: int,  # in {0, 1, 2, 3}
        kappa: float, db_dd: float, d2b_dd2: float,
        dd_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    pass


@wp.func
def pt_distance_hessian(
        x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3,
        kappa: float, db_dd: float, d2b_dd2: float,
        dd_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    pass


@wp.func
def ee_distance_hessian(
        x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3,
        kappa: float, db_dd: float, d2b_dd2: float,
        dd_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    pass


