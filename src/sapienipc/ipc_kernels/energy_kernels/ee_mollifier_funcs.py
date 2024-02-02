import warp as wp


@wp.func
def ee_pair_mollifier_grad(
    x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3, dc_dx: wp.array(dtype=wp.vec3)
):
    pass


@wp.func
def ee_pair_mollifier_hessian(
        x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3,
        kappa: float, c_hessian_coeff: float, c_grad_coeff: float,
        dc_dx: wp.array(dtype=wp.vec3),
        energy_hessian: wp.array(dtype=wp.mat33, ndim=2)
):
    pass


