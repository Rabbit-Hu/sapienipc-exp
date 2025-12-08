import warp as wp
from .abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_func as abd_accum,
)
from ..ipc_utils.global_defs import *
from .distance_kernels.distance_kernels import (
    pt_pair_classify,
    pt_pair_distance,
    ee_pair_classify,
    ee_pair_distance,
)


@wp.func
def cubic_ccd_check_solution(
    a: float, b: float, c: float, d: float, s: float, x: float
):
    """if x >= 0. and f'(x) < 0, update s = min(s, x)"""
    return wp.select(
        x >= 0.0 and 3.0 * a * x * x + 2.0 * b * x + c < 0.0, s, wp.min(s, x)
    )


@wp.func
def cubic_ccd(
    a: float,
    b: float,
    c: float,
    d: float,
):
    if d <= 0: 
        wp.printf("[ccd] d <= 0! d = %f\n", d)
        return 0.
    
    s = float(1e9)  # step size to return
    
    x1 = float(1e9)
    x2 = float(1e9)
    x3 = float(1e9)
    
    a_abs = wp.abs(a)
    b_abs = wp.abs(b)
    c_abs = wp.abs(c)
    num_eps = 1e-5

    if a_abs <= num_eps and b_abs <= num_eps:  # Case for handling Liner Equation
        if c_abs > num_eps:
            s = cubic_ccd_check_solution(a, b, c, d, s, -d / c)
        if s < 1e-6:
            wp.printf("[ccd] cubic ccd step too small: %.3e (x = -1*%.3e/%.3e)\n", s, d, c)
        return s

    elif a_abs <= num_eps:  # Case for handling Quadratic Equations
        D = c * c - 4.0 * b * d
        if D >= 0:
            D = wp.sqrt(D)
            x1 = (-c + D) / (2.0 * b)
            x2 = (-c - D) / (2.0 * b)
            s = cubic_ccd_check_solution(a, b, c, d, s, x1)
            s = cubic_ccd_check_solution(a, b, c, d, s, x2)
        else:
            s = wp.select(d <= 0, s, 0.0)
        if s < 1e-6:
            wp.printf("[ccd] cubic ccd step too small: %.3e (x1=%.3e, x2=%.3e, a=%.3e, b=%.3e, c=%.3e, d=%.3e)\n", s, x1, x2, a, b, c, d)
        return s

    f = ((3.0 * c / a) - ((b**2.0) / (a**2.0))) / 3.0
    g = (
        ((2.0 * (b**3.0)) / (a**3.0))
        - ((9.0 * b * c) / (a**2.0))
        + (27.0 * d / a)
    ) / 27.0
    h = (g**2.0) / 4.0 + (f**3.0) / 27.0

    if wp.abs(f) <= num_eps and wp.abs(g) <= num_eps and wp.abs(h) <= num_eps:  # All 3 Roots are Real and Equal
        if (d / a) >= 0:
            x1 = (d / (1.0 * a)) ** (1.0 / 3.0) * -1.0
        else:
            x1 = (-d / (1.0 * a)) ** (1.0 / 3.0)

        s = cubic_ccd_check_solution(a, b, c, d, s, x1)

    elif h <= 0:  # All 3 roots are Real
        i = wp.sqrt(((g**2.0) / 4.0) - h)
        j = i ** (1.0 / 3.0)
        k = wp.acos(-(g / (2.0 * i)))
        L = j * -1.0
        M = wp.cos(k / 3.0)
        N = wp.sqrt(3.0) * wp.sin(k / 3.0)
        P = (b / (3.0 * a)) * -1.0

        x1 = 2.0 * j * wp.cos(k / 3.0) - (b / (3.0 * a))
        x2 = L * (M + N) + P
        x3 = L * (M - N) + P

        s = cubic_ccd_check_solution(a, b, c, d, s, x1)
        s = cubic_ccd_check_solution(a, b, c, d, s, x2)
        s = cubic_ccd_check_solution(a, b, c, d, s, x3)

    elif h > 0:  # One Real Root and two Complex Roots
        R = -(g / 2.0) + wp.sqrt(h)
        if R >= 0:
            S = R ** (1.0 / 3.0)
        else:
            S = (-R) ** (1.0 / 3.0) * -1.0
        T = -(g / 2.0) - wp.sqrt(h)
        if T >= 0.0:
            U = T ** (1.0 / 3.0)
        else:
            U = ((-T) ** (1.0 / 3.0)) * -1.0

        x1 = (S + U) - (b / (3.0 * a))

        s = cubic_ccd_check_solution(a, b, c, d, s, x1)
        
    if s < 1e-6:
        wp.printf("[ccd] cubic ccd step too small: %.3e (x1=%.3e, x2=%.3e, x3=%.3e)\n", s, x1, x2, x3)

    return s


@wp.func
def tet_ccd(  # prevent inversion
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    v3: wp.vec3,
    tet_inversion_thres: float,
):
    x1 = x1 - x0
    x2 = x2 - x0
    x3 = x3 - x0
    X = wp.mat33(x1[0], x2[0], x3[0], x1[1], x2[1], x3[1], x1[2], x2[2], x3[2])

    v1 = v1 - v0
    v2 = v2 - v0
    v3 = v3 - v0
    V = wp.mat33(v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2])

    # Inequality: det(X + V * t) > tet_inversion_thres

    A = wp.determinant(V)  # cubic coefficient
    D = wp.determinant(X)  # constant term
    B = float(0.0)  # quadratic coefficient
    C = float(0.0)  # linear coefficient
    
    if D <= 0:
        wp.printf("[ccd] D <= 0! D = %f\n", D)
        return 0.

    for i in range(3):
        for j in range(3):
            X_sub = wp.mat22(
                X[(i + 1) % 3, (j + 1) % 3],
                X[(i + 1) % 3, (j + 2) % 3],
                X[(i + 2) % 3, (j + 1) % 3],
                X[(i + 2) % 3, (j + 2) % 3],
            )
            V_sub = wp.mat22(
                V[(i + 1) % 3, (j + 1) % 3],
                V[(i + 1) % 3, (j + 2) % 3],
                V[(i + 2) % 3, (j + 1) % 3],
                V[(i + 2) % 3, (j + 2) % 3],
            )
            B += wp.determinant(V_sub) * X[i, j]  # v * v * x, only x is (i, j)
            C += wp.determinant(X_sub) * V[i, j]  # x * x * v, only v is (i, j)

    return cubic_ccd(A / D, B / D, C / D, 1.0 - tet_inversion_thres)


@wp.kernel
def ccd_kernel(  # One kernel for all contact types
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    particle_scene: wp.array(dtype=wp.int32),
    n_static_blocks: int,
    contact_type: wp.array(dtype=wp.int32, ndim=2),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_block_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    # ground_altitude: float,
    plane_normals: wp.array(dtype=wp.vec3),
    plane_offsets: wp.array(dtype=float),
    slackness: float,
    thickness: float,
    max_iters: int,
    tet_inversion_thres: float,
    ee_classify_thres: float,
    ccd_step: wp.array(dtype=float),
):
    block_id = wp.tid()

    i0 = b_indices[block_id, 0]
    i1 = b_indices[block_id, 1]
    i2 = b_indices[block_id, 2]
    i3 = b_indices[block_id, 3]

    scene_id = particle_scene[i0]

    type1 = contact_type[block_id, 0]
    type2 = contact_type[block_id, 1]

    if type1 == FEM_BLOCK:  # prevent inversion
        # No need to call abd_apply
        
        if type2 == FEM_TET_BLOCK:
            x0 = x[i0]
            x1 = x[i1]
            x2 = x[i2]
            x3 = x[i3]
            
            v0 = v[i0]
            v1 = v[i1]
            v2 = v[i2]
            v3 = v[i3]

            step = tet_ccd(x0, x1, x2, x3, v0, v1, v2, v3, tet_inversion_thres)
            wp.atomic_min(ccd_step, scene_id, step * slackness)
            
            if step * slackness < 1e-6:
                wp.printf("[ccd] tet ccd step too small: %.3e\n", step * slackness)

    if type1 == PG_CONTACT_BLOCK:
        v0 = abd_apply(v, x_rest, b_indices, affine_block_ids, particle_affine, i0)
        x0 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i0)

        # v0_n = v0[2]  # v0 along ground normal (0, 0, 1)
        # d_signed = x0[2] - ground_altitude
        plane_i = -i1 - 1
        plane_normal = plane_normals[plane_i]
        plane_offset = plane_offsets[plane_i]
        d_signed = wp.dot(plane_normal, x0) - plane_offset
        v0_n = wp.dot(plane_normal, v0)

        if d_signed * v0_n < 0:
            step = max((wp.abs(d_signed) - thickness) / wp.abs(v0_n), 0.0)
            wp.atomic_min(ccd_step, scene_id, step * slackness)

    if type1 == PT_CONTACT_BLOCK or type1 == EE_CONTACT_BLOCK:
        v0 = abd_apply(v, x_rest, b_indices, affine_block_ids, particle_affine, i0)
        v1 = abd_apply(v, x_rest, b_indices, affine_block_ids, particle_affine, i1)
        v2 = abd_apply(v, x_rest, b_indices, affine_block_ids, particle_affine, i2)
        v3 = abd_apply(v, x_rest, b_indices, affine_block_ids, particle_affine, i3)

        x0 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i0)
        x1 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i1)
        x2 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i2)
        x3 = abd_apply(x, x_rest, b_indices, affine_block_ids, particle_affine, i3)

        v_avg = (v0 + v1 + v2 + v3) / 4.0
        v0 -= v_avg
        v1 -= v_avg
        v2 -= v_avg
        v3 -= v_avg

        if type1 == PT_CONTACT_BLOCK:
            l_v = wp.length(v0) + wp.max(
                wp.length(v1), wp.max(wp.length(v2), wp.length(v3))
            )
        else:
            l_v = wp.max(wp.length(v0), wp.length(v1)) + wp.max(
                wp.length(v2), wp.length(v3)
            )
        if l_v < 1e-12:
            return

        if type1 == PT_CONTACT_BLOCK:
            type2 = pt_pair_classify(x0, x1, x2, x3)
            d = pt_pair_distance(x0, x1, x2, x3, type2)
        else:
            type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
            d = ee_pair_distance(x0, x1, x2, x3, type2)

        if d < thickness:
            wp.printf("[ccd] type1 = %d, type2 = %d, x0 = (%f, %f, %f), x1 = (%f, %f, %f), x2 = (%f, %f, %f), x3 = (%f, %f, %f), d = %f, thickness = %f\n", type1, type2, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], x3[0], x3[1], x3[2], d, thickness)

        d_sqr = d * d
        d_minus_thickness = max(
            (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
        )
        g = (1.0 - slackness) * d_minus_thickness
        t = float(0.0)
        t_l = slackness * d_minus_thickness / l_v

        i = int(0)

        while True:
            x0 += t_l * v0
            x1 += t_l * v1
            x2 += t_l * v2
            x3 += t_l * v3

            if type1 == PT_CONTACT_BLOCK:
                type2 = pt_pair_classify(x0, x1, x2, x3)
                d = pt_pair_distance(x0, x1, x2, x3, type2)
            else:
                type2 = ee_pair_classify(x0, x1, x2, x3, ee_classify_thres)
                d = ee_pair_distance(x0, x1, x2, x3, type2)

            d_sqr = d * d
            d_minus_thickness = max(
                (d_sqr - thickness * thickness) / (wp.sqrt(d_sqr) + thickness), 0.0
            )
            if i > 0 and d_minus_thickness <= g or i >= max_iters:
                wp.atomic_min(ccd_step, scene_id, t)
                return

            t = t + t_l
            if t > 1.0:
                return

            t_l = 0.8 * d_minus_thickness / l_v
            i += 1
