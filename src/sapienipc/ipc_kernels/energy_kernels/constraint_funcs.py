import warp as wp

from ..abd_utils_kernels import (
    apply_parent_affine_func as abd_apply,
    accumulate_on_proxies_masked_func as abd_accum_m
)


@wp.func
def angular_derivative_control_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_prev_step: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    kd: float,
    omega_target: float,
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
    x01 = x0[0]
    x02 = x0[1]
    x03 = x0[2]

    x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
    x11 = x1[0]
    x12 = x1[1]
    x13 = x1[2]

    x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
    x21 = x2[0]
    x22 = x2[1]
    x23 = x2[2]

    x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)
    x31 = x3[0]
    x32 = x3[1]
    x33 = x3[2]

    x0_prev = abd_apply(x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i0)
    x0_prev1 = x0_prev[0]
    x0_prev2 = x0_prev[1]
    x0_prev3 = x0_prev[2]

    x1_prev = abd_apply(x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i1)

    x2_prev = abd_apply(x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i2)
    x2_prev1 = x2_prev[0]
    x2_prev2 = x2_prev[1]
    x2_prev3 = x2_prev[2]

    x3_prev = abd_apply(x_prev_step, x_rest, b_indices, affine_b_ids, particle_affine, i3)
    x3_prev1 = x3_prev[0]
    x3_prev2 = x3_prev[1]
    x3_prev3 = x3_prev[2]

    x01_prev_norm = wp.length(x1_prev - x0_prev)
    x02_prev_norm_sq = wp.length_sq(x2_prev - x0_prev)
    x03_prev_norm_sq = wp.length_sq(x3_prev - x0_prev)

    t2 = 1.0/h
    t3 = -x11
    t4 = -x12
    t5 = -x13
    t6 = -x21
    t7 = -x22
    t8 = -x23
    t9 = -x31
    t10 = -x32
    t11 = -x33
    t12 = -x0_prev1
    t13 = -x0_prev2
    t14 = -x0_prev3
    t15 = 1.0/x01_prev_norm
    t16 = 1.0/x02_prev_norm_sq
    t17 = 1.0/x03_prev_norm_sq
    t18 = t3+x01
    t19 = t4+x02
    t20 = t5+x03
    t21 = t6+x01
    t22 = t7+x02
    t23 = t8+x03
    t24 = t9+x01
    t25 = t6+x11
    t26 = t10+x02
    t27 = t7+x12
    t28 = t11+x03
    t29 = t8+x13
    t30 = t9+x11
    t31 = t10+x12
    t32 = t11+x13
    t33 = t12+t21+x2_prev1
    t34 = t13+t22+x2_prev2
    t35 = t14+t23+x2_prev3
    t36 = t12+t24+x3_prev1
    t37 = t13+t26+x3_prev2
    t38 = t14+t28+x3_prev3
    t39 = t18*t22
    t40 = t19*t21
    t41 = t18*t23
    t42 = t20*t21
    t43 = t19*t23
    t44 = t20*t22
    t45 = t18*t26
    t46 = t19*t24
    t47 = t18*t28
    t48 = t20*t24
    t49 = t19*t28
    t50 = t20*t26
    t51 = -t40
    t52 = -t42
    t53 = -t44
    t54 = -t46
    t55 = -t48
    t56 = -t50
    t57 = t39+t51
    t58 = t41+t52
    t59 = t43+t53
    t60 = t45+t54
    t61 = t47+t55
    t62 = t49+t56
    t63 = t35*t57
    t64 = t34*t58
    t65 = t33*t59
    t66 = t38*t60
    t67 = t37*t61
    t68 = t36*t62
    t69 = -t64
    t70 = -t67
    t71 = t63+t65+t69
    t72 = t66+t68+t70
    t73 = t2*t15*t16*t71
    t74 = t2*t15*t17*t72
    t75 = -t74
    t76 = t73+t75+omega_target

    upd0 = wp.vec3(kd*t76*(t2*t15*t16*(t59+t27*t35-t29*t34)-t2*t15*t17*(t62+t31*t38-t32*t37)),
                -kd*t76*(t2*t15*t16*(t58+t25*t35-t29*t33)-t2*t15*t17*(t61+t30*t38-t32*t36)),
                kd*t76*(t2*t15*t16*(t57+t25*t34-t27*t33)-t2*t15*t17*(t60+t30*t37-t31*t36))) * coeff
    upd1 = wp.vec3(-kd*t76*(t2*t15*t16*(t22*t35-t23*t34)-t2*t15*t17*(t26*t38-t28*t37)),
                kd*t76*(t2*t15*t16*(t21*t35-t23*t33)-t2*t15*t17*(t24*t38-t28*t36)),
                -kd*t76*(t2*t15*t16*(t21*t34-t22*t33)-t2*t15*t17*(t24*t37-t26*t36))) * coeff
    upd2 = wp.vec3(-kd*t2*t15*t16*t76*(t59-t19*t35+t20*t34),
                kd*t2*t15*t16*t76*(t58-t18*t35+t20*t33),
                -kd*t2*t15*t16*t76*(t57-t18*t34+t19*t33)) * coeff
    upd3 = wp.vec3(kd*t2*t15*t17*t76*(t62-t19*t38+t20*t37),
                -kd*t2*t15*t17*t76*(t61-t18*t38+t20*t36),
                kd*t2*t15*t17*t76*(t60-t18*t37+t19*t36)) * coeff
    
    abd_accum_m(energy_grad, upd0, x_rest, b_indices, affine_b_ids, particle_affine, mask, i0)
    abd_accum_m(energy_grad, upd1, x_rest, b_indices, affine_b_ids, particle_affine, mask, i1)
    abd_accum_m(energy_grad, upd2, x_rest, b_indices, affine_b_ids, particle_affine, mask, i2)
    abd_accum_m(energy_grad, upd3, x_rest, b_indices, affine_b_ids, particle_affine, mask, i3)

    t2 = 1.0/h
    t4 = -x11
    t5 = -x12
    t6 = -x13
    t7 = -x21
    t8 = -x22
    t9 = -x23
    t10 = -x31
    t11 = -x32
    t12 = -x33
    t13 = -x0_prev1
    t14 = -x0_prev2
    t15 = -x0_prev3
    t16 = -x2_prev1
    t17 = -x2_prev2
    t18 = -x2_prev3
    t19 = -x3_prev1
    t20 = -x3_prev2
    t21 = -x3_prev3
    t22 = 1.0/x01_prev_norm
    t24 = 1.0/x02_prev_norm_sq
    t26 = 1.0/x03_prev_norm_sq
    t3 = t2*t2
    t23 = t22*t22
    t25 = t24*t24
    t27 = t26*t26
    t28 = t4+x01
    t29 = t5+x02
    t30 = t6+x03
    t31 = t7+x01
    t32 = t8+x02
    t33 = t9+x03
    t34 = t10+x01
    t35 = t7+x11
    t36 = t11+x02
    t37 = t8+x12
    t38 = t12+x03
    t39 = t9+x13
    t40 = t10+x11
    t41 = t11+x12
    t42 = t12+x13
    t43 = t16+x0_prev1
    t44 = t17+x0_prev2
    t45 = t18+x0_prev3
    t46 = t19+x0_prev1
    t47 = t20+x0_prev2
    t48 = t21+x0_prev3
    t49 = t13+t31+x2_prev1
    t50 = t14+t32+x2_prev2
    t51 = t15+t33+x2_prev3
    t52 = t13+t34+x3_prev1
    t53 = t14+t36+x3_prev2
    t54 = t15+t38+x3_prev3
    t55 = t28*t32
    t56 = t29*t31
    t57 = t28*t33
    t58 = t30*t31
    t59 = t29*t33
    t60 = t30*t32
    t61 = t28*t36
    t62 = t29*t34
    t63 = t28*t38
    t64 = t30*t34
    t65 = t29*t38
    t66 = t30*t36
    t115 = t2*t22*t24*t43
    t116 = t2*t22*t24*t44
    t117 = t2*t22*t24*t45
    t118 = t2*t22*t26*t46
    t119 = t2*t22*t26*t47
    t120 = t2*t22*t26*t48
    t68 = -t56
    t70 = -t58
    t72 = -t60
    t74 = -t62
    t76 = -t64
    t78 = -t66
    t79 = t29*t49
    t80 = t28*t50
    t81 = t30*t49
    t82 = t28*t51
    t83 = t30*t50
    t84 = t29*t51
    t85 = t32*t49
    t86 = t31*t50
    t87 = t33*t49
    t88 = t31*t51
    t89 = t33*t50
    t90 = t32*t51
    t91 = t29*t52
    t92 = t37*t49
    t93 = t28*t53
    t94 = t30*t52
    t95 = t35*t50
    t96 = t39*t49
    t97 = t28*t54
    t98 = t30*t53
    t99 = t35*t51
    t100 = t39*t50
    t101 = t29*t54
    t102 = t37*t51
    t103 = t36*t52
    t104 = t34*t53
    t105 = t38*t52
    t106 = t34*t54
    t107 = t38*t53
    t108 = t36*t54
    t109 = t41*t52
    t110 = t40*t53
    t111 = t42*t52
    t112 = t40*t54
    t113 = t42*t53
    t114 = t41*t54
    t139 = -t118
    t140 = -t119
    t141 = -t120
    t121 = -t80
    t122 = -t82
    t123 = -t84
    t124 = -t86
    t125 = -t88
    t126 = -t90
    t127 = -t93
    t129 = -t97
    t131 = -t101
    t133 = -t104
    t134 = -t106
    t135 = -t108
    t142 = t55+t68
    t143 = t57+t70
    t144 = t59+t72
    t145 = t61+t74
    t146 = t63+t76
    t147 = t65+t78
    t160 = t115+t139
    t161 = t116+t140
    t162 = t117+t141
    t148 = t85+t124
    t149 = t87+t125
    t150 = t89+t126
    t151 = t103+t133
    t152 = t105+t134
    t153 = t107+t135
    t154 = t51*t142
    t155 = t50*t143
    t156 = t49*t144
    t157 = t54*t145
    t158 = t53*t146
    t159 = t52*t147
    t174 = t79+t121+t142
    t175 = t81+t122+t143
    t176 = t83+t123+t144
    t179 = t91+t127+t145
    t181 = t94+t129+t146
    t182 = t98+t131+t147
    t189 = -t2*t22*t26*(-t109+t110+t145)
    t190 = -t2*t22*t26*(-t111+t112+t146)
    t191 = -t2*t22*t26*(-t113+t114+t147)
    t163 = -t155
    t164 = -t158
    t165 = t2*t22*t24*t148
    t166 = t2*t22*t24*t149
    t167 = t2*t22*t24*t150
    t168 = t2*t22*t26*t151
    t169 = t2*t22*t26*t152
    t170 = t2*t22*t26*t153
    t203 = kd*t3*t23*t25*t174*t175
    t204 = kd*t3*t23*t25*t174*t176
    t205 = kd*t3*t23*t25*t175*t176
    t206 = kd*t3*t23*t27*t179*t181
    t207 = kd*t3*t23*t27*t179*t182
    t208 = kd*t3*t23*t27*t181*t182
    t213 = kd*t3*t23*t24*t26*t174*t179
    t214 = kd*t3*t23*t24*t26*t174*t181
    t215 = kd*t3*t23*t24*t26*t175*t179
    t216 = kd*t3*t23*t24*t26*t174*t182
    t217 = kd*t3*t23*t24*t26*t175*t181
    t218 = kd*t3*t23*t24*t26*t176*t179
    t219 = kd*t3*t23*t24*t26*t175*t182
    t220 = kd*t3*t23*t24*t26*t176*t181
    t221 = kd*t3*t23*t24*t26*t176*t182
    t262 = -kd*t2*t22*t24*t174*(t189+t2*t22*t24*(-t92+t95+t142))
    t264 = -kd*t2*t22*t24*t176*(t189+t2*t22*t24*(-t92+t95+t142))
    t266 = -kd*t2*t22*t24*t175*(t190+t2*t22*t24*(-t96+t99+t143))
    t268 = -kd*t2*t22*t24*t174*(t191+t2*t22*t24*(-t100+t102+t144))
    t270 = -kd*t2*t22*t24*t176*(t191+t2*t22*t24*(-t100+t102+t144))
    t272 = -kd*t2*t22*t26*t181*(t189+t2*t22*t24*(-t92+t95+t142))
    t274 = -kd*t2*t22*t26*t179*(t190+t2*t22*t24*(-t96+t99+t143))
    t276 = -kd*t2*t22*t26*t182*(t190+t2*t22*t24*(-t96+t99+t143))
    t278 = -kd*t2*t22*t26*t181*(t191+t2*t22*t24*(-t100+t102+t144))
    t280 = kd*t2*t22*t24*t175*(t189+t2*t22*t24*(-t92+t95+t142))
    t281 = kd*t2*t22*t24*t174*(t190+t2*t22*t24*(-t96+t99+t143))
    t282 = kd*t2*t22*t24*t176*(t190+t2*t22*t24*(-t96+t99+t143))
    t283 = kd*t2*t22*t24*t175*(t191+t2*t22*t24*(-t100+t102+t144))
    t284 = kd*t2*t22*t26*t179*(t189+t2*t22*t24*(-t92+t95+t142))
    t285 = kd*t2*t22*t26*t182*(t189+t2*t22*t24*(-t92+t95+t142))
    t286 = kd*t2*t22*t26*t181*(t190+t2*t22*t24*(-t96+t99+t143))
    t287 = kd*t2*t22*t26*t179*(t191+t2*t22*t24*(-t100+t102+t144))
    t288 = kd*t2*t22*t26*t182*(t191+t2*t22*t24*(-t100+t102+t144))
    t322 = kd*(t189+t2*t22*t24*(-t92+t95+t142))*(t190+t2*t22*t24*(-t96+t99+t143))
    t323 = kd*(t189+t2*t22*t24*(-t92+t95+t142))*(t191+t2*t22*t24*(-t100+t102+t144))
    t324 = kd*(t190+t2*t22*t24*(-t96+t99+t143))*(t191+t2*t22*t24*(-t100+t102+t144))
    t171 = -t168
    t172 = -t169
    t173 = -t170
    t198 = t154+t156+t163
    t199 = t157+t159+t164
    t209 = -t203
    t210 = -t205
    t211 = -t206
    t212 = -t208
    t222 = -t213
    t223 = -t216
    t224 = -t217
    t225 = -t218
    t226 = -t221
    t325 = -t322
    t326 = -t324
    t195 = t165+t171
    t196 = t166+t172
    t197 = t167+t173
    t200 = t2*t22*t24*t198
    t201 = t2*t22*t26*t199
    t202 = -t201
    t230 = kd*t2*t22*t24*t174*t195
    t231 = kd*t2*t22*t24*t175*t195
    t232 = kd*t2*t22*t24*t174*t196
    t233 = kd*t2*t22*t24*t176*t195
    t234 = kd*t2*t22*t24*t175*t196
    t235 = kd*t2*t22*t24*t174*t197
    t236 = kd*t2*t22*t24*t176*t196
    t237 = kd*t2*t22*t24*t175*t197
    t238 = kd*t2*t22*t24*t176*t197
    t239 = kd*t2*t22*t26*t179*t195
    t240 = kd*t2*t22*t26*t181*t195
    t241 = kd*t2*t22*t26*t179*t196
    t242 = kd*t2*t22*t26*t182*t195
    t243 = kd*t2*t22*t26*t181*t196
    t244 = kd*t2*t22*t26*t179*t197
    t245 = kd*t2*t22*t26*t182*t196
    t246 = kd*t2*t22*t26*t181*t197
    t247 = kd*t2*t22*t26*t182*t197
    t257 = kd*t195*t196
    t258 = kd*t195*t197
    t259 = kd*t196*t197
    t290 = -kd*t196*(t189+t2*t22*t24*(-t92+t95+t142))
    t291 = -kd*t195*(t190+t2*t22*t24*(-t96+t99+t143))
    t295 = -kd*t197*(t190+t2*t22*t24*(-t96+t99+t143))
    t296 = -kd*t196*(t191+t2*t22*t24*(-t100+t102+t144))
    t298 = kd*t195*(t189+t2*t22*t24*(-t92+t95+t142))
    t299 = kd*t197*(t189+t2*t22*t24*(-t92+t95+t142))
    t300 = kd*t196*(t190+t2*t22*t24*(-t96+t99+t143))
    t301 = kd*t195*(t191+t2*t22*t24*(-t100+t102+t144))
    t302 = kd*t197*(t191+t2*t22*t24*(-t100+t102+t144))
    t248 = -t230
    t249 = -t233
    t250 = -t234
    t251 = -t235
    t252 = -t238
    t253 = -t240
    t254 = -t241
    t255 = -t245
    t256 = -t246
    t260 = -t257
    t261 = -t259
    t303 = t200+t202+omega_target
    t304 = kd*t115*t303
    t305 = kd*t116*t303
    t306 = kd*t117*t303
    t307 = kd*t118*t303
    t308 = kd*t119*t303
    t309 = kd*t120*t303
    t313 = kd*t139*t303
    t314 = kd*t140*t303
    t315 = kd*t141*t303
    t316 = kd*t160*t303
    t317 = kd*t161*t303
    t318 = kd*t162*t303
    t310 = -t304
    t311 = -t305
    t312 = -t306
    t319 = -t316
    t320 = -t317
    t321 = -t318
    t327 = t231+t304
    t328 = t236+t306
    t329 = t242+t308
    t331 = t251+t305
    t333 = t254+t307
    t334 = t244+t314
    t335 = t256+t309
    t337 = t253+t313
    t338 = t255+t315
    t339 = t264+t305
    t340 = t272+t307
    t341 = t276+t309
    t342 = t281+t304
    t344 = t283+t306
    t345 = t274+t313
    t346 = t287+t308
    t347 = t278+t315
    t350 = t285+t314
    t351 = t290+t316
    t352 = t295+t318
    t354 = t301+t317
    t330 = t232+t310
    t332 = t237+t312
    t336 = t249+t311
    t343 = t268+t311
    t348 = t280+t310
    t349 = t282+t312
    t353 = t291+t319
    t355 = t296+t321
    t356 = t299+t320
  
    energy_hessian[0][0] = wp.mat33(kd*wp.pow(t191+t2*t22*t24*(-t100+t102+t144),2.0), t326, t323, t326, kd*wp.pow(t190+t2*t22*t24*(-t96+t99+t143),2.0), t325, t323, t325, kd*wp.pow(t189+t2*t22*t24*(-t92+t95+t142),2.0))
    energy_hessian[0][1] = wp.mat33(t302, t355, t354, t352, t300, t353, t356, t351, t298)
    energy_hessian[0][2] = wp.mat33(t270, t344, t343, t349, t266, t342, t339, t348, t262)
    energy_hessian[0][3] = wp.mat33(t288, t347, t346, t341, t286, t345, t350, t340, t284)
    energy_hessian[1][0] = wp.mat33(t302, t352, t356, t355, t300, t351, t354, t353, t298)
    energy_hessian[1][1] = wp.mat33(kd*(t197*t197), t261, t258, t261, kd*(t196*t196), t260, t258, t260, kd*(t195*t195))
    energy_hessian[1][2] = wp.mat33(t252, t332, t331, t328, t250, t330, t336, t327, t248)
    energy_hessian[1][3] = wp.mat33(t247, t335, t334, t338, t243, t333, t329, t337, t239)
    energy_hessian[2][0] = wp.mat33(t270, t349, t339, t344, t266, t348, t343, t342, t262)
    energy_hessian[2][1] = wp.mat33(t252, t328, t336, t332, t250, t327, t331, t330, t248)
    energy_hessian[2][2] = wp.mat33(kd*t3*t23*t25*(t176*t176), t210, t204, t210, kd*t3*t23*t25*(t175*t175), t209, t204, t209, kd*t3*t23*t25*(t174*t174))
    energy_hessian[2][3] = wp.mat33(t226, t220, t225, t219, t224, t215, t223, t214, t222)
    energy_hessian[3][0] = wp.mat33(t288, t341, t350, t347, t286, t340, t346, t345, t284)
    energy_hessian[3][1] = wp.mat33(t247, t338, t329, t335, t243, t337, t334, t333, t239)
    energy_hessian[3][2] = wp.mat33(t226, t219, t223, t220, t224, t214, t225, t215, t222)
    energy_hessian[3][3] = wp.mat33(kd*t3*t23*t27*(t182*t182), t212, t207, t212, kd*t3*t23*t27*(t181*t181), t211, t207, t211, kd*t3*t23*t27*(t179*t179))


@wp.func
def _symmetric(A: wp.mat33):
    return A + wp.transpose(A)


@wp.func
def angular_proportional_control_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    kp: float,
    theta_target: float,
    h: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    # Convert to the same numbering as bending energy:
    #   x0 = x[i2], x1 = x[i1], x2 = x[i0], x3 = x[i3]

    x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
    x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
    x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
    x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)

    e0 = x2 - x1
    e1 = x0 - x2
    e2 = x0 - x1
    e1_ = x3 - x2
    e2_ = x3 - x1

    e0_lensq = wp.length_sq(e0)
    e1_lensq = wp.length_sq(e1)
    e2_lensq = wp.length_sq(e2)
    e1_lensq_ = wp.length_sq(e1_)
    e2_lensq_ = wp.length_sq(e2_)

    e0_len = wp.sqrt(e0_lensq)
    e1_len = wp.sqrt(e1_lensq)
    e2_len = wp.sqrt(e2_lensq)
    e1_len_ = wp.sqrt(e1_lensq_)
    e2_len_ = wp.sqrt(e2_lensq_)
    
    e0_unit = wp.normalize(e0)
    e1_unit = wp.normalize(e1)
    e2_unit = wp.normalize(e2)
    e1_unit_ = wp.normalize(e1_)
    e2_unit_ = wp.normalize(e2_)

    cos_alpha1 = wp.dot(e2_unit, e0_unit)
    cos_alpha2 = -wp.dot(e1_unit, e0_unit)
    cos_alpha1_ = wp.dot(e2_unit_, e0_unit)
    cos_alpha2_ = -wp.dot(e1_unit_, e0_unit)
    
    h0 = wp.length(wp.cross(e1, e2)) / e0_len
    h1 = wp.length(wp.cross(e0, e2)) / e1_len
    h2 = wp.length(wp.cross(e0, e1)) / e2_len
    h0_ = wp.length(wp.cross(e1_, e2_)) / e0_len
    h1_ = wp.length(wp.cross(e0, e2_)) / e1_len_
    h2_ = wp.length(wp.cross(e0, e1_)) / e2_len_

    n = wp.normalize(wp.cross(e0, e2))
    n_ = wp.normalize(wp.cross(e2_, e0))

    cos_theta = wp.dot(n, n_)
    sin_theta = wp.dot(wp.cross(n, n_), e0) / e0_len
    theta = wp.atan2(sin_theta, cos_theta)  # range (-pi, pi)
    
    delta_theta = theta - (wp.pi - theta_target)
    # project delta_theta to [-pi, pi]
    while delta_theta > wp.pi:
        delta_theta -= 2.0 * wp.pi
    while delta_theta < -wp.pi:
        delta_theta += 2.0 * wp.pi
    dE_dtheta = kp * delta_theta
    d2E_dtheta2 = kp

    dtheta_dx1 = n * (cos_alpha2 / h1) + n_ * (cos_alpha2_ / h1_)
    dtheta_dx2 = n * (cos_alpha1 / h2) + n_ * (cos_alpha1_ / h2_)
    dtheta_dx0 = n * (-1.0 / h0)
    dtheta_dx3 = n_ * (-1.0 / h0_)

    abd_accum_m(energy_grad, dE_dtheta * dtheta_dx0 * coeff, x_rest, b_indices, affine_b_ids, particle_affine, mask, i2)
    abd_accum_m(energy_grad, dE_dtheta * dtheta_dx1 * coeff, x_rest, b_indices, affine_b_ids, particle_affine, mask, i1)
    abd_accum_m(energy_grad, dE_dtheta * dtheta_dx2 * coeff, x_rest, b_indices, affine_b_ids, particle_affine, mask, i0)
    abd_accum_m(energy_grad, dE_dtheta * dtheta_dx3 * coeff, x_rest, b_indices, affine_b_ids, particle_affine, mask, i3)

    m0 = wp.normalize(wp.cross(e0, n))
    m1 = wp.normalize(wp.cross(e1, n))
    m2 = wp.normalize(wp.cross(n, e2))
    m0_ = wp.normalize(wp.cross(n_, e0))
    m1_ = wp.normalize(wp.cross(n_, e1_))
    m2_ = wp.normalize(wp.cross(e2_, n_))

    M0 = wp.outer(n, m0)
    M1 = wp.outer(n, m1)
    M2 = wp.outer(n, m2)
    M0_ = wp.outer(n_, m0_)
    M1_ = wp.outer(n_, m1_)
    M2_ = wp.outer(n_, m2_)

    N0 = M0 / e0_lensq
    N0_ = M0_ / e0_lensq

    P10 = cos_alpha1 * wp.transpose(M0) / (h1 * h0)
    P11 = cos_alpha1 * wp.transpose(M1) / (h1 * h1)
    P12 = cos_alpha1 * wp.transpose(M2) / (h1 * h2)
    P20 = cos_alpha2 * wp.transpose(M0) / (h2 * h0)
    P21 = cos_alpha2 * wp.transpose(M1) / (h2 * h1)
    P22 = cos_alpha2 * wp.transpose(M2) / (h2 * h2)
    P10_ = cos_alpha1_ * wp.transpose(M0_) / (h1_ * h0_)
    P11_ = cos_alpha1_ * wp.transpose(M1_) / (h1_ * h1_)
    P12_ = cos_alpha1_ * wp.transpose(M2_) / (h1_ * h2_)
    P20_ = cos_alpha2_ * wp.transpose(M0_) / (h2_ * h0_)
    P21_ = cos_alpha2_ * wp.transpose(M1_) / (h2_ * h1_)
    P22_ = cos_alpha2_ * wp.transpose(M2_) / (h2_ * h2_)

    Q0 = M0 / (h0 * h0)
    Q1 = M1 / (h0 * h1)
    Q2 = M2 / (h0 * h2)
    Q0_ = M0_ / (h0_ * h0_)
    Q1_ = M1_ / (h0_ * h1_)
    Q2_ = M2_ / (h0_ * h2_)

    energy_hessian[2, 2] = -_symmetric(Q0) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx0, dtheta_dx0)
    energy_hessian[3, 3] = -_symmetric(Q0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx3, dtheta_dx3)
    energy_hessian[1, 1] = (_symmetric(P11) - N0 + _symmetric(P11_) - N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx1)
    energy_hessian[0, 0] = (_symmetric(P22) - N0 + _symmetric(P22_) - N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx2)
    energy_hessian[1, 2] = (P10 - Q1) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx0)
    energy_hessian[0, 2] = (P20 - Q2) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx0)
    energy_hessian[1, 3] = (P10_ - Q1_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx3)
    energy_hessian[0, 3] = (P20_ - Q2_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx2, dtheta_dx3)
    energy_hessian[1, 0] = (P12 + wp.transpose(P21) + N0 + P12_ + wp.transpose(P21_) + N0_) * dE_dtheta + d2E_dtheta2 * wp.outer(dtheta_dx1, dtheta_dx2)
    energy_hessian[2, 3] = d2E_dtheta2 * wp.outer(dtheta_dx0, dtheta_dx3)

    energy_hessian[2, 1] = wp.transpose(energy_hessian[1, 2])
    energy_hessian[2, 0] = wp.transpose(energy_hessian[0, 2])
    energy_hessian[3, 1] = wp.transpose(energy_hessian[1, 3])
    energy_hessian[3, 0] = wp.transpose(energy_hessian[0, 3])
    energy_hessian[0, 1] = wp.transpose(energy_hessian[1, 0])
    energy_hessian[3, 2] = wp.transpose(energy_hessian[2, 3])


@wp.func
def area_constrant_diff_func(
    x: wp.array(dtype=wp.vec3),
    x_rest: wp.array(dtype=wp.vec3),
    b_indices: wp.array(dtype=wp.int32, ndim=2),
    affine_b_ids: wp.array(dtype=wp.int32),
    particle_affine: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.float32),
    i0: int,
    i1: int,
    i2: int,
    i3: int,
    area_target: float,
    mu: float,
    la: float,
    coeff: float,
    energy_grad: wp.array(dtype=wp.vec3),
    energy_hessian: wp.array(dtype=wp.mat33, ndim=2),
):
    x0 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i0)
    x01 = x0[0]
    x02 = x0[1]
    x03 = x0[2]

    x1 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i1)
    x11 = x1[0]
    x12 = x1[1]
    x13 = x1[2]

    x2 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i2)
    x21 = x2[0]
    x22 = x2[1]
    x23 = x2[2]

    x3 = abd_apply(x, x_rest, b_indices, affine_b_ids, particle_affine, i3)
    x31 = x3[0]
    x32 = x3[1]
    x33 = x3[2]

    t2 = -x11
    t3 = -x12
    t4 = -x13
    t5 = -x31
    t6 = -x32
    t7 = -x33
    t8 = t2+x01
    t9 = t3+x02
    t10 = t4+x03
    t11 = t5+x21
    t12 = t6+x22
    t13 = t7+x23
    t14 = t8*t12
    t15 = t9*t11
    t16 = t8*t13
    t17 = t10*t11
    t18 = t9*t13
    t19 = t10*t12
    t20 = -t15
    t21 = -t17
    t22 = -t19
    t23 = t14+t20
    t24 = t16+t21
    t25 = t18+t22
    t26 = t23*t23
    t27 = t24*t24
    t28 = t25*t25
    t29 = t8*t23*2.0
    t30 = t9*t23*2.0
    t31 = t8*t24*2.0
    t32 = t10*t24*2.0
    t33 = t9*t25*2.0
    t34 = t10*t25*2.0
    t35 = t11*t23*2.0
    t36 = t12*t23*2.0
    t37 = t11*t24*2.0
    t38 = t13*t24*2.0
    t39 = t12*t25*2.0
    t40 = t13*t25*2.0
    t41 = -t34
    t42 = -t40
    t43 = t30+t32
    t44 = t31+t33
    t45 = t36+t38
    t46 = t37+t39
    t49 = t26+t27+t28 + 1e-9
    t47 = t29+t41
    t48 = t35+t42
    # wp.printf("t49 = %f\n", t49)
    t50 = wp.sqrt(t49)
    t51 = 1.0/t50
    t52 = -t50
    t53 = area_target+t52
    t54 = (la*t43*t51)/2.0
    t55 = (la*t44*t51)/2.0
    t56 = (la*t45*t51)/2.0
    t57 = (la*t46*t51)/2.0
    t58 = (la*t47*t51)/2.0
    t59 = (la*t48*t51)/2.0
    t60 = (mu*t43*t51*t53)/2.0
    t61 = (mu*t44*t51*t53)/2.0
    t62 = (mu*t45*t51*t53)/2.0
    t63 = (mu*t46*t51*t53)/2.0
    t64 = (mu*t47*t51*t53)/2.0
    t65 = (mu*t48*t51*t53)/2.0

    grad0 = wp.vec3(-t56-t62, t59+t65, t57+t63) * coeff
    grad1 = wp.vec3(t56+t62, -t59-t65, -t57-t63) * coeff
    grad2 = wp.vec3(t54+t60, -t58-t64, -t55-t61) * coeff
    grad3 = wp.vec3(-t54-t60, t58+t64, t55+t61) * coeff
    
    abd_accum_m(energy_grad, grad0, x_rest, b_indices, affine_b_ids, particle_affine, mask, i0)
    abd_accum_m(energy_grad, grad1, x_rest, b_indices, affine_b_ids, particle_affine, mask, i1)
    abd_accum_m(energy_grad, grad2, x_rest, b_indices, affine_b_ids, particle_affine, mask, i2)
    abd_accum_m(energy_grad, grad3, x_rest, b_indices, affine_b_ids, particle_affine, mask, i3)

    t2 = -x11
    t3 = -x12
    t4 = -x13
    t5 = -x31
    t6 = -x32
    t7 = -x33
    t8 = t2+x01
    t9 = t3+x02
    t10 = t4+x03
    t11 = t5+x21
    t12 = t6+x22
    t13 = t7+x23
    t14 = t8*t8
    t15 = t9*t9
    t16 = t10*t10
    t17 = t11*t11
    t18 = t12*t12
    t19 = t13*t13
    t26 = t8*t12
    t27 = t9*t11
    t28 = t8*t13
    t29 = t10*t11
    t30 = t9*t13
    t31 = t10*t12
    t32 = t8*t11*2.0
    t38 = t9*t12*2.0
    t46 = t10*t13*2.0
    t20 = t14*2.0
    t21 = t15*2.0
    t22 = t16*2.0
    t23 = t17*2.0
    t24 = t18*2.0
    t25 = t19*2.0
    t33 = t26*2.0
    t34 = t27*2.0
    t35 = t26*4.0
    t36 = t28*2.0
    t37 = t27*4.0
    t39 = t29*2.0
    t40 = t28*4.0
    t41 = t30*2.0
    t42 = t29*4.0
    t43 = t31*2.0
    t44 = t30*4.0
    t45 = t31*4.0
    t47 = -t27
    t49 = -t29
    t52 = -t31
    t63 = t32+t38
    t65 = t32+t46
    t67 = t38+t46
    t50 = -t37
    t53 = -t42
    t55 = -t45
    t56 = t20+t21
    t57 = t20+t22
    t58 = t21+t22
    t59 = t23+t24
    t60 = t23+t25
    t61 = t24+t25
    t62 = t26+t47
    t64 = t28+t49
    t66 = t30+t52
    t68 = t33+t50
    t70 = t36+t53
    t72 = t41+t55
    t74 = t62*t62
    t75 = t64*t64
    t76 = t66*t66
    t77 = t8*t62*2.0
    t78 = t9*t62*2.0
    t79 = t8*t64*2.0
    t80 = t10*t64*2.0
    t81 = t9*t66*2.0
    t82 = t10*t66*2.0
    t83 = t11*t62*2.0
    t84 = t12*t62*2.0
    t85 = t11*t64*2.0
    t86 = t13*t64*2.0
    t87 = t12*t66*2.0
    t88 = t13*t66*2.0
    t89 = -t82
    t90 = -t88
    t91 = t78+t80
    t92 = t79+t81
    t93 = t84+t86
    t94 = t85+t87
    t103 = t74+t75+t76 + 1e-9
    t95 = t77+t89
    t96 = t83+t90
    t97 = t91*t91
    t98 = t92*t92
    t99 = t93*t93
    t100 = t94*t94
    t104 = 1.0/t103
    t105 = wp.sqrt(t103)
    t101 = t95*t95
    t102 = t96*t96
    t106 = 1.0/t105
    t108 = -t105
    t152 = (mu*t97*t104)/4.0
    t153 = (mu*t98*t104)/4.0
    t154 = (mu*t99*t104)/4.0
    t155 = (mu*t100*t104)/4.0
    t218 = (mu*t91*t92*t104)/4.0
    t219 = (mu*t91*t93*t104)/4.0
    t220 = (mu*t91*t94*t104)/4.0
    t221 = (mu*t92*t93*t104)/4.0
    t222 = (mu*t92*t94*t104)/4.0
    t223 = (mu*t93*t94*t104)/4.0
    t224 = (mu*t91*t95*t104)/4.0
    t226 = (mu*t92*t95*t104)/4.0
    t228 = (mu*t91*t96*t104)/4.0
    t229 = (mu*t93*t95*t104)/4.0
    t232 = (mu*t94*t95*t104)/4.0
    t233 = (mu*t92*t96*t104)/4.0
    t235 = (mu*t93*t96*t104)/4.0
    t237 = (mu*t94*t96*t104)/4.0
    t248 = (mu*t95*t96*t104)/4.0
    t107 = t106*t106*t106
    t109 = area_target+t108
    t110 = la*t8*t9*t106
    t111 = la*t8*t10*t106
    t112 = la*t9*t10*t106
    t113 = la*t11*t12*t106
    t114 = la*t11*t13*t106
    t115 = la*t12*t13*t106
    t122 = (la*t56*t106)/2.0
    t123 = (la*t57*t106)/2.0
    t124 = (la*t58*t106)/2.0
    t125 = (la*t59*t106)/2.0
    t126 = (la*t60*t106)/2.0
    t127 = (la*t61*t106)/2.0
    t134 = (la*t63*t106)/2.0
    t135 = (la*t65*t106)/2.0
    t136 = (la*t67*t106)/2.0
    t138 = (la*t68*t106)/2.0
    t139 = la*t106*(t34-t35)*(-1.0/2.0)
    t141 = (la*t70*t106)/2.0
    t142 = la*t106*(t39-t40)*(-1.0/2.0)
    t144 = (la*t72*t106)/2.0
    t145 = la*t106*(t43-t44)*(-1.0/2.0)
    t147 = (la*t106*(t34-t35))/2.0
    t149 = (la*t106*(t39-t40))/2.0
    t151 = (la*t106*(t43-t44))/2.0
    t156 = -t152
    t157 = (mu*t101*t104)/4.0
    t158 = -t153
    t159 = -t154
    t160 = (mu*t102*t104)/4.0
    t161 = -t155
    t225 = -t218
    t227 = -t219
    t230 = -t220
    t231 = -t221
    t234 = -t222
    t236 = -t223
    t244 = -t224
    t245 = -t226
    t246 = -t228
    t247 = -t229
    t249 = -t232
    t250 = -t233
    t251 = -t235
    t252 = -t237
    t267 = -t248
    t116 = -t110
    t117 = -t111
    t118 = -t112
    t119 = -t113
    t120 = -t114
    t121 = -t115
    t128 = -t122
    t129 = -t123
    t130 = -t124
    t131 = -t125
    t132 = -t126
    t133 = -t127
    t137 = -t134
    t140 = -t135
    t143 = -t136
    t146 = -t138
    t148 = -t141
    t150 = -t144
    t162 = (la*t97*t107)/4.0
    t163 = (la*t98*t107)/4.0
    t164 = (la*t99*t107)/4.0
    t165 = (la*t100*t107)/4.0
    t166 = -t157
    t167 = -t160
    t169 = (la*t101*t107)/4.0
    t172 = (la*t102*t107)/4.0
    t176 = mu*t8*t9*t106*t109
    t177 = mu*t8*t10*t106*t109
    t178 = mu*t9*t10*t106*t109
    t179 = mu*t11*t12*t106*t109
    t180 = mu*t11*t13*t106*t109
    t181 = mu*t12*t13*t106*t109
    t188 = (mu*t56*t106*t109)/2.0
    t189 = (mu*t57*t106*t109)/2.0
    t190 = (mu*t58*t106*t109)/2.0
    t191 = (mu*t59*t106*t109)/2.0
    t192 = (mu*t60*t106*t109)/2.0
    t193 = (mu*t61*t106*t109)/2.0
    t200 = (mu*t63*t106*t109)/2.0
    t201 = (mu*t65*t106*t109)/2.0
    t202 = (mu*t67*t106*t109)/2.0
    t204 = (mu*t68*t106*t109)/2.0
    t205 = mu*t106*t109*(t34-t35)*(-1.0/2.0)
    t207 = (mu*t70*t106*t109)/2.0
    t208 = mu*t106*t109*(t39-t40)*(-1.0/2.0)
    t210 = (mu*t72*t106*t109)/2.0
    t211 = mu*t106*t109*(t43-t44)*(-1.0/2.0)
    t213 = (mu*t106*t109*(t34-t35))/2.0
    t215 = (mu*t106*t109*(t39-t40))/2.0
    t217 = (mu*t106*t109*(t43-t44))/2.0
    t238 = (la*t91*t92*t107)/4.0
    t239 = (la*t91*t93*t107)/4.0
    t240 = (la*t91*t94*t107)/4.0
    t241 = (la*t92*t93*t107)/4.0
    t242 = (la*t92*t94*t107)/4.0
    t243 = (la*t93*t94*t107)/4.0
    t253 = (la*t91*t95*t107)/4.0
    t255 = (la*t92*t95*t107)/4.0
    t257 = (la*t91*t96*t107)/4.0
    t258 = (la*t93*t95*t107)/4.0
    t261 = (la*t94*t95*t107)/4.0
    t262 = (la*t92*t96*t107)/4.0
    t264 = (la*t93*t96*t107)/4.0
    t266 = (la*t94*t96*t107)/4.0
    t272 = (la*t95*t96*t107)/4.0
    t278 = (mu*t97*t107*t109)/4.0
    t279 = (mu*t98*t107*t109)/4.0
    t280 = (mu*t99*t107*t109)/4.0
    t281 = (mu*t100*t107*t109)/4.0
    t283 = (mu*t101*t107*t109)/4.0
    t286 = (mu*t102*t107*t109)/4.0
    t290 = (mu*t91*t92*t107*t109)/4.0
    t291 = (mu*t91*t93*t107*t109)/4.0
    t292 = (mu*t91*t94*t107*t109)/4.0
    t293 = (mu*t92*t93*t107*t109)/4.0
    t294 = (mu*t92*t94*t107*t109)/4.0
    t295 = (mu*t93*t94*t107*t109)/4.0
    t296 = (mu*t91*t95*t107*t109)/4.0
    t298 = (mu*t92*t95*t107*t109)/4.0
    t300 = (mu*t91*t96*t107*t109)/4.0
    t301 = (mu*t93*t95*t107*t109)/4.0
    t304 = (mu*t94*t95*t107*t109)/4.0
    t305 = (mu*t92*t96*t107*t109)/4.0
    t307 = (mu*t93*t96*t107*t109)/4.0
    t309 = (mu*t94*t96*t107*t109)/4.0
    t314 = (mu*t95*t96*t107*t109)/4.0
    t168 = -t162
    t170 = -t163
    t171 = -t164
    t173 = -t165
    t174 = -t169
    t175 = -t172
    t182 = -t176
    t183 = -t177
    t184 = -t178
    t185 = -t179
    t186 = -t180
    t187 = -t181
    t194 = -t188
    t195 = -t189
    t196 = -t190
    t197 = -t191
    t198 = -t192
    t199 = -t193
    t203 = -t200
    t206 = -t201
    t209 = -t202
    t212 = -t204
    t214 = -t207
    t216 = -t210
    t254 = -t238
    t256 = -t239
    t259 = -t240
    t260 = -t241
    t263 = -t242
    t265 = -t243
    t268 = -t253
    t269 = -t255
    t270 = -t257
    t271 = -t258
    t273 = -t261
    t274 = -t262
    t275 = -t264
    t276 = -t266
    t277 = -t272
    t282 = -t278
    t284 = -t279
    t285 = -t280
    t287 = -t281
    t288 = -t283
    t289 = -t286
    t297 = -t290
    t299 = -t291
    t302 = -t292
    t303 = -t293
    t306 = -t294
    t308 = -t295
    t310 = -t296
    t311 = -t298
    t312 = -t300
    t313 = -t301
    t315 = -t304
    t316 = -t305
    t317 = -t307
    t318 = -t309
    t319 = -t314
    t333 = t112+t178+t226+t255+t298
    t335 = t115+t181+t237+t266+t309
    t346 = t141+t207+t220+t240+t292
    t349 = t149+t215+t221+t241+t293
    t350 = t138+t204+t228+t257+t300
    t352 = t145+t211+t233+t262+t305
    t353 = t147+t213+t229+t258+t301
    t320 = t130+t152+t162+t196+t278
    t321 = t128+t153+t163+t194+t279
    t322 = t133+t154+t164+t199+t280
    t323 = t131+t155+t165+t197+t281
    t324 = t124+t156+t168+t190+t282
    t325 = t122+t158+t170+t188+t284
    t326 = t127+t159+t171+t193+t285
    t327 = t125+t161+t173+t191+t287
    t328 = t129+t157+t169+t195+t283
    t329 = t132+t160+t172+t198+t286
    t330 = t123+t166+t174+t189+t288
    t331 = t126+t167+t175+t192+t289
    t332 = t111+t177+t225+t254+t297
    t334 = t114+t180+t236+t265+t308
    t336 = t117+t183+t218+t238+t290
    t337 = t120+t186+t223+t243+t295
    t338 = t110+t176+t244+t268+t310
    t339 = t113+t179+t251+t275+t317
    t340 = t116+t182+t224+t253+t296
    t341 = t119+t185+t235+t264+t307
    t342 = t118+t184+t245+t269+t311
    t343 = t121+t187+t252+t276+t318
    t344 = t143+t209+t219+t239+t291
    t345 = t137+t203+t222+t242+t294
    t347 = t136+t202+t227+t256+t299
    t348 = t134+t200+t234+t263+t306
    t351 = t142+t208+t231+t260+t303
    t354 = t148+t214+t230+t259+t302
    t355 = t150+t216+t232+t261+t304
    t356 = t139+t205+t247+t271+t313
    t357 = t140+t206+t248+t272+t314
    t358 = t144+t210+t249+t273+t315
    t359 = t135+t201+t267+t277+t319
    t360 = t146+t212+t246+t270+t312
    t361 = t151+t217+t250+t274+t316
    
    energy_hessian[0][0] = wp.mat33(t322, t339, t334, t339, t329, t335, t334, t335, t323)
    energy_hessian[0][1] = wp.mat33(t326, t341, t337, t341, t331, t343, t337, t343, t327)
    energy_hessian[0][2] = wp.mat33(t347, t353, t349, t350, t359, t361, t346, t358, t348)
    energy_hessian[0][3] = wp.mat33(t344, t356, t351, t360, t357, t352, t354, t355, t345)
    energy_hessian[1][0] = wp.mat33(t326, t341, t337, t341, t331, t343, t337, t343, t327)
    energy_hessian[1][1] = wp.mat33(t322, t339, t334, t339, t329, t335, t334, t335, t323)
    energy_hessian[1][2] = wp.mat33(t344, t356, t351, t360, t357, t352, t354, t355, t345)
    energy_hessian[1][3] = wp.mat33(t347, t353, t349, t350, t359, t361, t346, t358, t348)
    energy_hessian[2][0] = wp.mat33(t347, t350, t346, t353, t359, t358, t349, t361, t348)
    energy_hessian[2][1] = wp.mat33(t344, t360, t354, t356, t357, t355, t351, t352, t345)
    energy_hessian[2][2] = wp.mat33(t320, t338, t332, t338, t328, t333, t332, t333, t321)
    energy_hessian[2][3] = wp.mat33(t324, t340, t336, t340, t330, t342, t336, t342, t325)
    energy_hessian[3][0] = wp.mat33(t344, t360, t354, t356, t357, t355, t351, t352, t345)
    energy_hessian[3][1] = wp.mat33(t347, t350, t346, t353, t359, t358, t349, t361, t348)
    energy_hessian[3][2] = wp.mat33(t324, t340, t336, t340, t330, t342, t336, t342, t325)
    energy_hessian[3][3] = wp.mat33(t320, t338, t332, t338, t328, t333, t332, t333, t321)
