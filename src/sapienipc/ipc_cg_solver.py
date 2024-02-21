"""
    References:
    - "IPC paper": Li, Minchen, Zachary Ferguson, Teseo Schneider, Timothy R. Langlois, Denis Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. "Incremental potential contact: intersection-and inversion-free, large-deformation dynamics." ACM Trans. Graph. 39, no. 4 (2020): 49.
    - "CG slides": Xu, Zhiliang. "ACMS 40212/60212: Advanced Scientific Computing, Lecture 8: Fast Linear Solvers (Part 5)." (https://www3.nd.edu/~zxu2/acms60212-40212/Lec-09-5.pdf)
    - "FEM tutorial": Sifakis, Eftychios. "FEM simulation of 3D deformable solids: a practitioner's guide to theory, discretization and model reduction. Part One: The classical FEM method and discretization methodology." In Acm siggraph 2012 courses, pp. 1-50. 2012.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from sapienipc.ipc_utils.warp_types import wp_slice

if TYPE_CHECKING:
    from .ipc_system import IPCSystem, IPCSystemConfig

import numpy as np
import warp as wp

from .ipc_kernels.cg_solver_kernels import *
from .ipc_kernels.utils_kernels import *
from .ipc_utils.logging_utils import ipc_logger


class CGSolver:
    def __init__(self, system: IPCSystem):
        self.system = system
        config = system.config

        # Copy from config
        self.max_iters = config.cg_max_iters
        self.precondition = config.precondition
        self.error_tolerance = config.cg_error_tolerance
        self.error_frequency = config.cg_error_frequency
        self.max_particles = config.max_particles
        self.max_blocks = config.max_blocks
        self.device = config.device  # ["cpu", "cuda:0", ...]
        self.debug = config.debug

        self.use_graph = config.use_graph
        self.graph = None
        self.graph_n_particles = None

        # self.skipping_threshold = (config.eps_d * config.time_step) ** 2

        # CG solver (Page 16, CG slides)
        self.r = wp.zeros(self.max_particles, wp.vec3, device=self.device)
        self.v = wp.zeros(self.max_particles, wp.vec3, device=self.device)
        self.A_v = wp.zeros(self.max_particles, wp.vec3, device=self.device)
        self.v_A_v = wp.zeros(config.max_scenes, dtype=wp.float32, device=self.device)
        self.r_2 = wp.zeros(config.max_scenes, dtype=wp.float32, device=self.device)

        # CG Preconditioner
        self.A_diag = wp.zeros(self.max_particles, wp.vec3, device=self.device)
        self.z = wp.zeros(self.max_particles, wp.vec3, device=self.device)
        self.zr = wp.zeros(config.max_scenes, dtype=wp.float32, device=self.device)
        self.zr_new = wp.zeros(config.max_scenes, dtype=wp.float32, device=self.device)

        # CG best solution (in case A is not positive definite)
        self.r_2_best = wp.zeros(
            config.max_scenes, dtype=wp.float32, device=self.device
        )
        self.p_best = wp.zeros(self.max_particles, wp.vec3, device=self.device)

    def launch_iterations(self, outer_iter: int):
        break_flag = False
        anomaly_flag = False
        output_debug_info = self.debug and not self.use_graph

        s = self.system
        c = s.config

        for cg_iter in range(outer_iter, outer_iter + self.error_frequency):
            # One iteration of CG (Page 16, CG slides)

            if output_debug_info:
                assert not np.isnan(
                    self.v.numpy()
                ).any(), f"v contains NaN at cg_iter={cg_iter}"

            # Compute A * v
            # self.A_v.zero_()
            wp_slice(self.A_v, 0, s.n_particles).zero_()
            wp.launch(
                kernel=hess_block_mul_dx_kernel,
                dim=max(c.max_blocks, s.n_particles),
                inputs=[
                    s.n_particles,
                    s.n_static_blocks,
                    s.contact_counter,
                    s.particle_mass,
                    s.particle_mask,
                    s.particle_dbc_mask,
                    c.kappa_con,
                    s.block_indices,
                    s.affine_block_ids,
                    s.particle_affine,
                    s.blocks,
                    s.block_status,
                    self.v,
                    s.particle_q_rest,
                ],
                outputs=[self.A_v],
                device=self.device,
            )

            if output_debug_info:
                assert not np.isnan(self.A_v.numpy()).any()

            # Compute v^T * A * v
            wp_slice(self.v_A_v, 0, s.n_scenes).zero_()
            wp.launch(
                kernel=compute_dot_kernel,
                dim=s.n_particles,
                inputs=[self.v, self.A_v, s.particle_scene],
                outputs=[self.v_A_v],
                device=self.device,
            )

            if output_debug_info:
                # assert self.v_A_v.numpy()[0] >= 0
                if np.any(self.v_A_v.numpy()[: s.n_scenes] <= 0):
                    ipc_logger.warning(
                        f"CG solver anomaly: v_A_v={self.v_A_v.numpy()[: s.n_scenes]} <= 0"
                    )
                    break_flag = True
                    anomaly_flag = True
                    break

            # if self.v_A_v.numpy()[0] <= 0:
            #     ipc_logger.warning(
            #         f"CG solver anomaly: v_A_v={self.v_A_v.numpy()[0]:.4e} <= 0, cg_iter = {cg_iter}"
            #     )
            #     break_flag = True
            #     anomaly_flag = True

            #     # if self.debug and (cg_iter == 0 or self.r_2_best.numpy()[0] >= r_2_init):
            #     if self.debug and cg_iter == 0:
            #         # save and exit
            #         dense_matrix = self.system._get_dense_matrix_brute_force()
            #         grad_np = self.system.particle_grad.numpy()[:self.system.n_particles].reshape(-1)

            #         np.save("/home/xiaodi/Desktop/warp_IPC/examples/output/debug/mat.npy", {
            #             "hessian": dense_matrix,
            #             "grad": grad_np,
            #         })

            #         x_np = self.system.particle_q.numpy()[:self.system.n_particles].reshape(-1)
            #         v_np = self.system.particle_qd.numpy()[:self.system.n_particles].reshape(-1)
            #         x_prev_np = self.system.particle_q_prev_step.numpy()[:self.system.n_particles].reshape(-1)
            #         v_prev_np = self.system.particle_qd_prev_step.numpy()[:self.system.n_particles].reshape(-1)
            #         np.save("/home/xiaodi/Desktop/warp_IPC/examples/output/debug/state.npy", {
            #             "x": x_np,
            #             "v": v_np,
            #             "x_prev": x_prev_np,
            #             "v_prev": v_prev_np,
            #         })

            #         exit(0)

            wp.copy(self.zr, self.zr_new, count=s.n_scenes)
            wp_slice(self.zr_new, 0, s.n_scenes).zero_()
            wp_slice(self.r_2, 0, s.n_scenes).zero_()

            wp.launch(
                kernel=update_p_r_z_compute_r_2_zr_kernel,
                dim=s.n_particles,
                inputs=[
                    self.v,
                    self.A_v,
                    self.v_A_v,
                    self.zr,
                    self.A_diag,
                    s.particle_scene,
                    s.particle_p,
                    self.r,
                    self.z,
                    self.r_2,
                    self.zr_new,
                ],
                device=self.device,
            )

            if output_debug_info:
                # ipc_logger.debug(f"cg_iter={cg_iter}, r_2={self.r_2.numpy()[0]:.6f}, r_2_best={self.r_2_best.numpy()[0]:.6f}, v_A_v={self.v_A_v.numpy()[0]}")

                # if np.isnan(self.r_2.numpy()[0]) or self.r_2.numpy()[0] > 1e6:
                #     break_flag = True
                #     np.save("/home/xiaodi/Desktop/warp_IPC/examples/output/debug/mat.npy", self.system._get_dense_matrix_brute_force())
                #     break

                assert not np.any(np.isnan(self.r_2.numpy()[:s.n_scenes]))
                assert not np.any(np.isnan(self.zr_new.numpy()[:s.n_scenes]))

            wp.launch(
                kernel=update_v_p_best_kernel,
                dim=s.n_particles,
                inputs=[
                    self.z,
                    self.zr,
                    self.zr_new,
                    s.particle_p,
                    self.r_2,
                    s.particle_scene,
                    self.v,
                    self.p_best,
                    self.r_2_best,
                ],
                device=self.device,
            )

            wp.launch(
                kernel=update_r_2_best_kernel,
                dim=s.n_scenes,
                inputs=[self.r_2, self.r_2_best],
                device=self.device,
            )

        return break_flag, anomaly_flag

    def solve(self) -> int:  # return 1 if not converged, 0 if converged
        """Conjugate Gradient solver for the linear system Hx = grad, where H is the
        Hessian of the IP and grad is the gradient of the IP.
        H is a sparse matrix expressed as a sum of blocks, each of which is a 12x12
        matrix involving 4 particles, stored as a 4x4 array of 3x3 wp.mat33 matrices.

        Write the solution into self.system.particle_p.
        """

        # refactored from ../contact/optimizer.py@Optimizer::solve_linear_system_CG

        s = self.system
        c = s.config

        if s.debug:
            assert not np.isnan(s.particle_grad.numpy()[:s.n_particles]).any()
            assert not np.isnan(s.blocks.numpy()[:s.n_blocks_this_step]).any()
            # np.save("./output/debug/grad.npy", s.particle_grad.numpy()[:s.n_particles])
            # np.save("./output/debug/mat.npy", s._get_dense_matrix_brute_force())

        anomaly_flag = False

        wp_slice(s.particle_p, 0, s.n_particles).zero_()
        wp_slice(self.p_best, 0, s.n_particles).zero_()

        # r = b - A * x^{(0)} = grad - 0 = grad
        wp.copy(self.r, s.particle_grad, count=s.n_particles)

        # compute Hessian matrix diagonal
        if self.precondition == "jacobi":
            # Jacobi preconditioner (Page 16, CG slides)
            # Compute M = Diag(A), here we denote the diagonal matrix with Diag and ...
            # ... the vector with diag
            wp_slice(self.A_diag, 0, s.n_particles).zero_()
            wp.launch(
                kernel=compute_block_diag_kernel,
                dim=max(c.max_blocks, s.n_particles),
                inputs=[
                    s.n_particles,
                    s.n_static_blocks,
                    s.contact_counter,
                    s.particle_mass,
                    s.particle_mask,
                    s.particle_dbc_mask,
                    c.kappa_con,
                    s.block_indices,
                    s.affine_block_ids,
                    s.particle_affine,
                    s.blocks,
                    s.block_status,
                    s.particle_q_rest,
                ],
                outputs=[self.A_diag],
                device=self.device,
            )

            # if self.debug:
            if False:
                hessian_dense = self.system._get_dense_matrix_brute_force()
                hessian_dense_diag = np.diag(hessian_dense).reshape(-1, 3)
                A_diag_np = self.A_diag.numpy()[:n_particles]
                print(
                    "max diag difference:", np.abs(hessian_dense_diag - A_diag_np).max()
                )
                max_diff_idx = np.argmax(np.abs(hessian_dense_diag - A_diag_np))
                print("max diff idx:", max_diff_idx)
                print(
                    "hessian_dense_diag[max_diff_idx]:",
                    hessian_dense_diag[max_diff_idx // 3, max_diff_idx % 3],
                )
                print(
                    "A_diag_np[max_diff_idx]:",
                    A_diag_np[max_diff_idx // 3, max_diff_idx % 3],
                )

                # print("self.A_diag[:6]")
                # print(self.A_diag.numpy()[:n_particles][:6])
                # print("hessian_dense_diag[:6]")
                # print(hessian_dense_diag[:6])

                assert np.allclose(hessian_dense_diag, A_diag_np)

            # Compute M^{-1} = Diag(A)^{-1}, and z = M^{-1} * r = r / diag(A)
            wp.launch(
                kernel=compute_block_diag_inv_kernel,
                dim=s.n_particles,
                inputs=[
                    s.particle_mask,
                    s.particle_affine,
                    self.A_diag,
                    s.particle_grad,  # grad = r
                    self.z,
                ],
                device=self.device,
            )
        elif self.precondition == "none":
            # No preconditioner, M = I
            wp_slice(self.A_diag, 0, s.n_particles).fill_(wp.vec3(1.0, 1.0, 1.0))
            # z = r
            wp.copy(self.z, self.r, count=s.n_particles)

        wp.copy(self.v, self.z, count=s.n_particles)

        # ipc_logger.info(f"self.v=\n{self.v.numpy()[:n_particles]}")

        wp_slice(self.r_2, 0, s.n_scenes).zero_()
        wp.launch(
            kernel=compute_dot_kernel,
            dim=s.n_particles,
            inputs=[self.r, self.r, s.particle_scene],
            outputs=[self.r_2],
            device=self.device,
        )
        wp.copy(self.r_2_best, self.r_2)
        r_2_init = self.r_2.numpy()[: s.n_scenes]

        wp_slice(self.zr_new, 0, s.n_scenes).zero_()
        wp.launch(
            kernel=compute_dot_kernel,
            dim=s.n_particles,
            inputs=[self.z, self.r, s.particle_scene],
            outputs=[self.zr_new],
            device=self.device,
        )
        zr_init = self.zr_new.numpy()[: s.n_scenes]

        if self.debug:
            ipc_logger.debug(f"r_2_init={r_2_init}, zr_init={zr_init}")

        if self.debug:
            assert not np.any(np.isnan(r_2_init))

        # if r_2_init < self.skipping_threshold:
        #     ipc_logger.debug(
        #         f"r_2_init={r_2_init:.4e} is too small, skipping CG solver. zr_init={zr_init:.4e}."
        #     )
        #     return

        outer_iter = 0
        cg_iter = 0
        break_flag = False

        if self.use_graph and (
            self.graph is None or self.graph_n_particles != s.n_particles
        ):
            wp.capture_begin(device=self.device)
            self.launch_iterations(0)
            self.graph = wp.capture_end(device=self.device)
            self.graph_n_particles = s.n_particles

        # ipc_logger.debug("blocks" + str(blocks.numpy()[:n_static_blocks]))

        for outer_iter in range(0, self.max_iters, self.error_frequency):
            if break_flag:
                break

            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                break_flag, anomaly_flag = self.launch_iterations(outer_iter)
            cg_iter = outer_iter + self.error_frequency

            # print(f"cg_iter={cg_iter}, last_zr={self.zr_new.numpy()[: s.n_scenes]}")

            if np.all(self.r_2.numpy()[: s.n_scenes] <= r_2_init * self.error_tolerance**2):
                if self.debug:
                    ipc_logger.debug(
                        f"CG solver converged after {cg_iter + 1} iterations, "
                        + f"last_error={np.sqrt(self.r_2.numpy()[: s.n_scenes] / r_2_init)}, "
                        + f"last_zr={np.sqrt(self.zr_new.numpy()[: s.n_scenes] / zr_init)}, "
                        + f"min_error={np.sqrt(self.r_2_best.numpy()[: s.n_scenes] / r_2_init)}"
                    )
                break_flag = True
                break

        # if not self.debug:
        #     break_flag = True

        if not break_flag or anomaly_flag:
            anomaly_flag = True
            if self.debug:
                ipc_logger.debug(
                    f"CG solver did not converge after {cg_iter + 1} iterations, "
                    + f"last_error={np.sqrt(self.r_2.numpy()[: s.n_scenes] / r_2_init)}, "
                    + f"last_zr={np.sqrt(self.zr_new.numpy()[: s.n_scenes] / zr_init)}, "
                    + f"min_error={np.sqrt(self.r_2_best.numpy()[: s.n_scenes] / r_2_init)}"
                )

        # wp.copy(p, self.p_best, count=n_particles)

        return anomaly_flag
