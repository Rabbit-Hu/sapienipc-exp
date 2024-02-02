"""
    References:
    - "IPC paper": Li, Minchen, Zachary Ferguson, Teseo Schneider, Timothy R. Langlois, Denis Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. "Incremental potential contact: intersection-and inversion-free, large-deformation dynamics." ACM Trans. Graph. 39, no. 4 (2020): 49.
    - "CG slides": Xu, Zhiliang. "ACMS 40212/60212: Advanced Scientific Computing, Lecture 8: Fast Linear Solvers (Part 5)." (https://www3.nd.edu/~zxu2/acms60212-40212/Lec-09-5.pdf)
    - "FEM tutorial": Sifakis, Eftychios. "FEM simulation of 3D deformable solids: a practitioner's guide to theory, discretization and model reduction. Part One: The classical FEM method and discretization methodology." In Acm siggraph 2012 courses, pp. 1-50. 2012.
    
    Supported Energies:
    - Kinetic
    - Elastic
    - Affine
    - Collision
    - Friction
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ipc_system import IPCSystem

import numpy as np
import warp as wp
import sapien

from .ipc_utils.warp_types import *

from .ipc_kernels.cg_solver_kernels import *
from .ipc_kernels.utils_kernels import *
from .ipc_kernels.energy_kernels.kinetic_kernels import *
from .ipc_kernels.energy_kernels.elastic_kernels import *
from .ipc_kernels.energy_kernels.collision_kernels import *
from .ipc_kernels.energy_kernels.friction_kernels import *
from .ipc_kernels.energy_kernels.affine_kernels import *
from .ipc_kernels.energy_kernels.dbc_kernels import *
from .ipc_kernels.energy_kernels.constraint_kernels import *
from .ipc_utils.logging_utils import ipc_logger


class EnergyCalculator:
    def __init__(self, system: IPCSystem):
        self.system = system

    def preprocess(self):
        raise NotImplementedError

    def compute_energy(self, x):
        raise NotImplementedError

    def compute_diff(self, x, grad_coeff):
        raise NotImplementedError


class KineticEnergyCalculator(EnergyCalculator):
    """
    The kinetic (+ gravity potential) term of the Incremental Potential.
    K = 1/2 (x-x_hat)^T M (x-x_hat),
    where x_hat = x_prev + h v_prev + h^2 M^{-1} f_ext, f_ext = gravity.
    """

    def __init__(self, system: IPCSystem):
        super().__init__(system)

    def compute_energy(self, x: wp.array):
        s = self.system
        c = s.config
        wp.launch(
            kernel=kinetic_energy_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                s.particle_scene,
                s.particle_component,
                s.component_removed,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_q_prev_step,
                s.particle_qd_prev_step,
                s.particle_mass,
                s.particle_mask,
                c.gravity,
                c.time_step,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config
        wp.launch(
            kernel=kinetic_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                s.particle_component,
                s.component_removed,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_q_prev_step,
                s.particle_qd_prev_step,
                s.particle_mass,
                s.particle_mask,
                c.gravity,
                c.time_step,
                grad_coeff,
            ],
            outputs=[s.particle_grad],
            device=c.device,
        )


class ElasticEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config
        wp.launch(
            kernel=elastic_energy_kernel,
            dim=s.n_tets,
            inputs=[
                x,
                s.particle_scene,
                s.block_indices,
                s.tet_block_ids,
                s.block_status,
                s.tet_inv_Dm,
                s.rest_volumes,
                s.tet_materials,
                c.time_step,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config
        wp.launch(
            kernel=elastic_diff_kernel,
            dim=s.n_tets,
            inputs=[
                x,
                s.particle_mask,
                s.block_indices,
                s.tet_block_ids,
                s.block_status,
                s.tet_inv_Dm,
                s.rest_volumes,
                s.tet_materials,
                c.time_step,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks],
            device=c.device,
        )


class ElasticEnergy2DCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config
        wp.launch(
            kernel=elastic_energy_2d_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.particle_scene,
                s.block_indices,
                s.tri_block_ids,
                s.block_status,
                s.tri_inv_Dm,
                s.rest_areas,
                s.thicknesses,
                s.tri_materials,
                c.time_step,
            ],
            outputs=[s.energy],
            device=c.device,
        )
        wp.launch(
            kernel=bending_energy_2d_kernel,
            dim=s.n_hinges,
            inputs=[
                x,
                s.particle_q_rest,
                s.particle_scene,
                s.block_indices,
                s.hinge_block_ids,
                s.block_status,
                s.hinge_rest_angles,
                s.hinge_materials,
                c.time_step,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config
        wp.launch(
            kernel=elastic_diff_2d_kernel,
            dim=s.n_triangles,
            inputs=[
                x,
                s.particle_mask,
                s.block_indices,
                s.tri_block_ids,
                s.block_status,
                s.tri_inv_Dm,
                s.rest_areas,
                s.thicknesses,
                s.tri_materials,
                c.time_step,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks],
            device=c.device,
        )
        wp.launch(
            kernel=bending_diff_2d_kernel,
            dim=s.n_hinges,
            inputs=[
                x,
                s.particle_q_rest,
                s.particle_mask,
                s.block_indices,
                s.hinge_block_ids,
                s.block_status,
                s.hinge_rest_angles,
                s.hinge_materials,
                c.time_step,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks],
            device=c.device,
        )


class AffineEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=affine_energy_kernel,
            dim=s.n_affines,
            inputs=[
                x,
                s.particle_scene,
                s.block_indices,
                s.affine_block_ids,
                s.block_status,
                s.affine_volumes,
                s.affine_mass,
                s.particle_q_prev_step,
                s.particle_qd_prev_step,
                s.particle_mask,
                c.gravity,
                c.time_step,
                c.kappa_affine,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config

        wp.launch(
            kernel=affine_diff_kernel,
            dim=s.n_affines,
            inputs=[
                x,
                s.block_indices,
                s.affine_block_ids,
                s.block_status,
                s.affine_volumes,
                s.affine_mass,
                s.particle_q_prev_step,
                s.particle_qd_prev_step,
                s.particle_mask,
                c.gravity,
                c.time_step,
                c.kappa_affine,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks],
            device=c.device,
        )


class CollisionEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_energy_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                s.particle_scene,
                s.n_static_blocks,
                s.contact_counter,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                # c.ground_altitude,
                s.plane_normals,
                s.plane_offsets,
                c.kappa,
                c.d_hat,
                c.ee_classify_thres,
                c.ee_mollifier_thres,
            ],
            outputs=[
                s.contact_d,
                s.contact_c,
                s.contact_eps_cross,
                s.block_status,
                s.energy,
            ],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config

        wp.launch(
            kernel=collision_diff_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                s.particle_mask,
                s.n_static_blocks,
                s.contact_counter,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                # c.ground_altitude,
                s.plane_normals,
                s.plane_offsets,
                c.kappa,
                c.d_hat,
                c.time_step,
                s.contact_d,
                s.contact_c,
                s.contact_eps_cross,
                grad_coeff,
            ],
            outputs=[
                s.contact_dd_dx,
                s.contact_dc_dx,
                s.particle_grad,
                s.blocks,
                s.particle_collision_force,
            ],
            device=c.device,
        )


class FrictionEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def preprocess(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=friction_preprocess_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                s.n_static_blocks,
                s.contact_counter,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                c.kappa,
                c.d_hat,
                c.time_step,
                s.contact_d,
                s.contact_c,
                s.contact_eps_cross,
                s.block_status,
                s.contact_force,
                s.contact_T,
            ],
            device=c.device,
        )

    def compute_energy(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=friction_energy_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                s.particle_scene,
                s.particle_q_prev_step,
                s.n_static_blocks,
                s.contact_counter,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                c.eps_v,
                c.time_step,
                s.block_status,
                s.contact_force,
                s.contact_T,
                s.contact_friction,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config

        wp.launch(
            kernel=friction_diff_kernel,
            dim=c.max_blocks,
            inputs=[
                x,
                s.particle_q_prev_step,
                s.particle_mask,
                s.n_static_blocks,
                s.contact_counter,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                c.eps_v,
                c.time_step,
                s.block_status,
                s.contact_force,
                s.contact_T,
                s.contact_friction,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks, s.particle_friction_force],
            device=c.device,
        )


class DBCEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=dbc_energy_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                s.particle_scene,
                s.particle_component,
                s.component_removed,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_mass,
                s.particle_dbc_mask,
                s.particle_dbc_q,
                c.kappa_con,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config

        wp.launch(
            kernel=dbc_diff_kernel,
            dim=s.n_particles,
            inputs=[
                x,
                s.particle_component,
                s.component_removed,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_mass,
                s.particle_dbc_mask,
                s.particle_dbc_q,
                c.kappa_con,
                grad_coeff,
            ],
            outputs=[s.particle_grad],
            device=c.device,
        )


class ConstraintEnergyCalculator(EnergyCalculator):
    def __init__(self, system):
        super().__init__(system)

    def compute_energy(self, x):
        s = self.system
        c = s.config

        wp.launch(
            kernel=constraint_energy_kernel,
            dim=s.n_constraints,
            inputs=[
                x,
                s.particle_q_prev_step,
                s.particle_scene,
                s.constraint_param,
                s.constraint_lambda,
                s.constraint_block_ids,
                s.block_status,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.constraint_mu,
                c.time_step,
            ],
            outputs=[s.energy],
            device=c.device,
        )

    def compute_diff(self, x, grad_coeff):
        s = self.system
        c = s.config

        wp.launch(
            kernel=constraint_diff_kernel,
            dim=s.n_constraints,
            inputs=[
                x,
                s.particle_q_prev_step,
                s.particle_scene,
                s.particle_mask,
                s.constraint_param,
                s.constraint_lambda,
                s.constraint_block_ids,
                s.block_status,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.constraint_mu,
                c.time_step,
                grad_coeff,
            ],
            outputs=[s.particle_grad, s.blocks],
            device=c.device,
        )
