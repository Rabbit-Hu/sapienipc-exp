"""
References: \n
- "IPC paper": Li, Minchen, Zachary Ferguson, Teseo Schneider, Timothy R. Langlois, Denis Zorin, Daniele Panozzo, Chenfanfu Jiang, and Danny M. Kaufman. "Incremental potential contact: intersection-and inversion-free, large-deformation dynamics." ACM Trans. Graph. 39, no. 4 (2020): 49. \n
- "C-IPC paper": Li, Minchen, Danny M. Kaufman, and Chenfanfu Jiang. “Codimensional Incremental Potential Contact.” ACM Transactions on Graphics 40, no. 4 (August 31, 2021): 1–24. https://doi.org/10.1145/3450626.3459767. \n
- "ABD paper": Lan, Lei, Danny M. Kaufman, Minchen Li, Chenfanfu Jiang, and Yin Yang. “Affine Body Dynamics: Fast, Stable & Intersection-Free Simulation of Stiff Materials.” arXiv, January 31, 2022. https://doi.org/10.48550/arXiv.2201.10022. \n
- "CG slides": Xu, Zhiliang. "ACMS 40212/60212: Advanced Scientific Computing, Lecture 8: Fast Linear Solvers (Part 5)." (https://www3.nd.edu/~zxu2/acms60212-40212/Lec-09-5.pdf) \n
- "FEM tutorial": Sifakis, Eftychios. "FEM simulation of 3D deformable solids: a practitioner's guide to theory, discretization and model reduction. Part One: The classical FEM method and discretization methodology." In Acm siggraph 2012 courses, pp. 1-50. 2012. \n
"""

import typing
from typing import Union

import numpy as np
import torch
import warp as wp
import sapien

from .ipc_kernels.utils_kernels import *
from .ipc_kernels.abd_utils_kernels import *
from .ipc_kernels.line_search_kernels import *
from .ipc_kernels.cg_solver_kernels import *
from .ipc_kernels.energy_kernels.constraint_kernels import *
from .ipc_energy import *
from .ipc_component import *
from .ipc_cg_solver import CGSolver
from .ipc_contact_filter import IPCContactFilter
from .ipc_ccd import IPCCCD
from .ipc_utils.warp_types import *
from .ipc_utils.global_defs import *
from sapienipc.ipc_utils.logging_utils import ipc_logger


class IPCSystemConfig:
    """Configuration class for IPCSystem."""

    def __init__(self) -> None:
        ######## Memory config ########
        #: Max number of components in the system.
        self.max_components: int = 1 << 12
        #: Max number of scenes in the system.
        self.max_scenes: int = 1 << 8
        #: Max number of planes in the system.
        self.max_planes: int = 1 << 12
        #: Max number of planes in each scene handled by the system.
        #: This is used in the broad phase (filtering contact pairs).
        self.max_planes_per_scene: int = 1 << 8
        #: Max number of particles in the system.
        self.max_particles: int = 1 << 16
        #: Max number of particles in each scene handled by the system.
        #: This is used in the broad phase (filtering contact pairs).
        self.max_surface_primitives_per_scene: int = 1 << 14
        #: Max number of blocks in each scene.
        #: Number of blocks = sum(number of FEM tetrahedra) + number of ABD bodies
        #: + number of contact pairs after the broad phase.
        self.max_blocks: int = 1 << 20
        self.max_constraints: int = 1 << 10
        self.max_motors: int = 1 << 10

        #: Device of the system, like "cpu", "cuda:0", etc.
        self.device = wp.get_preferred_device()  # ["cpu", "cuda:0", ...]
        #: :default: False
        #:
        #: Set to True to enable the debug mode. In the debug mode, there
        #: will be more debug outputs and assertions, lowering the efficiency
        #: of the system.
        self.debug = False
        #: :default: True
        #:
        #: Set to True to enable CUDA graph for faster line search and CG solver.
        self.use_graph = True

        ######## Algorithm hyperparameters ########
        #: :default: 0.025 s
        #:
        #: Time step size of the simulation.
        self.time_step = 0.025
        #: :default: (0, 0, -9.8) kg m / s^2
        #:
        #: Gravity in the simulation.
        self.gravity: wp.vec3 = wp.vec3(0, 0, -9.8)
        #: :default: 1e-2 m
        #:
        #: Distance threshold for barrier function (Eq 6, IPC paper).
        self.d_hat = 1e-2
        #: :default: 1e-2 m
        #:
        #: Stopping threshold for Newton's method (Algo 1, IPC paper).
        self.eps_d = 1e-2
        #: :default: 1e-2 m / s
        #:
        #: Relative velocity threshold for smoothed friction (Eq 13, IPC paper).
        self.eps_v = 1e-2
        #: :default: 1e9 m / s
        #:
        #: Max velocity for broad phase of collision detection. At the beginning
        #: of each time step, contact pairs that are at least (v_max * time_step + d_hat)
        #: apart from each other may be ignored.
        #:
        #: .. note:: When v_max is too small, model penetration may happen,
        #:     leading to wrong results.
        self.preprocess_every_newton_step = True
        #: :default: True
        #:
        #: If True: re-compute all contact pairs at the beginning of each Newton 
        #: iteration (slower; more accurate friction). If False: re-compute all 
        #: contact pairs only at the beginning of each time step (faster; less 
        #: accurate friction).
        self.v_max = 1e9
        #: :default: False
        #:
        #: Set to True to clip all FEM particles' velocities into [0, v_max]
        self.clip_velocity = False
        #: :default: 1e3
        #:
        #: Coefficient for barrier functions (Eq 5, IPC paper).
        #:
        #: .. note:: When kappa is too small, the simulator may get stuck when there is contact.
        #:     When kappa is too large, there will be larger damping, and
        #:     the simulator may take more substeps (Newton & CG steps) to converge or
        #:     even never converge and get stuck.
        self.kappa = 1e3
        #: :default: 1e3
        #:
        #: Coefficient for the orthogonality potential for ABD (Eq 6, ABD paper).
        #:
        #: .. note:: When it kappa_affine too small, the ABD objects may deform undesirably.
        #:     When kappa_affine is too large, there will be larger damping, and
        #:     the simulator may take more substeps (Newton & CG steps) to converge or
        #:     even never converge and get stuck.
        self.kappa_affine = 1e3
        #: :default: 1e2
        #:
        #: Coefficient for the L2 penalty terms for kinematic targets (kappa_con * ||x_target - x||^2).
        #:
        #: .. note:: When kappa_con is too small, the particles may not reach their
        #:     kinematic targets (if any).
        #:     When kappa_con is too large, there will be larger damping, and
        #:     the simulator may take more substeps (Newton & CG steps) to converge or
        #:     even never converge and get stuck.
        self.kappa_con = 1e2
        #: :default: 0.7
        #: :range: (0.0, 1.0)
        #:
        #: Slackness parameters in CCD (Continuous Collision Detection) (Alg 1, C-IPC paper).
        self.constraint_mu_init = 1e3
        #: :default: 1e1
        #:
        #: Initial value of the Lagrange multiplier in Augmented Lagrangian method for constraints.
        self.constraint_mu_scale = 2.0
        #: :default: 2.0
        #:
        #: Scaling factor of the Lagrange multiplier in Augmented Lagrangian method for constraints.
        #: At each Newton iteration, the Lagrange multiplier is multiplied by this factor.
        self.constraint_mu_max = 1e8
        self.use_augmented_lagrangian = True
        self.ccd_slackness = 0.7
        #: :default: 0.0 m
        #:
        #: Thickness value. Set to a very small positive number (much smaller than d_hat)
        #: to avoid model penetration due to numerical error in CCD.
        self.ccd_thickness = 0.0
        #: :default: 0.0
        #:
        #: Tet inversion threshold. Set to a very small positive number
        #: to avoid tetrahedron inversion due to numerical error in CCD.
        self.ccd_tet_inversion_thres = 0.0
        #: :default: 1e-6
        #:
        #: When the angle between two edges is smaller than arcsin(ee_classify_thres),
        #: the system does not compute the distance between their corresponding lines as their
        #: edge-edge distance, as this may underestimate their distance due to numerical error.
        self.ee_classify_thres = 1e-3  # force using point-point distance if sin < thres
        #: :default: 1e-3
        #:
        #: Activate mollifier for edge-edge barrier function when
        #: (\|cross(e_1, e_2)\|^2 < \|e_1_rest\|^2 * \|e_1_rest\|^2 * thres).
        #: (Eq 24, IPC paper).
        self.ee_mollifier_thres = 1e-3

        # Energies enabling and disabling
        self.enable_kinetic = True
        self.enable_elastic = True
        self.enable_elastic_2d = True
        self.enable_affine = True
        self.enable_collision = True
        self.enable_friction = True
        self.enable_dbc_energy = True
        self.enable_constraint_energy = True

        #: :default: True
        #:
        #: Set to False to ignore self collision (collision between parts of
        #: the same component). If you know there will not be any self collision
        #: in the environment, you can safely set this to False to save memory
        #: improve efficiency.
        self.allow_self_collision = True

        ######## Solver config ########
        #: :default: 10
        #:
        #: Max number of iterations in Newton's method. This can significantly influence
        #: the quality of simulation results.
        self.newton_max_iters = 10
        #: :default: 40
        #:
        #: Max number of iterations in CG (Conjugate Gradients) in each Newton iteration.
        self.cg_max_iters = 40
        #: :default: 10
        #:
        #: Max number of iterations in line search (Alg 1 Line 11-15, IPC paper).
        self.line_search_max_iters = 10
        #: :default: 100
        #:
        #: Max number of iterations in Additive CCD (Alg 1, C-IPC paper).
        self.ccd_max_iters = 100
        #: :default: "jacobi"
        #: :choices: ["jacobi", "none"]
        #:
        #: Preconditioner in CG solver.
        self.precondition = "jacobi"
        #: :default: 1e-3
        #:
        #: CG convergence threshold: quit CG if dot(z, r) < cg_error_tolerance ** 2 * dot (z0, r0) (Page 16, CG slides).
        self.cg_error_tolerance = 1e-3
        #: :default: 10
        #:
        #: Check if CG converges every cg_error_frequency steps.
        self.cg_error_frequency = 10

        #: :default: 3
        #:
        #: Number of sweeps in Jacobi EVD (Eigenvalue Decomposition).
        self.spd_project_max = 3

        #: :default: False
        #:
        #: Set to True to use the DBC (Dirichlet Boundary Condition) energy (an L2 penalty)
        #: without first computing CCD and step the particles towards the kinematic targets.
        self.use_constraint_dbc_only = False

    def __repr__(self) -> str:
        return f"IPCSystemConfig({self.__dict__})"


class IPCSystem(sapien.System):
    def __init__(self, config: IPCSystemConfig):
        super().__init__()

        ipc_logger.info(config)

        self.name = "ipc"

        self.config = config
        self.debug = config.debug

        # sub-systems (handle different parts of the IPC algorithm)
        self.cg_solver = CGSolver(self)
        self.filter = IPCContactFilter(self)
        self.ccd = IPCCCD(self)

        self.energy_calcs: typing.Dict[str, EnergyCalculator] = {}
        if config.enable_kinetic:
            self.energy_calcs["kinetic"] = KineticEnergyCalculator(self)
        if config.enable_elastic:
            self.energy_calcs["elastic"] = ElasticEnergyCalculator(self)
        if config.enable_elastic_2d:
            self.energy_calcs["elastic_2d"] = ElasticEnergy2DCalculator(self)
        if config.enable_affine:
            self.energy_calcs["affine"] = AffineEnergyCalculator(self)
        if config.enable_collision:
            self.energy_calcs["collision"] = CollisionEnergyCalculator(self)
        if config.enable_friction:
            assert config.enable_collision
            self.energy_calcs["friction"] = FrictionEnergyCalculator(self)
        if config.enable_dbc_energy:
            self.energy_calcs["dbc"] = DBCEnergyCalculator(self)
        if config.enable_constraint_energy:
            self.energy_calcs["constraint"] = ConstraintEnergyCalculator(self)

        max_c = config.max_components
        max_scenes = config.max_scenes
        max_pla = config.max_planes
        max_plane_per_scene = config.max_planes_per_scene
        max_p = config.max_particles
        max_sur = config.max_surface_primitives_per_scene
        max_b = config.max_blocks
        max_constraints = config.max_constraints
        device = config.device

        self.n_components = 0
        self.n_scenes = 0
        self.n_planes = 0
        self.n_particles = 0  # FEM vertices + ABD affine proxies + ABD vertices
        self.n_tets = 0  # FEM tets
        self.n_triangles = 0  # FEM triangles
        self.n_hinges = 0
        self.n_affines = 0  # ABD affine bodies
        self.n_static_blocks = 0  # FEM tets + ABD affine bodies
        self.n_blocks_this_step = 0  # static blocks + contact blocks
        self.n_constraints = 0  # general constraints
        self.n_surface_triangles_list = []  # surface triangles
        self.n_surface_edges_list = []  # surface edges
        self.n_surface_particles_list = []  # surface vertices
        self.n_surface_planes_list = []  # surface planes
        self.n_surface_triangles_max = 0
        self.n_surface_edges_max = 0
        self.n_surface_particles_max = 0
        self.n_surface_planes_max = 0

        self.step_count = 0
        self.time = 0.0  # increment by config.time_step every step

        self.line_search_graph = None

        self.components = []
        self.scenes = []

        ######## Component Data ########

        self.component_friction = wp.zeros(max_c, dtype=wp.float32, device=device)
        self.component_removed = wp.zeros(max_c, dtype=wp.int32, device=device)
        self.component_affine = wp.zeros(max_c, dtype=wp.int32, device=device)
        self.component_group = wp.zeros(max_c, dtype=wp.int32, device=device)

        ######## 1-Element Data ########

        self.contact_counter = wp.zeros(1, dtype=int, device=device)

        ######## Per-Scene Data ########

        self.energy = wp.zeros(max_scenes, dtype=wp.float32, device=device)
        self.ccd_step = wp.zeros(max_scenes, dtype=wp.float32, device=device)
        self.energy_prev = wp.zeros(max_scenes, dtype=wp.float32, device=device)

        ######## Plane Data ########

        self.plane_normals = wp.zeros(max_pla, wp.vec3, device=device)
        self.plane_offsets = wp.zeros(max_pla, dtype=wp.float32, device=device)
        self.plane_component = wp.zeros(max_pla, wp.int32, device=device)
        self.plane_scene = wp.zeros(max_pla, wp.int32, device=device)

        ######## Particle Data ########

        self.particle_q = wp.zeros(max_p, wp.vec3, device=device)
        self.particle_qd = wp.zeros(max_p, wp.vec3, device=device)
        self.particle_q_prev_step = wp.zeros(max_p, wp.vec3, device=device)
        self.particle_qd_prev_step = wp.zeros(max_p, wp.vec3, device=device)

        self.particle_q_prev_it = wp.zeros(max_p, wp.vec3, device=device)
        # particle_q of the previous Newton iteration
        self.particle_p = wp.zeros(max_p, wp.vec3, device=device)
        # update of particle_q in the current Newton iteration
        # Target position: particle_q_next_it = particle_q_prev_step + particle_p * alpha

        self.particle_q_rest = wp.zeros(max_p, wp.vec3, device=device)
        self.particle_mass = wp.zeros(max_p, dtype=wp.float32, device=device)
        self.particle_mask = wp.zeros(max_p, dtype=wp.float32, device=device)
        # For each particle, 1.0 if free, 0.0 if fixed
        self.particle_affine = wp.zeros(max_p, dtype=wp.int32, device=device)
        # For each particle, the index of the affine body that it corresponds to
        # -1 if not an affine body, i.e. a FEM vertex
        # -2 if one of the 4 affine proxies of some affine body
        self.particle_component = wp.zeros(max_p, wp.int32, device=device)
        self.particle_scene = wp.zeros(max_p, wp.int32, device=device)
        # For each particle, the id of the component that it corresponds to
        self.particle_grad = wp.zeros(max_p, wp.vec3, device=device)
        # Gradient of the IP function

        ######## Particle Data: Contact Forces (for User API) ########

        self.particle_collision_force = wp.zeros(max_p, wp.vec3, device=device)
        self.particle_friction_force = wp.zeros(max_p, wp.vec3, device=device)

        ######## Particle Data (DBC: equality Constraints that changes every time step) ########

        self.particle_dbc_tag = wp.zeros(
            max_p, dtype=wp.int32, device=device
        )  # before _process_dbc
        self.particle_dbc_mask = wp.zeros(
            max_p, dtype=wp.float32, device=device
        )  # after _process_dbc
        self.particle_dbc_q = wp.zeros(max_p, dtype=wp.vec3, device=device)
        self.particle_dbc_delta_q = wp.zeros(max_p, dtype=wp.vec3, device=device)

        ######## Surface Data ########

        self.surface_planes = wp.zeros(
            (max_scenes, max_plane_per_scene), dtype=wp.int32, device=device
        )
        self.surface_particles = wp.zeros(
            (max_scenes, max_sur), dtype=wp.int32, device=device
        )
        self.surface_edges = wp.zeros(
            (max_scenes, max_sur, 2), dtype=wp.int32, device=device
        )
        self.surface_triangles = wp.zeros(
            (max_scenes, max_sur, 3), dtype=wp.int32, device=device
        )

        self.n_surface_planes = wp.zeros(max_scenes, dtype=wp.int32, device=device)
        self.n_surface_particles = wp.zeros(max_scenes, dtype=wp.int32, device=device)
        self.n_surface_edges = wp.zeros(max_scenes, dtype=wp.int32, device=device)
        self.n_surface_triangles = wp.zeros(max_scenes, dtype=wp.int32, device=device)

        ######## Block Data ########

        self.blocks = wp.zeros((max_b, 4, 4), wp.mat33, device=device)
        # Hessian of the IP function, each 12x12 submatrix is stored as a 4x4 array
        # of 3x3 wp.mat33 matrices
        self.block_indices = wp.zeros((max_b, 4), dtype=int, device=device)
        # For each block, the indices of <= 4 particles that it corresponds to
        # -1 if no particle
        # Possible block types: [FEM tet, ABD affine body, contact pairs]
        self.block_status = wp.zeros(max_b, dtype=wp.int32, device=device)
        # For each block, 1 if enabled, 0 if disabled

        ######## Block Data: FEM Tets ########

        self.tet_block_ids = wp.zeros(max_b, dtype=wp.int32, device=device)
        # List of block ids (first dim coordinates in self.blocks) that are FEM tets
        self.tet_inv_Dm = wp.zeros(max_b, dtype=wp.mat33, device=device)
        # For each FEM tet, the inverse of its "reference shape matrix" Dm (Page 28, FEM tutorial)
        self.tet_materials = wp.zeros((max_b, 2), dtype=wp.float32, device=device)
        self.rest_volumes = wp.zeros(max_b, dtype=wp.float32, device=device)

        ######## Block Data: FEM Triangles ########

        self.tri_block_ids = wp.zeros(max_b, dtype=wp.int32, device=device)
        # List of block ids (first dim coordinates in self.blocks) that are FEM tets
        self.tri_inv_Dm = wp.zeros(max_b, dtype=wp.mat22, device=device)
        # For each FEM tet, the inverse of its "reference shape matrix" Dm (Page 28, FEM tutorial)
        self.tri_materials = wp.zeros((max_b, 2), dtype=wp.float32, device=device)
        self.rest_areas = wp.zeros(max_b, dtype=wp.float32, device=device)
        self.thicknesses = wp.zeros(max_b, dtype=wp.float32, device=device)

        ######## Block Data: FEM Hinges ########

        self.hinge_block_ids = wp.zeros(max_b, dtype=wp.int32, device=device)
        # List of block ids (first dim coordinates in self.blocks) that are FEM hinges
        self.hinge_rest_angles = wp.zeros(max_b, dtype=wp.float32, device=device)
        self.hinge_materials = wp.zeros(max_b, dtype=wp.float32, device=device)

        ######## Block Data: ABD Affine Bodies ########

        self.affine_block_ids = wp.zeros(max_c, dtype=wp.int32, device=device)
        # List of block ids (first dim coordinates in self.blocks) that are affine proxies
        self.affine_volumes = wp.zeros(max_c, dtype=wp.float32, device=device)
        self.affine_mass = wp.zeros((max_c, 4, 4), dtype=wp.mat33, device=device)

        ######## Block Data: General Constraints ########

        self.constraint_block_ids = wp.zeros(
            max_constraints, dtype=wp.int32, device=device
        )
        # List of block ids (first dim coordinates in self.blocks) that are general constraints
        self.constraint_param = wp.zeros(
            (max_constraints, 5), dtype=wp.float32, device=device
        )
        # For each general constraint, the constraint parameters (meaning depends on the type of constraint)
        self.constraint_lambda = wp.zeros(
            max_constraints, dtype=wp.float32, device=device
        )
        self.constraint_mu = self.config.constraint_mu_init
        # For each general constraint, the Lagrange multiplier in Augmented Lagrangian method

        ######## Block Data: Contact ########

        self.block_type = wp.zeros((max_b, 2), dtype=wp.int32, device=device)
        # For block types see ipc_utils/global_defs.py
        self.contact_d = wp.zeros(max_b, dtype=wp.float32, device=device)
        self.contact_dd_dx = wp.zeros((max_b, 4), dtype=wp.vec3, device=device)
        self.contact_c = wp.zeros(
            max_b, dtype=wp.float32, device=device
        )  # EE mollifier
        self.contact_eps_cross = wp.zeros(max_b, dtype=wp.float32, device=device)
        self.contact_dc_dx = wp.zeros((max_b, 4), dtype=wp.vec3, device=device)
        self.contact_force = wp.zeros(max_b, dtype=wp.float32, device=device)
        self.contact_T = wp.zeros(
            (max_b, 4), dtype=mat32, device=device
        )  # sliding basis
        self.contact_friction = wp.zeros(max_b, dtype=wp.float32, device=device)

        ######## Debug Utils ########

        self.random_states = wp.zeros(max_p, wp.uint32, device=device)

    def get_name(self):
        return self.name

    def step(self):
        ipc_logger.info(f"Step = {self.step_count}, time = {self.time:.4f}")

        if self.debug:
            assert not np.isnan(self.particle_q.numpy()).any()
            assert not np.isnan(self.particle_qd.numpy()).any()

        wp.copy(self.particle_q_prev_step, self.particle_q, count=self.n_particles)
        wp.copy(self.particle_qd_prev_step, self.particle_qd, count=self.n_particles)

        self._process_dbc()

        self._preprocess_energy(self.particle_q_prev_step)

        if self.config.use_graph:
            wp.capture_begin()
            self._line_search()
            self.line_search_graph = wp.capture_end()

        # if self.debug:
        #     ipc_logger.debug(
        #         f"self.block_indices:\n"
        #         + str(self.block_indices.numpy()[: self.n_blocks_this_step])
        #     )

        newton_iter = 0

        while True:
            newton_iter += 1

            if newton_iter == 1:
                self._constraint_reset()
            else:
                self._constraint_update()
            # print(f"mu = {self.constraint_mu}")
            # print(f"la = {self.constraint_lambda.numpy()[:self.n_constraints]}")

            # if self.debug:
            #     ipc_logger.debug(
            #         f"newton_iter = {newton_iter}, self.particle_q:\n"
            #         + str(self.particle_q.numpy()[: self.n_particles])
            #     )

            # if self.config.debug:
            #     ipc_logger.debug(f"Newton iteration {newton_iter}")

            wp.copy(self.particle_q_prev_it, self.particle_q, count=self.n_particles)

            if self.config.preprocess_every_newton_step:
                self._preprocess_energy(self.particle_q_prev_it)

            self._compute_energy(
                self.particle_q,
                output_debug_info=self.config.debug,  # and not self.config.use_graph,
            )
            # if self.config.debug:
            # ipc_logger.debug(f"Newton iteration {newton_iter}, initial energy = {self.energy.numpy()[0]}")

            self._compute_diff(self.particle_q, -1.0)

            # if self.debug:
            #     ipc_logger.debug(
            #         f"self.contact_d:\n"
            #         + str(self.contact_d.numpy()[: self.n_blocks_this_step])
            #     )

            # print("block_indices", self.block_indices.numpy()[:self.n_static_blocks])
            # print("blocks", self.blocks.numpy()[:self.n_static_blocks])

            # ipc_logger.info(f"self.particle_grad = \n{self.particle_grad.numpy()[:self.n_particles]}")

            self.cg_solver.solve()

            # # DEBUG: solve with np
            # hess_np = self._get_dense_matrix_brute_force()

            # # eig, U = np.linalg.eig(hess_np)
            # # eig, U = eig.real, U.real
            # # print(f"max eigen value = {eig.max():.4e}, min eigen value = {eig.min():.4e}")
            # # eig = np.clip(eig, a_min=0., a_max=None)
            # # hess_np = U @ np.diag(eig) @ U.T

            # grad_np = self.particle_grad.numpy()[: self.n_particles].reshape(-1)
            # p_np = np.linalg.lstsq(hess_np, grad_np, rcond=None)[0]
            # wp_slice(self.particle_p, 0, self.n_particles).assign(p_np.reshape(-1, 3))

            if self.config.debug:
                assert not np.isnan(self.particle_p.numpy()).any()

            # TODO: compute p_inf_norm on CUDA
            p_inf_norm = np.abs(self.particle_p.numpy()[: self.n_particles]).max()
            if p_inf_norm < self.config.eps_d * self.config.time_step:
                break

            if self.config.clip_velocity:
                self._clip_velocity()

            self.ccd.compute_step(self.particle_q, self.particle_p)

            if self.config.use_graph:
                wp.capture_launch(self.line_search_graph)
            else:
                self._line_search()

            # Check line search success
            if self.debug:
                if np.any(
                    self.energy.numpy()[: self.n_scenes]
                    >= self.energy_prev.numpy()[: self.n_scenes]
                ):
                    ipc_logger.warning(
                        f"Line search failed, energy increased for scenes {np.where(self.energy.numpy()[:self.n_scenes] >= self.energy_prev.numpy()[:self.n_scenes])[0]}"
                    )

            if self.debug:
                ipc_logger.debug(
                    f"Newton iteration {newton_iter} finished, energy = {self.energy.numpy()[:self.n_scenes]}"
                )

            if newton_iter >= self.config.newton_max_iters:
                ipc_logger.info(
                    f"Newton iteration did not converge after {newton_iter} iterations, "
                    + f"p_inf = {p_inf_norm:.4e}"
                )
                break

        if self.config.debug:
            assert not np.isnan(self.particle_q.numpy()).any()

        wp.launch(
            kernel=apply_parent_affine_kernel,
            dim=self.n_particles,
            inputs=[
                0,
                self.particle_q,
                self.particle_q_rest,
                self.block_indices,
                self.affine_block_ids,
                self.particle_affine,
            ],
            device=self.config.device,
        )

        wp.launch(
            kernel=compute_velocity_kernel,
            dim=self.n_particles,
            inputs=[self.particle_q_prev_step, self.particle_q, self.config.time_step],
            outputs=[self.particle_qd],
            device=self.config.device,
        )

        # Clear DBC (if the user does not set them again, the particles will be free)
        wp_slice(self.particle_dbc_tag, 0, self.n_particles).zero_()

        self.step_count += 1
        self.time += self.config.time_step

    def register_fem_component(self, component: IPCFEMComponent):
        """
        component
        particle
        surface (particle, edge, triangle)
        block (general, tet)
        """

        c = component
        assert c.tet_mesh is not None

        ######## Component Data ########

        cid = self.n_components
        self.n_components += 1
        assert self.n_components <= self.config.max_components
        c.id_in_system = cid
        self.components.append(c)

        wp_slice(self.component_friction, cid, cid + 1).assign(np.array(c.friction))
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(0))
        wp_slice(self.component_group, cid, cid + 1).assign(
            np.array(c.group, dtype=np.int32)
        )

        ######## Scene Data ########

        if c.entity.scene not in self.scenes:
            sid = self.n_scenes
            self.n_scenes += 1
            assert self.n_scenes <= self.config.max_scenes
            self.scenes.append(c.entity.scene)
            self.n_surface_planes_list.append(0)
            self.n_surface_triangles_list.append(0)
            self.n_surface_edges_list.append(0)
            self.n_surface_particles_list.append(0)
        else:
            sid = self.scenes.index(c.entity.scene)

        ######## Particle Data ########

        pl = self.n_particles  # Particle Left (begin) id
        pr = pl + c.tet_mesh.n_vertices  # Particle Right (end) id
        self.n_particles = pr
        assert self.n_particles <= self.config.max_particles

        particle_q_slice = wp_slice(self.particle_q, pl, pr)
        c.array_slice = particle_q_slice
        c.particle_begin_index = pl
        c.particle_end_index = pr
        c.cuda_pointer = particle_q_slice.ptr
        c.cuda_stream = self.config.device.stream.cuda_stream
        c.size = type_size_in_bytes(particle_q_slice.dtype) * particle_q_slice.size

        pose = c.entity.get_pose()
        transformation_matrix = pose.to_transformation_matrix()
        wp_slice(self.particle_q, pl, pr).assign(
            c.tet_mesh.vertices @ transformation_matrix[:3, :3].T
            + transformation_matrix[:3, 3]
        )
        wp_slice(self.particle_qd, pl, pr).assign(
            np.tile(c.init_velocity, (pr - pl, 1))
        )
        wp_slice(self.particle_q_rest, pl, pr).assign(c.tet_mesh.vertices)
        wp_slice(self.particle_mass, pl, pr).assign(c.vertex_mass)
        wp_slice(self.particle_mask, pl, pr).assign(np.tile(1.0, pr - pl))
        wp_slice(self.particle_affine, pl, pr).assign(np.tile(AFFINE_NONE, pr - pl))
        wp_slice(self.particle_component, pl, pr).assign(np.tile(cid, pr - pl))
        wp_slice(self.particle_scene, pl, pr).assign(np.tile(sid, pr - pl))
        wp_slice(self.particle_dbc_tag, pl, pr).assign(np.tile(0, pr - pl))

        ######## Surface Data ########

        if c.tet_mesh.n_surface_vertices > 0:
            wp_slice(
                self.surface_particles[sid],
                self.n_surface_particles_list[sid],
                self.n_surface_particles_list[sid] + c.tet_mesh.n_surface_vertices,
            ).assign(c.tet_mesh.surface_vertices + pl)
            self.n_surface_particles_list[sid] += c.tet_mesh.n_surface_vertices
            self.n_surface_particles_max = max(
                self.n_surface_particles_max, self.n_surface_particles_list[sid]
            )
            assert (
                self.n_surface_particles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_particles, sid, sid + 1).assign(
                np.array(self.n_surface_particles_list[sid], dtype=np.int32)
            )

        if c.tet_mesh.n_surface_edges > 0:
            wp_slice(
                self.surface_edges[sid],
                self.n_surface_edges_list[sid],
                self.n_surface_edges_list[sid] + c.tet_mesh.n_surface_edges,
            ).assign(c.tet_mesh.surface_edges + pl)
            self.n_surface_edges_list[sid] += c.tet_mesh.n_surface_edges
            self.n_surface_edges_max = max(
                self.n_surface_edges_max, self.n_surface_edges_list[sid]
            )
            assert (
                self.n_surface_edges_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_edges, sid, sid + 1).assign(
                np.array(self.n_surface_edges_list[sid], dtype=np.int32)
            )

        if c.tet_mesh.n_surface_triangles > 0:
            wp_slice(
                self.surface_triangles[sid],
                self.n_surface_triangles_list[sid],
                self.n_surface_triangles_list[sid] + c.tet_mesh.n_surface_triangles,
            ).assign(c.tet_mesh.surface_triangles + pl)
            self.n_surface_triangles_list[sid] += c.tet_mesh.n_surface_triangles
            self.n_surface_triangles_max = max(
                self.n_surface_triangles_max, self.n_surface_triangles_list[sid]
            )
            assert (
                self.n_surface_triangles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_triangles, sid, sid + 1).assign(
                np.array(self.n_surface_triangles_list[sid], dtype=np.int32)
            )

        ######## Block Data ########

        bl = self.n_static_blocks  # block left (begin) id
        br = bl + c.tet_mesh.n_tets  # block right (end) id
        self.n_static_blocks = br
        assert self.n_static_blocks <= self.config.max_blocks

        c.block_begin_index = bl
        c.block_end_index = br

        if c.tet_mesh.n_tets > 0:
            wp_slice(self.block_indices, bl, br).assign(c.tet_mesh.tets + pl)
            wp_slice(self.block_status, bl, br).assign(np.tile(1, br - bl))
            wp_slice(self.block_type, bl, br).assign(
                np.tile(np.array([FEM_BLOCK, FEM_TET_BLOCK]), (br - bl, 1))
            )

        ######## Block Data: FEM Tets ########

        if c.tet_mesh.n_tets > 0:
            btl = self.n_tets  # tet left (begin) id
            btr = btl + c.tet_mesh.n_tets  # tet right (end) id
            self.n_tets = btr
            assert self.n_tets <= self.config.max_blocks

            wp_slice(self.tet_block_ids, btl, btr).assign(np.arange(bl, br))
            wp_slice(self.rest_volumes, btl, btr).assign(c.rest_volumes)
            wp_slice(self.tet_inv_Dm, btl, btr).assign(c.inv_Dm)
            wp_slice(self.tet_materials, btl, btr).assign(
                np.tile(np.array([c.k_mu, c.k_lambda]), (btr - btl, 1))
            )

    def unregister_fem_component(self, component: IPCFEMComponent):
        cid = component.id_in_system
        self.components.remove(component)

        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(1))

        if component.tet_mesh.n_tets > 0:
            bl = component.block_begin_index
            br = component.block_end_index
            wp_slice(self.block_status, bl, br).assign(np.tile(0, br - bl))

    def get_vertex_positions(
        self, component: Union[IPCFEMComponent, IPCABDComponent, IPCABDJointComponent]
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_q)[begin:end].clone()

    def get_vertex_velocities(
        self, component: Union[IPCFEMComponent, IPCABDComponent, IPCABDJointComponent]
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_qd)[begin:end].clone()

    def get_vertex_collision_forces(
        self, component: Union[IPCFEMComponent, IPCABDComponent]
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_collision_force)[begin:end].clone()

    def get_vertex_friction_forces(
        self, component: Union[IPCFEMComponent, IPCABDComponent]
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_friction_force)[begin:end].clone()

    def set_fem_positions(
        self,
        component: IPCFEMComponent,
        positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        if isinstance(positions, torch.Tensor) and positions.is_cuda:
            wp_wait_torch_stream(self.config.device)

        wp_slice(self.particle_q, begin, end).assign(
            convert_to_wp_array(positions, dtype=wp.vec3, device=self.config.device)
        )

    def set_fem_velocities(
        self,
        component: IPCFEMComponent,
        velocities: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        if isinstance(velocities, torch.Tensor) and velocities.is_cuda:
            wp_wait_torch_stream(self.config.device)

        wp_slice(self.particle_qd, begin, end).assign(
            convert_to_wp_array(velocities, dtype=wp.vec3, device=self.config.device)
        )

    def set_fem_kinematic_target(
        self,
        component: IPCFEMComponent,
        indices: Union[np.ndarray, torch.Tensor, wp.array],
        positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        # TODO: assert component is registered

        begin = component.particle_begin_index
        end = component.particle_end_index
        assert 0 <= begin <= end <= self.n_particles

        if (isinstance(indices, torch.Tensor) and indices.is_cuda) or (
            isinstance(positions, torch.Tensor) and positions.is_cuda
        ):
            wp_wait_torch_stream(self.config.device)

        indices = convert_to_wp_array(
            indices, dtype=wp.int32, device=self.config.device
        )
        positions = convert_to_wp_array(
            positions, dtype=wp.vec3, device=self.config.device
        )

        wp.launch(
            set_to_ones_at_indices_int_kernel,
            dim=len(indices),
            inputs=[
                wp_slice(self.particle_dbc_tag, begin, end),
                indices,
            ],
            device=self.config.device,
        )
        wp.launch(
            set_values_at_indices_vec3_kernel,
            dim=len(indices),
            inputs=[
                wp_slice(self.particle_dbc_q, begin, end),
                indices,
                positions,
            ],
            device=self.config.device,
        )

    def register_fem2d_component(self, component: IPCFEM2DComponent):
        c = component
        assert c.tri_mesh is not None

        ######## Component Data ########

        cid = self.n_components
        self.n_components += 1
        assert self.n_components <= self.config.max_components
        c.id_in_system = cid
        self.components.append(c)

        wp_slice(self.component_friction, cid, cid + 1).assign(np.array(c.friction))
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(0))
        wp_slice(self.component_group, cid, cid + 1).assign(
            np.array(c.group, dtype=np.int32)
        )

        ######## Scene Data ########

        if c.entity.scene not in self.scenes:
            sid = self.n_scenes
            self.n_scenes += 1
            assert self.n_scenes <= self.config.max_scenes
            self.scenes.append(c.entity.scene)
            self.n_surface_planes_list.append(0)
            self.n_surface_triangles_list.append(0)
            self.n_surface_edges_list.append(0)
            self.n_surface_particles_list.append(0)
        else:
            sid = self.scenes.index(c.entity.scene)

        ######## Particle Data ########

        pl = self.n_particles  # Particle Left (begin) id
        pr = pl + c.tri_mesh.n_vertices  # Particle Right (end) id
        self.n_particles = pr
        assert self.n_particles <= self.config.max_particles

        particle_q_slice = wp_slice(self.particle_q, pl, pr)
        c.array_slice = particle_q_slice
        c.particle_begin_index = pl
        c.particle_end_index = pr
        c.cuda_pointer = particle_q_slice.ptr
        c.cuda_stream = self.config.device.stream.cuda_stream
        c.size = type_size_in_bytes(particle_q_slice.dtype) * particle_q_slice.size

        pose = c.entity.get_pose()
        transformation_matrix = pose.to_transformation_matrix()
        wp_slice(self.particle_q, pl, pr).assign(
            c.tri_mesh.vertices @ transformation_matrix[:3, :3].T
            + transformation_matrix[:3, 3]
        )
        wp_slice(self.particle_qd, pl, pr).assign(
            np.tile(c.init_velocity, (pr - pl, 1))
        )
        wp_slice(self.particle_q_rest, pl, pr).assign(c.tri_mesh.vertices)
        wp_slice(self.particle_mass, pl, pr).assign(c.vertex_mass)
        wp_slice(self.particle_mask, pl, pr).assign(np.tile(1.0, pr - pl))
        wp_slice(self.particle_affine, pl, pr).assign(np.tile(AFFINE_NONE, pr - pl))
        wp_slice(self.particle_component, pl, pr).assign(np.tile(cid, pr - pl))
        wp_slice(self.particle_scene, pl, pr).assign(np.tile(sid, pr - pl))
        wp_slice(self.particle_dbc_tag, pl, pr).assign(np.tile(0, pr - pl))

        ######## Surface Data ########

        if c.tri_mesh.n_surface_vertices > 0:
            wp_slice(
                self.surface_particles[sid],
                self.n_surface_particles_list[sid],
                self.n_surface_particles_list[sid] + c.tri_mesh.n_surface_vertices,
            ).assign(c.tri_mesh.surface_vertices + pl)
            self.n_surface_particles_list[sid] += c.tri_mesh.n_surface_vertices
            self.n_surface_particles_max = max(
                self.n_surface_particles_max, self.n_surface_particles_list[sid]
            )
            assert (
                self.n_surface_particles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_particles, sid, sid + 1).assign(
                np.array(self.n_surface_particles_list[sid], dtype=np.int32)
            )

        if c.tri_mesh.n_surface_edges > 0:
            wp_slice(
                self.surface_edges[sid],
                self.n_surface_edges_list[sid],
                self.n_surface_edges_list[sid] + c.tri_mesh.n_surface_edges,
            ).assign(c.tri_mesh.surface_edges + pl)
            self.n_surface_edges_list[sid] += c.tri_mesh.n_surface_edges
            self.n_surface_edges_max = max(
                self.n_surface_edges_max, self.n_surface_edges_list[sid]
            )
            assert (
                self.n_surface_edges_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_edges, sid, sid + 1).assign(
                np.array(self.n_surface_edges_list[sid], dtype=np.int32)
            )

        if c.tri_mesh.n_surface_triangles > 0:
            wp_slice(
                self.surface_triangles[sid],
                self.n_surface_triangles_list[sid],
                self.n_surface_triangles_list[sid] + c.tri_mesh.n_surface_triangles,
            ).assign(c.tri_mesh.surface_triangles + pl)
            self.n_surface_triangles_list[sid] += c.tri_mesh.n_surface_triangles
            self.n_surface_triangles_max = max(
                self.n_surface_triangles_max, self.n_surface_triangles_list[sid]
            )
            assert (
                self.n_surface_triangles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_triangles, sid, sid + 1).assign(
                np.array(self.n_surface_triangles_list[sid], dtype=np.int32)
            )

        ######## Block Data ########

        n_tri = c.tri_mesh.n_triangles
        tri_bl = self.n_static_blocks  # block left (begin) id
        tri_br = tri_bl + n_tri  # block right (end) id
        n_hinge = len(c.hinges)
        hinge_bl = tri_br  # block left (begin) id
        hinge_br = hinge_bl + n_hinge  # block right (end) id
        self.n_static_blocks = hinge_br
        assert self.n_static_blocks <= self.config.max_blocks

        bl, br = tri_bl, hinge_br

        c.block_begin_index = bl
        c.block_end_index = br

        if n_tri > 0:
            wp_slice(self.block_indices, tri_bl, tri_br).assign(
                np.hstack([c.tri_mesh.triangles + pl, np.tile(-1, (n_tri, 1))])
            )
        if n_hinge > 0:
            wp_slice(self.block_indices, hinge_bl, hinge_br).assign(c.hinges + pl)
        if n_tri + n_hinge > 0:
            wp_slice(self.block_status, bl, br).assign(np.tile(1, br - bl))
            wp_slice(self.block_type, bl, br).assign(
                np.tile(np.array([FEM_BLOCK, FEM_TRI_BLOCK]), (br - bl, 1))
            )

        ######## Block Data: FEM Triangles ########

        if c.tri_mesh.n_triangles > 0:
            btl = self.n_triangles
            btr = btl + c.tri_mesh.n_triangles
            self.n_triangles = btr
            assert self.n_triangles <= self.config.max_blocks

            wp_slice(self.tri_block_ids, btl, btr).assign(np.arange(tri_bl, tri_br))
            wp_slice(self.rest_areas, btl, btr).assign(c.rest_areas)
            wp_slice(self.thicknesses, btl, btr).assign(np.tile(c.thickness, btr - btl))
            wp_slice(self.tri_inv_Dm, btl, btr).assign(c.inv_Dm)
            wp_slice(self.tri_materials, btl, btr).assign(
                np.tile(np.array([c.k_mu, c.k_lambda]), (btr - btl, 1))
            )

        ######## Block Data: Hinges ########

        if len(c.hinges) > 0:
            bhl = self.n_hinges
            bhr = bhl + len(c.hinges)
            self.n_hinges = bhr
            assert self.n_hinges <= self.config.max_blocks

            wp_slice(self.hinge_block_ids, bhl, bhr).assign(
                np.arange(hinge_bl, hinge_br)
            )
            wp_slice(self.hinge_rest_angles, bhl, bhr).assign(c.hinge_rest_angles)
            wp_slice(self.hinge_materials, bhl, bhr).assign(
                np.tile(np.array([c.k_hinge]), (bhr - bhl, 1))
            )

    def set_fem2d_positions(
        self,
        component: IPCFEM2DComponent,
        positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        begin = component.particle_begin_index
        end = component.particle_end_index

        if isinstance(positions, torch.Tensor) and positions.is_cuda:
            wp_wait_torch_stream(self.config.device)

        wp_slice(self.particle_q, begin, end).assign(
            convert_to_wp_array(positions, dtype=wp.vec3, device=self.config.device)
        )

    def set_fem2d_kinematic_target(
        self,
        component: IPCFEMComponent,
        indices: Union[np.ndarray, torch.Tensor, wp.array],
        positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        # TODO: assert component is registered

        begin = component.particle_begin_index
        end = component.particle_end_index
        assert 0 <= begin <= end <= self.n_particles

        if (isinstance(indices, torch.Tensor) and indices.is_cuda) or (
            isinstance(positions, torch.Tensor) and positions.is_cuda
        ):
            wp_wait_torch_stream(self.config.device)

        indices = convert_to_wp_array(
            indices, dtype=wp.int32, device=self.config.device
        )
        positions = convert_to_wp_array(
            positions, dtype=wp.vec3, device=self.config.device
        )

        wp.launch(
            set_to_ones_at_indices_int_kernel,
            dim=len(indices),
            inputs=[
                wp_slice(self.particle_dbc_tag, begin, end),
                indices,
            ],
            device=self.config.device,
        )
        wp.launch(
            set_values_at_indices_vec3_kernel,
            dim=len(indices),
            inputs=[
                wp_slice(self.particle_dbc_q, begin, end),
                indices,
                positions,
            ],
            device=self.config.device,
        )

    def register_abd_component(self, component: IPCABDComponent):
        c = component
        assert c.tet_mesh is not None or c.tri_mesh is not None

        ######## Component Data ########

        cid = self.n_components
        self.n_components += 1
        assert self.n_components <= self.config.max_components
        c.id_in_system = cid
        self.components.append(c)

        wp_slice(self.component_friction, cid, cid + 1).assign(np.array(c.friction))
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(0))
        wp_slice(self.component_group, cid, cid + 1).assign(
            np.array(c.group, dtype=np.int32)
        )

        ######## Scene Data ########

        if c.entity.scene not in self.scenes:
            sid = self.n_scenes
            self.n_scenes += 1
            assert self.n_scenes <= self.config.max_scenes
            self.scenes.append(c.entity.scene)
            self.n_surface_planes_list.append(0)
            self.n_surface_triangles_list.append(0)
            self.n_surface_edges_list.append(0)
            self.n_surface_particles_list.append(0)
        else:
            sid = self.scenes.index(c.entity.scene)

        ######## Block Data ########

        bl = self.n_static_blocks  # block left (begin) id
        br = bl + 1  # block right (end) id
        self.n_static_blocks = br
        assert self.n_static_blocks <= self.config.max_blocks

        c.block_begin_index = bl
        c.block_end_index = br

        pxl = self.n_particles
        pxr = pxl + 4
        self.n_particles = pxr
        assert self.n_particles <= self.config.max_particles

        wp_slice(self.block_indices, bl, br).assign(np.arange(pxl, pxr))
        wp_slice(self.block_status, bl, br).assign(np.tile(1, br - bl))
        wp_slice(self.block_type, bl, br).assign(np.tile(ABD_AFFINE_BLOCK, br - bl))

        ######## Affine Data ########

        al = self.n_affines
        ar = al + 1
        self.n_affines = ar
        assert self.n_affines <= self.config.max_blocks

        wp_slice(self.affine_block_ids, al, ar).assign(np.array(bl))
        wp_slice(self.affine_volumes, al, ar).assign(np.array(c.volume))
        wp_slice(self.affine_mass, al, ar).assign(c.abd_mass)

        wp_slice(self.component_affine, cid, cid + 1).assign(np.array(al))

        ######## Affine Proxy Data ########

        pose = c.entity.get_pose()
        transformation_matrix = pose.to_transformation_matrix()
        wp_slice(self.particle_q, pxl, pxr).assign(
            np.vstack(
                (
                    transformation_matrix[0, :3],
                    transformation_matrix[1, :3],
                    transformation_matrix[2, :3],
                    transformation_matrix[:3, 3],
                )
            )
        )
        wp_slice(self.particle_qd, pxl, pxr).assign(
            np.concatenate(
                (
                    np.zeros((3, 3)),
                    np.array(c.init_velocity).reshape(1, 3),
                ),
                axis=0,
            )
        )
        wp_slice(self.particle_q_rest, pxl, pxr).assign(
            wp_slice(self.particle_q, pxl, pxr)
        )
        wp_slice(self.particle_mass, pxl, pxr).assign(np.tile(c.vertex_mass.sum(), 4))
        wp_slice(self.particle_mask, pxl, pxr).assign(np.tile(1.0, 4))
        wp_slice(self.particle_affine, pxl, pxr).assign(np.tile(AFFINE_PROXY, 4))
        wp_slice(self.particle_component, pxl, pxr).assign(np.tile(cid, 4))
        wp_slice(self.particle_scene, pxl, pxr).assign(np.tile(sid, 4))
        wp_slice(self.particle_dbc_tag, pxl, pxr).assign(np.tile(0, 4))

        ######## Particle Data ########

        mesh = c.tet_mesh if c.tet_mesh is not None else c.tri_mesh

        pl = self.n_particles  # Particle Left (begin) id
        pr = pl + mesh.n_vertices  # Particle Right (end) id
        self.n_particles = pr
        assert self.n_particles <= self.config.max_particles

        particle_q_slice = wp_slice(self.particle_q, pl, pr)
        c.array_slice = particle_q_slice
        c.particle_begin_index = pl
        c.particle_end_index = pr
        c.cuda_pointer = particle_q_slice.ptr
        c.cuda_stream = self.config.device.stream.cuda_stream
        c.size = type_size_in_bytes(particle_q_slice.dtype) * particle_q_slice.size

        wp_slice(self.particle_q, pl, pr).assign(
            mesh.vertices @ transformation_matrix[:3, :3].T
            + transformation_matrix[:3, 3]
        )
        wp_slice(self.particle_qd, pl, pr).assign(
            np.tile(c.init_velocity, (pr - pl, 1))
        )
        wp_slice(self.particle_q_rest, pl, pr).assign(mesh.vertices)
        wp_slice(self.particle_mass, pl, pr).assign(c.vertex_mass)
        wp_slice(self.particle_mask, pl, pr).assign(np.tile(1.0, pr - pl))
        wp_slice(self.particle_affine, pl, pr).assign(np.tile(al, pr - pl))
        wp_slice(self.particle_component, pl, pr).assign(np.tile(cid, pr - pl))
        wp_slice(self.particle_scene, pl, pr).assign(np.tile(sid, pr - pl))
        wp_slice(self.particle_dbc_tag, pl, pr).assign(np.tile(0, pr - pl))

        ######## Surface Data ########

        if mesh.n_surface_vertices > 0:
            wp_slice(
                self.surface_particles[sid],
                self.n_surface_particles_list[sid],
                self.n_surface_particles_list[sid] + mesh.n_surface_vertices,
            ).assign(mesh.surface_vertices + pl)
            self.n_surface_particles_list[sid] += mesh.n_surface_vertices
            self.n_surface_particles_max = max(
                self.n_surface_particles_max, self.n_surface_particles_list[sid]
            )
            assert (
                self.n_surface_particles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_particles, sid, sid + 1).assign(
                np.array(self.n_surface_particles_list[sid], dtype=np.int32)
            )

        if mesh.n_surface_edges > 0:
            wp_slice(
                self.surface_edges[sid],
                self.n_surface_edges_list[sid],
                self.n_surface_edges_list[sid] + mesh.n_surface_edges,
            ).assign(mesh.surface_edges + pl)
            self.n_surface_edges_list[sid] += mesh.n_surface_edges
            self.n_surface_edges_max = max(
                self.n_surface_edges_max, self.n_surface_edges_list[sid]
            )
            assert (
                self.n_surface_edges_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_edges, sid, sid + 1).assign(
                np.array(self.n_surface_edges_list[sid], dtype=np.int32)
            )

        if mesh.n_surface_triangles > 0:
            wp_slice(
                self.surface_triangles[sid],
                self.n_surface_triangles_list[sid],
                self.n_surface_triangles_list[sid] + mesh.n_surface_triangles,
            ).assign(mesh.surface_triangles + pl)
            self.n_surface_triangles_list[sid] += mesh.n_surface_triangles
            self.n_surface_triangles_max = max(
                self.n_surface_triangles_max, self.n_surface_triangles_list[sid]
            )
            assert (
                self.n_surface_triangles_list[sid]
                <= self.config.max_surface_primitives_per_scene
            )
            wp_slice(self.n_surface_triangles, sid, sid + 1).assign(
                np.array(self.n_surface_triangles_list[sid], dtype=np.int32)
            )

    def unregister_abd_component(self, component: IPCABDComponent):
        cid = component.id_in_system
        self.components.remove(component)
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(1))

        bl = component.block_begin_index
        br = component.block_end_index
        wp_slice(self.block_status, bl, br).assign(np.tile(0, br - bl))

    def get_abd_proxy_positions(self, component: IPCABDComponent):
        begin = component.particle_begin_index
        assert begin - 4 >= 0 and begin <= self.n_particles

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_q)[begin - 4 : begin].clone()

    def get_abd_proxy_velocities(self, component: IPCABDComponent):
        begin = component.particle_begin_index
        assert begin - 4 >= 0 and begin <= self.n_particles

        torch_wait_wp_stream(self.config.device)
        return wp.to_torch(self.particle_qd)[begin - 4 : begin].clone()

    def set_abd_proxy_positions(
        self,
        component: IPCFEMComponent,
        proxy_positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        """
        mat: 4x3 matrix. [a1, a2, a3, p].
        Affine transformation is x -> [a1.x, a2.x, a3.x] + p
        """
        begin = component.particle_begin_index

        if isinstance(proxy_positions, torch.Tensor) and proxy_positions.is_cuda:
            wp_wait_torch_stream(self.config.device)

        proxy_positions = convert_to_wp_array(
            proxy_positions, dtype=wp.vec3, device=self.config.device
        )

        wp_slice(self.particle_q, begin - 4, begin).assign(proxy_positions)

        wp.launch(
            kernel=apply_parent_affine_kernel,
            dim=self.n_particles,
            inputs=[
                begin,
                self.particle_q,
                self.particle_q_rest,
                self.block_indices,
                self.affine_block_ids,
                self.particle_affine,
            ],
            device=self.config.device,
        )

    def set_abd_proxy_velocities(
        self,
        component: IPCFEMComponent,
        proxy_velocities: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        """
        v_mat: 4x3 matrix. [v_a1, v_a2, v_a3, v_p] = d/dt [a1, a2, a3, p].
        """
        begin = component.particle_begin_index

        if isinstance(proxy_velocities, torch.Tensor) and proxy_velocities.is_cuda:
            wp_wait_torch_stream(self.config.device)

        proxy_velocities = convert_to_wp_array(
            proxy_velocities, dtype=wp.vec3, device=self.config.device
        )

        wp_slice(self.particle_qd, begin - 4, begin).assign(proxy_velocities)

        wp.launch(
            kernel=apply_parent_affine_kernel,
            dim=self.n_particles,
            inputs=[
                begin,
                self.particle_qd,
                self.particle_q_rest,
                self.block_indices,
                self.affine_block_ids,
                self.particle_affine,
            ],
            device=self.config.device,
        )

    def set_abd_kinematic_target(
        self,
        component: IPCABDComponent,
        proxy_positions: Union[np.ndarray, torch.Tensor, wp.array],
    ):
        begin = component.particle_begin_index
        assert begin - 4 >= 0 and begin <= self.n_particles

        if isinstance(proxy_positions, torch.Tensor) and proxy_positions.is_cuda:
            wp_wait_torch_stream(self.config.device)

        proxy_positions = convert_to_wp_array(
            proxy_positions, dtype=wp.vec3, device=self.config.device
        )

        wp_slice(self.particle_dbc_tag, begin - 4, begin).fill_(1)
        wp_slice(self.particle_dbc_q, begin - 4, begin).assign(proxy_positions)

    def _add_constraint(self, constraint: IPCConstraint):
        con_id = self.n_constraints
        self.n_constraints += 1
        assert self.n_constraints <= self.config.max_constraints
        constraint.id_in_system = con_id

        wp_slice(self.constraint_lambda, con_id, con_id + 1).assign(np.array(0.0))
        wp_slice(self.constraint_param, con_id, con_id + 1).assign(
            constraint.param
            if constraint.param is not None
            else np.zeros(5, dtype=np.float32)
        )

        b_id = self.n_static_blocks
        self.n_static_blocks += 1
        assert self.n_static_blocks <= self.config.max_blocks

        wp_slice(self.constraint_block_ids, con_id, con_id + 1).assign(
            np.array([b_id], dtype=np.int32)
        )

        # get particle ids in system
        pids_sys = [-1, -1, -1, -1]
        assert len(constraint.particle_ids) <= 4
        assert len(constraint.particle_ids) == len(constraint.components)
        for i, (particle_id, component) in enumerate(
            zip(constraint.particle_ids, constraint.components)
        ):
            assert component in self.components

            pid_sys = particle_id + component.particle_begin_index
            assert pid_sys < component.particle_end_index

            pids_sys[i] = pid_sys

        wp_slice(self.block_indices, b_id, b_id + 1).assign(
            np.array([pids_sys], dtype=np.int32)
        )
        wp_slice(self.block_status, b_id, b_id + 1).assign(
            np.array([1], dtype=np.int32)
        )
        wp_slice(self.block_type, b_id, b_id + 1).assign(
            np.array([CONSTRAINT_BLOCK, constraint.type], dtype=np.int32)
        )

    def _remove_constraint(self, constraint: IPCConstraint):
        con_id = constraint.id_in_system
        b_id = self.constraint_block_ids.numpy()[con_id].item()

        wp_slice(self.block_status, b_id, b_id + 1).assign(
            np.array([0], dtype=np.int32)
        )

    def update_constraint_param(self, constraint: IPCConstraint):
        con_id = constraint.id_in_system
        wp_slice(self.constraint_param, con_id, con_id + 1).assign(
            constraint.param
            if constraint.param is not None
            else np.zeros(5, dtype=np.float32)
        )

    def register_joint_component(self, component: IPCABDJointComponent):
        c = component

        ######## Component Data ########

        cid = self.n_components
        self.n_components += 1
        assert self.n_components <= self.config.max_components
        c.id_in_system = cid
        self.components.append(c)

        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(0))

        ######## Scene Data ########

        if c.entity.scene not in self.scenes:
            sid = self.n_scenes
            self.n_scenes += 1
            assert self.n_scenes <= self.config.max_scenes
            self.scenes.append(c.entity.scene)
            self.n_surface_planes_list.append(0)
            self.n_surface_triangles_list.append(0)
            self.n_surface_edges_list.append(0)
            self.n_surface_particles_list.append(0)
        else:
            sid = self.scenes.index(c.entity.scene)

        ######## Particle Data ########

        pl = self.n_particles  # Particle Left (begin) id
        n_virtual_particles = len(c.virtual_particle_q_rest)
        pr = pl + n_virtual_particles
        self.n_particles = pr
        assert self.n_particles <= self.config.max_particles
        c.particle_begin_index = pl
        c.particle_end_index = pr

        wp_slice(self.particle_q_rest, pl, pr).assign(c.virtual_particle_q_rest)
        wp_slice(self.particle_mask, pl, pr).assign(np.tile(1.0, n_virtual_particles))
        wp_slice(self.particle_component, pl, pr).assign(
            np.tile(cid, n_virtual_particles)
        )
        wp_slice(self.particle_scene, pl, pr).assign(np.tile(sid, n_virtual_particles))
        wp_slice(self.particle_dbc_tag, pl, pr).assign(np.tile(0, n_virtual_particles))

        component_affine = self.component_affine.numpy()[
            np.array(
                [
                    abd_component.id_in_system
                    for abd_component in c.virtual_particle_abd_component
                ]
            )
        ]
        wp_slice(self.particle_affine, pl, pr).assign(component_affine)

        particle_q = self.particle_q.numpy()
        particle_qd = self.particle_qd.numpy()

        for i in range(n_virtual_particles):
            abd_component = c.virtual_particle_abd_component[i]
            assert abd_component in self.components

            pxl = abd_component.particle_begin_index - 4
            x_rest = c.virtual_particle_q_rest[i]
            x_transformed = particle_q[pxl : pxl + 3] @ x_rest + particle_q[pxl + 3]
            v_transformed = particle_qd[pxl : pxl + 3] @ x_rest + particle_qd[pxl + 3]
            wp_slice(self.particle_q, pl + i, pl + i + 1).assign(x_transformed)
            wp_slice(self.particle_qd, pl + i, pl + i + 1).assign(v_transformed)

        ######## Constraints ########

        for con in c.constraints:
            self._add_constraint(con)

    def unregister_joint_component(self, component: IPCABDJointComponent):
        cid = component.id_in_system
        self.components.remove(component)
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(1))

        for con in component.constraints:
            self._remove_constraint(con)

    def register_plane_component(self, component: IPCPlaneComponent):
        c = component

        ######## Component Data ########

        cid = self.n_components
        self.n_components += 1
        assert self.n_components <= self.config.max_components
        c.id_in_system = cid
        self.components.append(c)

        wp_slice(self.component_friction, cid, cid + 1).assign(np.array(c.friction))
        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(0))
        wp_slice(self.component_group, cid, cid + 1).assign(
            np.array(c.group, dtype=np.int32)
        )

        ######## Scene Data ########

        if c.entity.scene not in self.scenes:
            sid = self.n_scenes
            self.n_scenes += 1
            assert self.n_scenes <= self.config.max_scenes
            self.scenes.append(c.entity.scene)
            self.n_surface_planes_list.append(0)
            self.n_surface_triangles_list.append(0)
            self.n_surface_edges_list.append(0)
            self.n_surface_particles_list.append(0)
        else:
            sid = self.scenes.index(c.entity.scene)

        ######## Plane Data ########

        gid = self.n_planes
        self.n_planes += 1
        assert self.n_planes <= self.config.max_planes

        pose = c.entity.get_pose()
        transformation_matrix = pose.to_transformation_matrix()
        new_normal = transformation_matrix[:3, :3] @ c.normal
        new_offset = c.offset + np.dot(new_normal, transformation_matrix[:3, 3])

        wp_slice(self.plane_normals, gid, gid + 1).assign(new_normal)
        wp_slice(self.plane_offsets, gid, gid + 1).assign(np.array(new_offset))
        wp_slice(self.plane_component, gid, gid + 1).assign(np.array(cid))
        wp_slice(self.plane_scene, gid, gid + 1).assign(np.array(sid))

        ######## Surface Data ########

        wp_slice(
            self.surface_planes[sid],
            self.n_surface_planes_list[sid],
            self.n_surface_planes_list[sid] + 1,
        ).assign(np.array(gid))
        self.n_surface_planes_list[sid] += 1
        self.n_surface_planes_max = max(
            self.n_surface_planes_max, self.n_surface_planes_list[sid]
        )
        assert (
            self.n_surface_planes_list[sid]
            <= self.config.max_surface_primitives_per_scene
        )
        wp_slice(self.n_surface_planes, sid, sid + 1).assign(
            np.array(self.n_surface_planes_list[sid], dtype=np.int32)
        )

    def unregister_plane_component(self, component: IPCPlaneComponent):
        cid = component.id_in_system
        self.components.remove(component)

        wp_slice(self.component_removed, cid, cid + 1).assign(np.array(1))

    def rebuild(self):
        """Rebuild the system with the current components, ignoring all unregistered (removed) components. This makes the memory contigous again, and improves performance."""
        components_to_register = self.components.copy()
        q_mem = []
        qd_mem = []

        for c in components_to_register:
            if isinstance(c, IPCFEMComponent):
                q_mem.append(c.get_positions())
                qd_mem.append(c.get_velocities())
            elif isinstance(c, IPCABDComponent):
                q_mem.append(c.get_proxy_positions())
                qd_mem.append(c.get_proxy_velocities())
            elif isinstance(c, IPCPlaneComponent):
                q_mem.append(None)
                qd_mem.append(None)
            else:
                raise NotImplementedError

        self.components.clear()
        self.scenes.clear()

        # For now, just clear all counts and re-register all components
        self.n_components = 0
        self.n_scenes = 0
        self.n_planes = 0
        self.n_particles = 0  # FEM vertices + ABD affine proxies + ABD vertices
        self.n_tets = 0  # FEM tets
        self.n_triangles = 0
        self.n_hinges = 0
        self.n_affines = 0  # ABD affine bodies
        self.n_static_blocks = 0  # FEM tets + ABD affine bodies
        self.n_blocks_this_step = 0  # static blocks + contact blocks
        self.n_constraints = 0  # constraints
        self.n_surface_triangles_list.clear()  # surface triangles
        self.n_surface_edges_list.clear()  # surface edges
        self.n_surface_particles_list.clear()  # surface vertices
        self.n_surface_planes_list.clear()  # surface planes
        self.n_surface_triangles_max = 0
        self.n_surface_edges_max = 0
        self.n_surface_particles_max = 0
        self.n_surface_planes_max = 0

        self.step_count = 0
        self.time = 0.0  # increment by config.time_step every step

        for c, q, qd in zip(components_to_register, q_mem, qd_mem):
            if isinstance(c, IPCFEMComponent):
                self.register_fem_component(c)
                c.set_positions(q)
                c.set_velocities(qd)
            elif isinstance(c, IPCABDComponent):
                self.register_abd_component(c)
                c.set_proxy_positions(q)
                c.set_proxy_velocities(qd)
            elif isinstance(c, IPCPlaneComponent):
                self.register_plane_component(c)
            else:
                raise NotImplementedError

    def _zero_energy(self):
        wp_slice(self.energy, 0, self.n_scenes).zero_()

    def _zero_diff(self):
        wp_slice(self.particle_grad, 0, self.n_particles).zero_()
        wp_slice(self.blocks, 0, self.n_blocks_this_step).zero_()
        wp_slice(self.particle_collision_force, 0, self.n_particles).zero_()
        wp_slice(self.particle_friction_force, 0, self.n_particles).zero_()

    def _process_dbc(self):
        wp.launch(
            kernel=masked_diff_kernel,
            dim=self.n_particles,
            inputs=[
                self.particle_dbc_q,
                self.particle_q,
                self.particle_dbc_tag,
            ],
            outputs=[self.particle_dbc_delta_q],
            device=self.config.device,
        )
        
        if not self.config.use_constraint_dbc_only:
            self.filter.filter(self.particle_q)
            self.ccd.compute_step(self.particle_q, self.particle_dbc_delta_q)
        else:
            wp_slice(self.ccd_step, 0, self.n_scenes).zero_()
            # self.ccd_step.zero_()

        wp.launch(
            kernel=process_dbc_kernel,
            dim=self.n_particles,
            inputs=[
                self.ccd_step,
                self.particle_scene,
                self.block_indices,
                self.affine_block_ids,
                self.particle_affine,
                self.particle_dbc_delta_q,
                self.particle_q,
                self.particle_dbc_tag,
                self.particle_dbc_mask,
                self.particle_mask,
            ],
            device=self.config.device,
        )

        # ipc_logger.debug(f"system.ccd_step: {self.ccd_step.numpy()[0]}")
        # ipc_logger.debug(f"system.particle_mask: {self.particle_mask.numpy()[:self.n_particles]}")
        # ipc_logger.debug(f"system.particle_dbc_mask: {self.particle_dbc_mask.numpy()[:self.n_particles]}")

    def _constraint_reset(self):
        wp_slice(self.constraint_lambda, 0, self.n_constraints).zero_()
        self.constraint_mu = self.config.constraint_mu_init

    def _constraint_update(self):
        if self.config.use_augmented_lagrangian:
            wp.launch(
                kernel=constraint_update_lambda_kernel,
                dim=self.n_constraints,
                inputs=[
                    self.particle_q,
                    self.particle_q_prev_step,
                    self.particle_scene,
                    self.particle_mask,
                    self.constraint_param,
                    self.constraint_lambda,
                    self.constraint_block_ids,
                    self.block_status,
                    self.block_type,
                    self.particle_q_rest,
                    self.block_indices,
                    self.affine_block_ids,
                    self.particle_affine,
                    self.constraint_mu,
                ],
                device=self.config.device,
            )
        self.constraint_mu = min(self.constraint_mu * self.config.constraint_mu_scale, self.config.constraint_mu_max)

    def _preprocess_energy(self, x_prev):
        self.filter.filter(x_prev)
        # Generate contact_type, contact_d, contact_c, contact_eps_cross, block_status...
        # ... for friction_preprocess_kernel to use
        if "friction" in self.energy_calcs:
            assert "collision" in self.energy_calcs
            self.energy_calcs["collision"].compute_energy(x_prev)
            self.energy_calcs["friction"].preprocess(x_prev)

    def _compute_energy(self, x, output_debug_info=False):
        self._zero_energy()
        last_energy = np.zeros(self.n_scenes, dtype=np.float32)
        debug_str = ""
        for name, energy in self.energy_calcs.items():
            energy.compute_energy(x)
            if output_debug_info:
                new_energy = self.energy.numpy()[: self.n_scenes]
                delta_energy = new_energy - last_energy
                last_energy = new_energy
                debug_str += f"{name}={delta_energy} "
        if output_debug_info:
            ipc_logger.debug(f"energy={last_energy} ( {debug_str})")

    def _compute_diff(self, x, grad_coeff):
        if self.n_blocks_this_step < self.n_static_blocks:
            raise RuntimeError(
                "n_blocks_this_step < n_static_blocks. Did you forget to call system.filter.filter()?"
            )
        self._zero_diff()
        for name, energy in self.energy_calcs.items():
            energy.compute_diff(x, grad_coeff)

            if self.config.debug:
                # min_distance = 0
                # if self.n_blocks_this_step > 0:
                #     min_distance = np.min(self.contact_d.numpy()[self.n_static_blocks: self.n_blocks_this_step])

                # if name == "collision":
                #     dd_dx = self.contact_dd_dx.numpy()[
                #         self.n_static_blocks : self.n_blocks_this_step
                #     ]
                #     contact_type = self.block_type.numpy()[
                #         self.n_static_blocks : self.n_blocks_this_step
                #     ]
                #     if np.isnan(dd_dx).any():
                #         print(
                #             f"buggy contact type: {contact_type[np.isnan(dd_dx).any(axis=-1).any(axis=-1)]}"
                #         )
                #         nan_idx = np.argwhere(
                #             np.isnan(dd_dx).any(axis=-1).any(axis=-1)
                #         ).reshape(-1)
                #         contact_ids = self.block_indices.numpy()[
                #             self.n_static_blocks : self.n_blocks_this_step
                #         ]
                #         x_np = x.numpy()
                #         for i in nan_idx:
                #             i0, i1, i2, i3 = contact_ids[i]
                #             x0, x1, x2, x3 = x_np[i0], x_np[i1], x_np[i2], x_np[i3]
                #             e0 = x1 - x0
                #             e1 = x3 - x2
                #             print(
                #                 f"buggy contact: e0 = {e0}, e1 = {e1}, len0 = {np.linalg.norm(e0)}, len1 = {np.linalg.norm(e1)}, sin = {np.linalg.norm(np.cross(e0, e1)) / (np.linalg.norm(e0) * np.linalg.norm(e1))}"
                #             )
                #             print(f"edges_np = {repr(np.array([x0, x1, x2, x3]))})")

                assert not np.isnan(
                    self.particle_grad.numpy()[: self.n_particles]
                ).any(), name

        if self.config.spd_project_max > 0:
            wp.launch(
                kernel=block_spd_project_kernel,
                dim=self.n_blocks_this_step,
                inputs=[
                    self.blocks,
                    self.block_type,
                    self.block_status,
                    self.config.spd_project_max,
                ],
                device=self.config.device,
            )

    def _clip_velocity(self):
        r = self.config.v_max * self.config.time_step
        # print(f"max velocity before clipping: {np.max(np.linalg.norm(self.particle_p.numpy()[self.particle_affine.numpy() == -1], axis=1) / self.config.time_step)}")
        wp.launch(
            kernel=clip_velocity_kernel,
            dim=self.n_particles,
            inputs=[
                self.particle_affine,
                self.particle_p,
                self.particle_q_prev_it,
                self.particle_q_prev_step,
                r,
            ],
            device=self.config.device,
        )

    def _line_search(self):
        """
        Line search: find E(self.particle_q_prev_it + alpha * self.particle_p)
        < E(self.particle_q_prev_it).
        Starting from self.ccd_step, halve alpha until the energy decreases.
        """

        wp.copy(self.energy_prev, self.energy, count=self.n_scenes)
        for n_halves in range(0, self.config.line_search_max_iters):
            wp.launch(
                kernel=line_search_iteration_kernel,
                dim=self.n_particles,
                inputs=[
                    self.particle_q_prev_it,
                    self.particle_p,
                    self.ccd_step,
                    float(n_halves),
                    self.particle_scene,
                    self.energy_prev,
                    self.energy,
                ],
                outputs=[self.particle_q],
                device=self.config.device,
            )
            self._compute_energy(
                self.particle_q,
                output_debug_info=self.config.debug
                and not self.config.use_graph
                and n_halves == self.config.line_search_max_iters - 1,
            )

    def _get_dense_matrix_brute_force(self):
        dense_matrix = np.zeros((3 * self.n_particles, 3 * self.n_particles))
        for j in range(3 * self.n_particles):
            one_hot_np = np.zeros(3 * self.n_particles)
            one_hot_np[j] = 1.0
            one_hot_wp = wp.array(
                one_hot_np.reshape(-1, 3), dtype=wp.vec3, device=self.config.device
            )
            prod = wp.zeros_like(one_hot_wp)
            wp.launch(
                kernel=hess_block_mul_dx_kernel,
                dim=max(self.config.max_blocks, self.n_particles),
                inputs=[
                    self.n_particles,
                    self.n_static_blocks,
                    self.contact_counter,
                    self.particle_mass,
                    self.particle_mask,
                    self.particle_dbc_mask,
                    self.config.kappa_con,
                    self.block_indices,
                    self.affine_block_ids,
                    self.particle_affine,
                    self.blocks,
                    self.block_status,
                    one_hot_wp,
                    self.particle_q_rest,
                ],
                outputs=[prod],
                device=self.config.device,
            )
            prod_np = prod.numpy().reshape(-1)
            dense_matrix[:, j] = prod_np
        return dense_matrix
