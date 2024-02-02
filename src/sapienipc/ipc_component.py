from __future__ import annotations
from typing import Union, List, Tuple

import time

import numpy as np
import torch
from sapienipc.ipc_utils.global_defs import *
import warp as wp
import sapien

from .ipc_utils.ipc_mesh import IPCTetMesh, IPCTriMesh
from .ipc_utils.logging_utils import ipc_logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ipc_system import IPCSystem, IPCSystemConfig


class IPCPlaneComponent(sapien.Component):
    def __init__(self):
        super().__init__()

        self.id_in_system = None

        # For now don't support changing plane normal or offset
        self.normal = np.array([1.0, 0.0, 0.0])
        self.offset = 0.0
        self.friction = 0.0

        self.group = -1  # -1 means no group

    def on_add_to_scene(self, scene):
        scene.get_system("ipc").register_plane_component(self)

    def on_remove_from_scene(self, scene):
        scene.get_system("ipc").unregister_plane_component(self)

    def on_set_pose(self, pose):
        assert self.entity is None or self.entity.scene is None

    def set_group(self, group: int):
        assert self.entity is None or self.entity.scene is None
        self.group = group
        # If two components have the same group (>= 0), their collision will be ignored

    def set_pose(self, pose):
        assert self.entity is not None and self.entity.scene is None
        self.entity.set_pose(pose)

    def set_friction(self, friction):
        assert self.entity is None or self.entity.scene is None
        self.friction = friction


class IPCBaseComponent(sapien.Component):
    def __init__(self):
        sapien.Component.__init__(self)

        self.id_in_system = None
        self.array_slice = None
        self.particle_begin_index = None
        self.particle_end_index = None
        self.block_begin_index = None
        self.block_end_index = None
        self.cuda_pointer = None
        self.cuda_stream = None
        self.size = None

        self.density = None
        self.friction = 0.0
        self.init_velocity = [0.0, 0.0, 0.0]

        self.vertex_mass = None

        self.group = -1  # -1 means no group

    def get_cuda_pointer(self):
        assert self.cuda_pointer is not None
        return self.cuda_pointer

    def get_cuda_stream(self):
        assert self.cuda_stream is not None
        return self.cuda_stream

    def get_size(self):
        assert self.size is not None
        return self.size

    def on_add_to_scene(self, scene):
        """called right after entity is added to scene or component is added to entity in scene"""
        raise NotImplementedError()

    def on_remove_from_scene(self, scene):
        """called right before entity is removed from scene or component is removed from entity in scene"""
        raise NotImplementedError()

    def on_set_pose(self, pose):
        assert self.entity is None or self.entity.scene is None
        pass

    def set_group(self, group: int):
        assert self.entity is None or self.entity.scene is None
        self.group = group
        # If two components have the same group (>= 0), their collision will be ignored

    def set_friction(self, friction):
        assert self.entity is None or self.entity.scene is None
        self.friction = friction

    def set_init_velocity(self, velocity):
        assert self.entity is None or self.entity.scene is None
        self.init_velocity = velocity

    def get_positions(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_vertex_positions(self)

    def get_velocities(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_vertex_velocities(self)

    def get_collision_forces(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_vertex_collision_forces(self)

    def get_friction_forces(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_vertex_friction_forces(self)

    def get_state_dict(self) -> dict:
        """Debugging utility function. May be changed in the future."""
        return {
            "density": self.density,
            "friction": self.friction,
            "init_velocity": self.init_velocity,
            "vertex_mass": self.vertex_mass,
        }

    def load_state_dict(self, state_dict):
        """Debugging utility function. May be changed in the future."""
        self.density = state_dict["density"]
        self.friction = state_dict["friction"]
        self.init_velocity = state_dict["init_velocity"]
        self.vertex_mass = state_dict["vertex_mass"]


class IPCABDComponent(IPCBaseComponent):
    def __init__(self):
        super().__init__()

        self.volume = 0.0

        self.tet_mesh: IPCTetMesh = None
        self.tri_mesh: IPCTriMesh = None

        self.vertex_mass = None
        self.abd_mass = np.zeros((4, 4, 3, 3), dtype=np.float32)

    def set_tet_mesh(self, tet_mesh: IPCTetMesh):
        assert self.entity is None or self.entity.scene is None  # not in scene
        assert self.tri_mesh is None, "Cannot set both tet_mesh and tri_mesh"
        self.tet_mesh = tet_mesh

    def set_tri_mesh(self, tri_mesh: IPCTriMesh):
        assert self.entity is None or self.entity.scene is None
        assert self.tet_mesh is None, "Cannot set both tet_mesh and tri_mesh"
        self.tri_mesh = tri_mesh

    def set_density(self, density: float):
        assert self.entity is None or self.entity.scene is None
        self.density = density

    def _process_tet_mesh(self):
        """Compute: vertex_mass, volume"""
        assert self.tet_mesh is not None
        assert self.density is not None

        self.volume = 0.0
        self.vertex_mass = np.zeros(self.tet_mesh.n_vertices)

        for b, (i, j, k, l) in enumerate(self.tet_mesh.tets):
            p = self.tet_mesh.vertices[i]
            q = self.tet_mesh.vertices[j]
            r = self.tet_mesh.vertices[k]
            s = self.tet_mesh.vertices[l]

            qp = q - p
            rp = r - p
            sp = s - p

            Dm = np.array((qp, rp, sp)).T
            volume = np.linalg.det(Dm) / 6.0
            tet_mass = self.density * volume

            self.volume += volume

            if volume <= 0.0:
                raise ValueError("Tet mesh has inverted tets")
            else:
                self.vertex_mass[np.array([i, j, k, l])] += tet_mass / 4.0

    def _process_tri_mesh(self):
        assert self.tri_mesh is not None
        assert self.density is not None

        self.volume = 0.0
        self.vertex_mass = np.zeros(self.tri_mesh.n_vertices)
        self.abd_mass.fill(0.0)

        """ 
        abd_intertia = sum_{tet} int_{x in tet} rho J^T J dV
            where J^T J =
            .......                  |
            ..x^Tx.                  x
            .......                  |
                    .......             |
                    ..x^Tx.             x
                    .......             |
                            .......        |
                            ..x^Tx.        x
                            .......        |
            ---x---                  1
                    ---x---             1
                            ---x---        1

        """

        p = self.tri_mesh.vertices.mean(axis=0)

        for i, j, k in self.tri_mesh.triangles:
            a = self.tri_mesh.vertices[i]
            b = self.tri_mesh.vertices[j]
            c = self.tri_mesh.vertices[k]

            pa = a - p
            pb = b - p
            pc = c - p

            t = np.array((p, a, b, c))
            x_col = t[:, 0]
            y_col = t[:, 1]
            z_col = t[:, 2]

            vol = np.linalg.det(np.array((pa, pb, pc))) / 6.0
            self.volume += vol

            center_of_mass = (a + b + c + p) / 4.0

            I_xy = np.outer(x_col, y_col).sum() + np.dot(x_col, y_col)
            I_xz = np.outer(x_col, z_col).sum() + np.dot(x_col, z_col)
            I_yz = np.outer(y_col, z_col).sum() + np.dot(y_col, z_col)
            I_xx = np.outer(x_col, x_col).sum() + np.dot(x_col, x_col)
            I_yy = np.outer(y_col, y_col).sum() + np.dot(y_col, y_col)
            I_zz = np.outer(z_col, z_col).sum() + np.dot(z_col, z_col)

            I_xTx = (
                np.array(
                    [[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]],
                    dtype=np.float32,
                )
                * self.density
                * vol
                / 20.0
            )

            self.abd_mass[0, 0] += I_xTx
            self.abd_mass[1, 1] += I_xTx
            self.abd_mass[2, 2] += I_xTx

            self.abd_mass[3, 3] += self.density * vol * np.eye(3, dtype=np.float32)

            m_CoM = self.density * vol * center_of_mass
            self.abd_mass[0, 3, :, 0] += m_CoM
            self.abd_mass[1, 3, :, 1] += m_CoM
            self.abd_mass[2, 3, :, 2] += m_CoM
            self.abd_mass[3, 0, 0, :] += m_CoM
            self.abd_mass[3, 1, 1, :] += m_CoM
            self.abd_mass[3, 2, 2, :] += m_CoM

    def on_add_to_scene(self, scene: sapien.Scene):
        if self.tet_mesh is not None:
            self._process_tet_mesh()
        elif self.tri_mesh is not None:
            self._process_tri_mesh()
        else:
            raise ValueError("Must set tet_mesh or tri_mesh")

        scene.get_system("ipc").register_abd_component(self)

        ipc_logger.debug(
            f"abd  \"{self.entity.name}\" on_add_to_scene (pose: sapien.Pose(p=[{', '.join(map(str, self.entity.pose.p))}], q=[{', '.join(map(str, self.entity.pose.q))}]))"
        )
        # print(self.get_positions())

    def on_remove_from_scene(self, scene: sapien.Scene):
        scene.get_system("ipc").unregister_abd_component(self)

    def set_kinematic_target(
        self, proxy_positions: Union[np.ndarray, torch.Tensor, wp.array]
    ):
        """
        proxy_positions: shape = (4, 3)
        the transformation matrix of the ABD body is T = [A, t; 0, 1]
            where A = proxy_positions[:3, :], t = proxy_positions[3, :]
        """
        assert self.entity is not None and self.entity.scene is not None

        assert proxy_positions.shape == (4, 3)

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_abd_kinematic_target(self, proxy_positions)

    def set_kinematic_target_transformation_matrix(
        self, T: Union[np.ndarray, torch.Tensor]
    ):
        assert self.entity is not None and self.entity.scene is not None

        if isinstance(T, torch.Tensor):
            proxy_positions = torch.concat((T[:3, :3], T[:3, 3].unsqueeze(0)), dim=0)
        elif isinstance(T, np.ndarray):
            proxy_positions = np.concatenate((T[:3, :3], T[None, :3, 3]), axis=0)

        self.set_kinematic_target(proxy_positions)

    def set_kinematic_target_pose(self, pose: sapien.Pose):
        """
        For CUDA simulation, this is slower than
        set_kinematic_target(torch.Tensor) because of the
        HtoD copy.
        """
        assert self.entity is not None and self.entity.scene is not None

        self.set_kinematic_target_transformation_matrix(pose.to_transformation_matrix())

    def get_proxy_positions(self) -> torch.Tensor:
        """
        proxy_positions: shape = (4, 3)
        the transformation matrix of the ABD body is T = [A, t; 0, 1]
            where A = proxy_positions[:3, :], t = proxy_positions[3, :]
        """
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_abd_proxy_positions(self)

    def get_transformation_matrix(self) -> torch.Tensor:
        proxy_positions = self.get_proxy_positions()
        transformation_matrix = torch.zeros(
            (4, 4), dtype=torch.float32, device=proxy_positions.device
        )
        transformation_matrix[3, 3] = 1.0
        transformation_matrix[:3, :3] = proxy_positions[:3, :]
        transformation_matrix[:3, 3] = proxy_positions[3, :]
        return transformation_matrix

    def get_proxy_velocities(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_abd_proxy_velocities(self)

    def set_proxy_positions(
        self, proxy_positions: Union[np.ndarray, torch.Tensor, wp.array]
    ):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_abd_proxy_positions(self, proxy_positions)

    def set_proxy_velocities(
        self, proxy_velocities: Union[np.ndarray, torch.Tensor, wp.array]
    ):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_abd_proxy_velocities(self, proxy_velocities)

    def set_transformation_matrix(self, T: Union[np.ndarray, torch.Tensor]):
        assert self.entity is not None and self.entity.scene is not None

        if isinstance(T, torch.Tensor):
            proxy_positions = torch.concat((T[:3, :3], T[:3, 3].unsqueeze(0)), dim=0)
        elif isinstance(T, np.ndarray):
            proxy_positions = np.concatenate((T[:3, :3], T[None, :3, 3]), axis=0)
        self.set_proxy_positions(proxy_positions)

    def set_velocities(
        self,
        linear_velocity: Union[list, np.ndarray, torch.Tensor],
        angular_velocity: Union[list, np.ndarray, torch.Tensor],
    ):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        proxy_posistions = self.get_proxy_positions()

        # convert to torch tensor
        if isinstance(linear_velocity, list):
            linear_velocity = torch.tensor(linear_velocity, dtype=torch.float32).to(
                proxy_posistions.device
            )
        elif isinstance(linear_velocity, np.ndarray):
            linear_velocity = (
                torch.from_numpy(linear_velocity).float().to(proxy_posistions.device)
            )

        skew_symmetric_matrix = torch.tensor(
            [
                [0, -angular_velocity[2], angular_velocity[1]],
                [angular_velocity[2], 0, -angular_velocity[0]],
                [-angular_velocity[1], angular_velocity[0], 0],
            ],
            dtype=torch.float32,
        ).to(proxy_posistions.device)

        proxy_velocities = torch.zeros((4, 3), dtype=torch.float32).to(
            proxy_posistions.device
        )
        proxy_velocities[3, :] = linear_velocity
        proxy_velocities[:3, :] = skew_symmetric_matrix @ proxy_posistions[:3, :]

        self.set_proxy_velocities(proxy_velocities)

    def set_pose_in_system(self, pose: sapien.Pose):
        """
        For CUDA simulation, this is slower than
        set_proxy_positions(torch.Tensor) because of the
        HtoD copy.
        """
        assert self.entity is not None and self.entity.scene is not None

        self.set_transformation_matrix(pose.to_transformation_matrix())

    def get_state_dict(self) -> dict:
        """Debugging utility function. May be changed in the future."""
        d = super(IPCABDComponent, self).get_state_dict()
        d.update(
            {
                "tri_mesh": self.tri_mesh,
                "tet_mesh": self.tet_mesh,
                "volume": self.volume,
                "affine_matrix": self.get_proxy_positions(),
                "affine_velocity": self.get_proxy_velocities(),
            }
        )
        return d

    def load_state_dict(self, state_dict):
        """Debugging utility function. May be changed in the future."""
        super(IPCABDComponent, self).load_state_dict(state_dict)
        self.tri_mesh = state_dict["tri_mesh"]
        self.tet_mesh = state_dict["tet_mesh"]
        self.volume = state_dict["volume"]

        if self.entity is not None and self.entity.scene is not None:
            # this overwrites the entity's pose and init_velocity
            system: IPCSystem = self.entity.scene.get_system("ipc")
            system.set_abd_proxy_positions(self, state_dict["affine_matrix"])
            system.set_abd_proxy_velocities(self, state_dict["affine_velocity"])


class IPCFEMComponent(IPCBaseComponent):
    def __init__(self):
        super().__init__()

        # Provided by user
        self.tet_mesh: IPCTetMesh = None
        # self.initial_vertices = None
        self.density = None
        self.k_mu = None
        self.k_lambda = None

        # Computed when added to scene
        self.rest_volumes = None  # [N_tets]
        self.inv_Dm = None  # [N_tets, 3, 3]
        self.vertex_mass = None  # [N_vertices]
        # the inverse of its "reference shape matrix" Dm (Page 28, FEM tutorial)

    def set_tet_mesh(self, tet_mesh: IPCTetMesh):
        assert self.entity is None or self.entity.scene is None  # not in scene
        self.tet_mesh = tet_mesh

    def set_material(self, density: float, young: float, poisson: float):
        """Set material properties of the FEM component

        Args:
            density (float): density in kg/m^3
            young (float): Young's modulus in Pa
            poisson (float): Poisson's ratio
        """
        assert self.entity is None or self.entity.scene is None
        self.density = density
        self.k_mu = young / (2 * (1 + poisson))
        self.k_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))

    def _process_tet_mesh(self, default_vertex_mass=1e2):
        """Compute: vertex_mass, inv_Dm"""
        assert self.tet_mesh is not None
        assert (_ is not None for _ in [self.density, self.k_mu, self.k_lambda])

        self.rest_volumes = np.zeros(self.tet_mesh.n_tets)
        self.inv_Dm = np.zeros((self.tet_mesh.n_tets, 3, 3))
        self.vertex_mass = np.zeros(self.tet_mesh.n_vertices)

        for b, (i, j, k, l) in enumerate(self.tet_mesh.tets):
            p = self.tet_mesh.vertices[i]
            q = self.tet_mesh.vertices[j]
            r = self.tet_mesh.vertices[k]
            s = self.tet_mesh.vertices[l]

            qp = q - p
            rp = r - p
            sp = s - p

            Dm = np.array((qp, rp, sp)).T
            volume = np.linalg.det(Dm) / 6.0
            tet_mass = self.density * volume

            if volume <= 0.0:
                raise ValueError("Tet mesh has inverted tets")
            else:
                self.inv_Dm[b] = np.linalg.inv(Dm)
                self.rest_volumes[b] = volume
                self.vertex_mass[np.array([i, j, k, l])] += tet_mass / 4.0

        self.vertex_mass[self.vertex_mass == 0.0] = default_vertex_mass

    def on_add_to_scene(self, scene: sapien.Scene):
        # compute vertex mass & inv_Dm
        self._process_tet_mesh()

        # allocate id for tet mesh vertices (in register?)
        scene.get_system("ipc").register_fem_component(self)

    def on_remove_from_scene(self, scene: sapien.Scene):
        scene.get_system("ipc").unregister_fem_component(self)

    def on_set_pose(self, pose: sapien.Pose):
        # assert self.entity is None or self.entity.scene is None
        pass

    # def set_pose_in_system(self, pose: sapien.Pose):
    #     assert self.entity is not None and self.entity.scene is not None

    #     system = self.entity.scene.get_system("ipc")
    #     indices = np.arange(0, self.tet_mesh.n_vertices, dtype=np.int32)
    #     pose_mat = pose.to_transformation_matrix()
    #     new_positions = self.tet_mesh.vertices @ pose_mat[:3, :3].T + pose_mat[:3, 3]
    #     system.set_fem_vertex_positions(self, indices, new_positions)

    def set_kinematic_target(self, indices, positions):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_fem_kinematic_target(self, indices, positions)

    def set_positions(self, positions):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_fem_positions(self, positions)

    def set_velocities(self, velocities):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_fem_velocities(self, velocities)

    def select_box(self, p_min, p_max) -> np.ndarray:
        """Utility function
        Select the vertices that is within the 3d box (p_min, p_max)
        Returns an array of vertex indices
        """
        assert self.tet_mesh is not None

        p_min = np.asarray(p_min)
        p_max = np.asarray(p_max)

        indices = np.where(
            np.all(
                np.logical_and(
                    p_min <= self.tet_mesh.vertices, self.tet_mesh.vertices <= p_max
                ),
                axis=1,
            )
        )[0]

        return indices.astype(np.int32)

    def get_state_dict(self) -> dict:
        """Debugging utility function. May be changed in the future."""
        d = super(IPCFEMComponent, self).get_state_dict()
        d.update(
            {
                "tet_mesh": self.tet_mesh,
                "k_mu": self.k_mu,
                "k_lambda": self.k_lambda,
                "rest_volumes": self.rest_volumes,
                "inv_Dm": self.inv_Dm,
                "positions": self.get_positions(),
                "velocities": self.get_velocities(),
            }
        )
        return d

    def load_state_dict(self, state_dict):
        """Debugging utility function. May be changed in the future."""
        super(IPCABDComponent, self).load_state_dict(state_dict)
        self.tet_mesh = state_dict["tet_mesh"]
        self.k_mu = state_dict["k_mu"]
        self.k_lambda = state_dict["k_lambda"]
        self.rest_volumes = state_dict["rest_volumes"]
        self.inv_Dm = state_dict["inv_Dm"]

        if self.entity is not None and self.entity.scene is not None:
            # this overwrites the entity's pose and init_velocity
            system: IPCSystem = self.entity.scene.get_system("ipc")
            system.set_fem_positions(self, state_dict["positions"])
            system.set_fem_velocities(self, state_dict["velocities"])


class IPCFEM2DComponent(IPCBaseComponent):
    def __init__(self):
        super().__init__()

        # Provided by user
        self.tri_mesh: IPCTriMesh = None
        self.density = None
        self.thickness = None
        self.k_mu = None
        self.k_lambda = None
        self.k_hinge = None

        self.vertex_mass = None  # [N_vertices]
        self.rest_areas = None  # [N_triangles]
        self.inv_Dm = None  # [N_triangles, 2, 2]
        self.hinges = None  # [N_non_boundary_edges, 4]
        self.hinge_rest_angles = None  # [N_non_boundary_edges]

    def set_tri_mesh(self, tri_mesh: IPCTriMesh):
        assert self.entity is None or self.entity.scene is None
        self.tri_mesh = tri_mesh

    def set_material(
        self, density: float, young: float, poisson: float, hinge_stiffness: float
    ):
        """Set material properties of the FEM component

        Args:
            density (float): density in kg/m^3
            young (float): Young's modulus in Pa
            poisson (float): Poisson's ratio
            hinge_stiffness (float): stiffness of elastic hinges
                E_hinge = k_hinge * (theta - theta_rest)^2 * ||e_rest||
                where h_e_rest is a third of the average of the heights
                of the two triangles incident to the edge e
        """
        assert self.entity is None or self.entity.scene is None
        self.density = density
        self.k_mu = young / (2 * (1 + poisson))
        self.k_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        self.k_hinge = hinge_stiffness

    def set_thickness(self, thickness: float):
        assert self.entity is None or self.entity.scene is None
        self.thickness = thickness

    def _process_tri_mesh(self, default_vertex_mass=1.0):
        assert self.tri_mesh is not None
        assert self.density is not None
        assert self.thickness is not None

        self.rest_areas = np.zeros(self.tri_mesh.n_triangles, dtype=np.float32)
        self.vertex_mass = np.zeros(self.tri_mesh.n_vertices, dtype=np.float32)
        self.inv_Dm = np.zeros((self.tri_mesh.n_triangles, 2, 2), dtype=np.float32)

        hinge_dict = {}  # (x1, x2) (the edge) -> x0, (x0, x1, x2) is a triangle
        self.hinges = []
        self.hinge_rest_angles = []

        # start_time = time.time()

        # for b, (i0, i1, i2) in enumerate(self.tri_mesh.triangles):
        #     x0 = self.tri_mesh.vertices[i0]
        #     x1 = self.tri_mesh.vertices[i1]
        #     x2 = self.tri_mesh.vertices[i2]

        #     e1 = x1 - x0
        #     e2 = x2 - x0
        #     basis1 = wp.normalize(e1)
        #     basis2 = wp.normalize(e2 - wp.dot(e2, basis1) * basis1)

        #     basis_mat_T = np.array([basis1, basis2])  # [2, 3]
        #     basis_mat = basis_mat_T.T  # [3, 2]
        #     Dm = basis_mat_T @ np.array([e1, e2]).T  # [2, 2]
        #     area = np.linalg.det(Dm) / 2.0
        #     tri_mass = self.density * area * self.thickness

        #     # print(f"area: {area}, tri_mass: {tri_mass}")

        #     if area <= 0.0:
        #         raise RuntimeError(
        #             f"Triangle area <= 0: area: {area}, triangle_id: {b + 1}, x0: {x0}, x1: {x1}, x2: {x2}"
        #         )
        #     else:
        #         self.inv_Dm[b] = np.linalg.inv(Dm)
        #         # print(f"inv_Dm: {self.inv_Dm[b]}")
        #         self.rest_areas[b] = area
        #         self.vertex_mass[np.array([i0, i1, i2])] += tri_mass / 3.0

        # Vectorize this!!

        triangles = self.tri_mesh.triangles  # [N_triangles, 3]
        vertices = self.tri_mesh.vertices  # [N_vertices, 3]
        tri_x = vertices[triangles]  # [N_triangles, 3 (tri), 3 (dim)]
        e1 = tri_x[:, 1] - tri_x[:, 0]  # [N_triangles, 3 (dim)]
        e2 = tri_x[:, 2] - tri_x[:, 0]  # [N_triangles, 3 (dim)]
        e1_norm = np.linalg.norm(e1, axis=1)  # [N_triangles]
        e2_norm = np.linalg.norm(e2, axis=1)  # [N_triangles]
        basis1 = e1 / e1_norm[:, None]  # [N_triangles, 3 (dim)]
        basis2 = (
            e2 - (e2 * basis1).sum(axis=1)[:, None] * basis1
        )  # [N_triangles, 3 (dim)]
        basis2_norm = np.linalg.norm(basis2, axis=1)  # [N_triangles]
        basis2 = basis2 / basis2_norm[:, None]  # [N_triangles, 3 (dim)]
        basis_mat_T = np.stack((basis1, basis2), axis=1)  # [N_triangles, 2, 3]
        basis_mat = basis_mat_T.transpose((0, 2, 1))  # [N_triangles, 3, 2]
        stacked_e = np.stack((e1, e2), axis=2)  # [N_triangles, 3, 2]
        Dm = np.matmul(basis_mat_T, stacked_e)
        # [N_triangles, 2, 2] = [N_triangles, 2, 3] @ [N_triangles, 3, 2]
        area = np.linalg.det(Dm) / 2.0  # [N_triangles]
        tri_mass = self.density * area * self.thickness  # [N_triangles]
        assert np.all(area > 0.0), f"Triangle area <= 0: {area.min()}"
        self.inv_Dm = np.linalg.inv(Dm)  # [N_triangles, 2, 2]
        self.rest_areas = area  # [N_triangles]
        self.vertex_mass = np.zeros(self.tri_mesh.n_vertices, dtype=np.float32)
        np.add.at(
            self.vertex_mass,
            triangles.reshape(-1),
            np.tile(tri_mass / 3.0, 3).reshape(-1),
        )

        # end_time = time.time()
        # print(f"process_tri_mesh: {end_time - start_time}")
        # exit(0)

        self.vertex_mass[self.vertex_mass == 0.0] = default_vertex_mass

        if self.k_hinge > 0.0:

            def insert_hinge(edge_endpoints, vertex):
                if edge_endpoints in hinge_dict:
                    raise RuntimeError("non-manifold edge")
                hinge_dict[edge_endpoints] = vertex

            for i0, i1, i2 in self.tri_mesh.triangles:
                insert_hinge((i0, i1), i2)
                insert_hinge((i1, i2), i0)
                insert_hinge((i2, i0), i1)

            for (i1, i2), i0 in hinge_dict.items():
                if i1 < i2:
                    if (i2, i1) in hinge_dict:
                        # non boundary edge
                        i3 = hinge_dict[(i2, i1)]
                        self.hinges.append([i0, i1, i2, i3])

                        x0 = self.tri_mesh.vertices[i0]
                        x1 = self.tri_mesh.vertices[i1]
                        x2 = self.tri_mesh.vertices[i2]
                        x3 = self.tri_mesh.vertices[i3]

                        e12 = x2 - x1
                        e12_len = np.linalg.norm(e12)
                        n1 = np.cross(x2 - x1, x0 - x1)  # triangle 012
                        n1 = n1 / np.linalg.norm(n1)
                        n2 = np.cross(x3 - x1, x2 - x1)  # triangle 132
                        n2 = n2 / np.linalg.norm(n2)
                        cos_theta = np.dot(n1, n2)
                        sin_theta = np.dot(np.cross(n1, n2), e12) / e12_len
                        theta = np.arctan2(sin_theta, cos_theta)
                        self.hinge_rest_angles.append(theta)

        if len(self.hinges) == 0:
            self.hinges = np.zeros((0, 4), dtype=np.int32)
            self.hinge_rest_angles = np.zeros(0, dtype=np.float32)
        else:
            self.hinges = np.array(self.hinges, dtype=np.int32)
            self.hinge_rest_angles = np.array(self.hinge_rest_angles, dtype=np.float32)

    def on_add_to_scene(self, scene: sapien.Scene):
        self._process_tri_mesh()

        scene.get_system("ipc").register_fem2d_component(self)

    def on_remove_from_scene(self, scene: sapien.Scene):
        scene.get_system("ipc").unregister_fem2d_component(self)

    def on_set_pose(self, pose: sapien.Pose):
        pass

    def set_kinematic_target(self, indices, positions):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_fem2d_kinematic_target(self, indices, positions)

    def set_positions(self, positions):
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.set_fem2d_positions(self, positions)

    def set_velocities(self, velocities):
        raise NotImplementedError()


class IPCConstraint:
    def __init__(
        self,
        constraint_type: int,
        components: List[Union[IPCABDComponent, IPCFEMComponent, IPCABDJointComponent]],
        particle_ids: List[int],
        param: np.ndarray = None,
        parent_joint: IPCABDJointComponent = None,
    ) -> None:
        """Constraint

        :param constraint_type: int, see global_defs.py
        :type constraint_type: int
        :param components: list of components related to this constraint
        :type components: List[Union[IPCABDComponent, IPCFEMComponent, IPCABDJointComponent]]
        :param particle_ids: list of particle_ids (in each component)
        :type particle_ids: List[int]
        :param parent_joint: parent joint or None, defaults to None
        :type parent_joint: IPCABDJointComponent, optional
        """
        self.id_in_system = None

        self.parent_joint = parent_joint
        self.type = constraint_type
        self.components = components
        self.particle_ids = particle_ids

        if param is not None and len(param) < 5:
            raise ValueError("param must be of length 5")
        self.param = param

        # each particle is: the particle_id-th particle of the component


class IPCABDJointComponent(sapien.Component):
    def __init__(self) -> None:
        super().__init__()

        self.id_in_system = None
        self.particle_begin_index = None
        self.particle_end_index = None

        self.type = None
        self.components = []
        self.virtual_particle_q_rest = None  # np.array([N, 3])
        self.virtual_particle_abd_component = (
            []
        )  # parent IPCABDComponent (to get affine_id)

        self.constraints = []  # list of IPCConstraint

    def on_add_to_scene(self, scene: sapien.Scene):
        scene.get_system("ipc").register_joint_component(self)

    def on_remove_from_scene(self, scene: sapien.Scene):
        scene.get_system("ipc").unregister_joint_component(self)

    def get_positions(self) -> torch.Tensor:
        assert self.entity is not None and self.entity.scene is not None

        system: IPCSystem = self.entity.scene.get_system("ipc")
        return system.get_vertex_positions(self)


class IPCABDBallJointComponent(IPCABDJointComponent):
    """A ball joint (spherical joint), which constrains two points to be
    the same, but allows rotation. It connects two ABD components.
    It creates one virtual particle for each component at the joint position.
    """

    def __init__(
        self,
        component0: IPCABDComponent,
        pos0: Union[tuple, list, np.array],
        component1: IPCABDComponent,
        pos1: Union[tuple, list, np.array],
    ) -> None:
        """Ball joint (spherical joint), which constrains two points to be
        the same, but allows rotation.

        :param component0: first ABD component
        :type component0: IPCABDComponent
        :param pos0: position of the joint relative to the rest position of component0
        :type pos0: Union[tuple, list, np.array]
        :param component1: second ABD component
        :type component1: IPCABDComponent
        :param pos1: position of the joint relative to the rest position of component1
        :type pos1: Union[tuple, list, np.array]

        :TODO: add support for rotation (limits)
        """

        super().__init__()

        self.type = "ball"
        self.virtual_particle_q_rest = np.array([pos0, pos1], dtype=np.float32)
        self.virtual_particle_abd_component = [component0, component1]

        self.constraints = [
            IPCConstraint(
                constraint_type=CONSTRAINT_EQ2,
                components=[self, self],
                particle_ids=[0, 1],
                parent_joint=self,
            )
        ]


class IPCABDHingeJointComponent(IPCABDJointComponent):
    def __init__(
        self,
        component0: IPCABDComponent,
        pos0: Union[tuple, list, np.array],
        fixed_axis0: Union[tuple, list, np.array],
        rotate_axis0: Union[tuple, list, np.array],
        component1: IPCABDComponent,
        pos1: Union[tuple, list, np.array],
        fixed_axis1: Union[tuple, list, np.array],
        rotate_axis1: Union[tuple, list, np.array],
        proportional_gain: float = 0.0,
        integral_gain: float = 0.0,  # TODO
        derivative_gain: float = 0.0,
        proportional_target: float = 0.0,
        derivative_target: float = 0.0,
    ) -> None:
        """Initialize a hinge joint (revolute joint), which constrains two points to be
        the same, but allows rotation around an axis. It connects two ABD components.
        It creates four virtual particles for each component at the joint position and
        (joint position + joint axis): {pos0, pos1, pos0 + axis0, pos1 + axis1}

        :param component0: first ABD component
        :type component0: IPCABDComponent
        :param pos0: position of the joint relative to the rest position of component0
        :type pos0: Union[tuple, list, np.array]
        :param axis0: axis of rotation of the joint relative to the rest position of component0
        :type axis0: Union[tuple, list, np.array]
        :param component1: second ABD component
        :type component1: IPCABDComponent
        :param pos1: position of the joint relative to the rest position of component1
        :type pos1: Union[tuple, list, np.array]
        :param axis1: axis of rotation of the joint relative to the rest position of component1
        :type axis1: Union[tuple, list, np.array]
        """

        super().__init__()

        self.type = "hinge"

        pos0 = np.array(pos0, dtype=np.float32)
        pos1 = np.array(pos1, dtype=np.float32)
        fixed_axis0 = np.array(fixed_axis0, dtype=np.float32)
        fixed_axis1 = np.array(fixed_axis1, dtype=np.float32)

        # 6 virtual particles
        self.virtual_particle_q_rest = np.array(
            [
                pos0,
                pos1,
                pos0 + fixed_axis0,
                pos1 + fixed_axis1,
                pos0 + rotate_axis0,
                pos1 + rotate_axis1,
            ],
            dtype=np.float32,
        )

        self.virtual_particle_abd_component = [
            component0,
            component1,
            component0,
            component1,
            component0,
            component1,
        ]

        self.constraints = [
            IPCConstraint(
                constraint_type=CONSTRAINT_EQ2,
                components=[self, self],
                particle_ids=[0, 1],
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_EQ2,
                components=[self, self],
                particle_ids=[2, 3],
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_ANGULAR_PID_CONTROL,
                components=[self, self, self, self],
                particle_ids=[0, 2, 5, 4],
                param=np.array(
                    [
                        proportional_gain,
                        integral_gain,
                        derivative_gain,
                        proportional_target,
                        derivative_target,
                    ],
                    dtype=np.float32,
                ),
                parent_joint=self,
            ),
        ]

    def set_pid_gain(
        self, proportional_gain=None, integral_gain=None, derivative_gain=None
    ):
        con = self.constraints[2]
        if proportional_gain is not None:
            con.param[0] = proportional_gain
        if integral_gain is not None:
            con.param[1] = integral_gain
        if derivative_gain is not None:
            con.param[2] = derivative_gain
        if self.entity is None or self.entity.scene is None:
            return
        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.update_constraint_param(con)

    def set_pid_target(self, proportional_target=None, derivative_target=None):
        con = self.constraints[2]
        if proportional_target is not None:
            con.param[3] = proportional_target
        if derivative_target is not None:
            con.param[4] = derivative_target
        if self.entity is None or self.entity.scene is None:
            return
        system: IPCSystem = self.entity.scene.get_system("ipc")
        system.update_constraint_param(con)


class IPCABDPrismaticJointComponent(IPCABDJointComponent):
    def __init__(
        self,
        component0,
        pos0,
        aligned_axis0,
        perp_axis0,  # an axis perpendicular to aligned_axis0, and always parallel to perp_axis_1
        component1,
        pos1,
        aligned_axis1,
        perp_axis1,
    ) -> None:
        super().__init__()

        self.type = "prismatic"

        pos0 = np.array(pos0, dtype=np.float32)
        pos1 = np.array(pos1, dtype=np.float32)
        aligned_axis0 = np.array(aligned_axis0, dtype=np.float32)
        aligned_axis1 = np.array(aligned_axis1, dtype=np.float32)
        perp_axis0 = np.array(perp_axis0, dtype=np.float32)
        perp_axis1 = np.array(perp_axis1, dtype=np.float32)

        # 6 virtual particles
        self.virtual_particle_q_rest = np.array(
            [
                pos0,
                pos1,
                pos0 + aligned_axis0,
                pos1 + aligned_axis1,
                pos0 + perp_axis0,
                pos1 + perp_axis1,
            ],
            dtype=np.float32,
        )

        self.virtual_particle_abd_component = [
            component0,
            component1,
            component0,
            component1,
            component0,
            component1,
        ]

        self.constraints = [
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 2, 0, 1],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 2, 0, 3],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 4, 1, 5],
                param=None,
                parent_joint=self,
            ),
        ]


class IPCABDPrismaticPairJointComponent(IPCABDJointComponent):
    def __init__(
        self,
        base_component,
        base_pos,
        base_aligned_axis,
        base_perp_axis,
        component0,
        pos0,
        aligned_axis0,
        perp_axis0,
        component1,
        pos1,
        aligned_axis1,
        perp_axis1,
        proportional_gain: float = 0.0,
        integral_gain: float = 0.0,  # TODO
        derivative_gain: float = 0.0,
        proportional_target: float = 0.0,
        derivative_target: float = 0.0,
    ) -> None:
        super().__init__()

        self.type = "prismatic_pair"

        """ 
              ___6__0__3_7__1__4      ---> aligned
                /|    /|             /
               8 |2  5 |            L perp
              ___|_____|___
                /     /  
        """

        base_pos = np.array(base_pos, dtype=np.float32)
        pos0 = np.array(pos0, dtype=np.float32)
        pos1 = np.array(pos1, dtype=np.float32)
        base_aligned_axis = np.array(base_aligned_axis, dtype=np.float32)
        aligned_axis0 = np.array(aligned_axis0, dtype=np.float32)
        aligned_axis1 = np.array(aligned_axis1, dtype=np.float32)
        base_perp_axis = np.array(base_perp_axis, dtype=np.float32)
        perp_axis0 = np.array(perp_axis0, dtype=np.float32)
        perp_axis1 = np.array(perp_axis1, dtype=np.float32)

        # 9 virtual particles
        self.virtual_particle_q_rest = np.array(
            [
                base_pos,  # 0
                pos0,  # 1
                pos1,  # 2
                base_pos + base_aligned_axis,  # 3
                pos0 + aligned_axis0,  # 4
                pos1 + aligned_axis1,  # 5
                base_pos + base_perp_axis,  # 6
                pos0 + perp_axis0,  # 7
                pos1 + perp_axis1,  # 8
            ],
            dtype=np.float32,
        )

        self.virtual_particle_abd_component = [
            base_component,
            component0,
            component1,
            base_component,
            component0,
            component1,
            base_component,
            component0,
            component1,
        ]

        self.constraints = [
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 3, 0, 1],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 3, 0, 4],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 6, 1, 7],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 3, 0, 2],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 3, 0, 5],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_AREA,
                components=[self, self, self, self],
                particle_ids=[0, 6, 2, 8],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_DIST_SYM,
                components=[self, self, self, self],
                particle_ids=[0, 1, 0, 2],
                param=None,
                parent_joint=self,
            ),
            IPCConstraint(
                constraint_type=CONSTRAINT_DIST_PID_CONTROL,
                components=[self, self, self],
                particle_ids=[0, 1, 3],
                param=np.array(
                    [
                        proportional_gain,
                        integral_gain,
                        derivative_gain,
                        proportional_target,
                        derivative_target,
                    ],
                    dtype=np.float32,
                ),
            ),
            # IPCConstraint(
            #     constraint_type=CONSTRAINT_DIST_PID_CONTROL,
            #     components=[self, self],
            #     particle_ids=[0, 2],
            #     param=np.array(
            #         [
            #             proportional_gain,
            #             integral_gain,
            #             derivative_gain,
            #             proportional_target,
            #             derivative_target,
            #         ],
            #         dtype=np.float32,
            #     ),
            # ),
        ]

    def set_pid_gain(
        self, proportional_gain=None, integral_gain=None, derivative_gain=None
    ):
        for con in self.constraints:
            if con.type == CONSTRAINT_DIST_PID_CONTROL:
                if proportional_gain is not None:
                    con.param[0] = proportional_gain
                if integral_gain is not None:
                    con.param[1] = integral_gain
                if derivative_gain is not None:
                    con.param[2] = derivative_gain
        if self.entity is None or self.entity.scene is None:
            return
        system: IPCSystem = self.entity.scene.get_system("ipc")
        for con in self.constraints:
            if con.type == CONSTRAINT_DIST_PID_CONTROL:
                system.update_constraint_param(con)

    def set_pid_target(self, proportional_target=None, derivative_target=None):
        for con in self.constraints:
            if con.type == CONSTRAINT_DIST_PID_CONTROL:
                if proportional_target is not None:
                    con.param[3] = proportional_target
                if derivative_target is not None:
                    con.param[4] = derivative_target
        if self.entity is None or self.entity.scene is None:
            return
        system: IPCSystem = self.entity.scene.get_system("ipc")
        for con in self.constraints:
            if con.type == CONSTRAINT_DIST_PID_CONTROL:
                system.update_constraint_param(con)
