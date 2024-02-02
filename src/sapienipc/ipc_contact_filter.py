from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ipc_system import IPCSystem, IPCSystemConfig

import unittest

import numpy as np
import warp as wp
import sapien

from .ipc_kernels.contact_filter_kernels import *
from .ipc_utils.logging_utils import ipc_logger


class IPCContactFilter:
    """
    r/w: contact_counter, block_type, block_indices
    r: x, x_rest, surface_particles, affine_block_ids, particle_affine
    w: contact_type
    """
    
    def __init__(self, system: IPCSystem) -> None:
        self.system = system
        
    def filter(self, x) -> None:
        s = self.system
        c = s.config
        
        s.contact_counter.zero_()
        
        wp.launch(
            kernel=pg_filter_kernel,
            dim=(s.n_scenes, s.n_surface_particles_max, s.n_surface_planes_max),
            inputs=[
                s.contact_counter,
                s.n_static_blocks,
                c.max_blocks,
                s.n_surface_particles_max * s.n_surface_planes_max,
                x,
                s.particle_q_rest,
                s.n_surface_particles,
                s.n_surface_planes,
                s.surface_particles,
                s.surface_planes,
                s.plane_normals,
                s.plane_offsets,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_component,
                s.plane_component,
                s.particle_scene,
                s.plane_scene,
                s.component_removed,
                s.component_friction,
                s.component_group,
                # c.ground_altitude,
                c.d_hat,
                c.time_step,
                c.v_max, 
                s.block_type,
                s.contact_friction,
            ],
            device=c.device,
        )
        
        wp.launch(
            kernel=pt_filter_kernel,
            dim=(s.n_scenes, s.n_surface_particles_max, s.n_surface_triangles_max),
            inputs=[
                s.contact_counter,
                s.n_static_blocks,
                c.max_blocks,
                s.n_surface_particles_max * s.n_surface_triangles_max,
                x,
                s.particle_q_rest,
                s.n_surface_particles,
                s.n_surface_triangles,
                s.surface_particles,
                s.surface_triangles,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_component,
                s.particle_scene,
                s.component_removed,
                s.component_friction,
                s.component_group,
                c.d_hat,
                c.time_step,
                c.v_max,
                c.allow_self_collision,
                s.block_type,
                s.contact_friction,
            ],
            device=c.device,
        )
        
        wp.launch(
            kernel=ee_filter_kernel,
            dim=(s.n_scenes, s.n_surface_edges_max, s.n_surface_edges_max),
            inputs=[
                s.contact_counter,
                s.n_static_blocks,
                c.max_blocks, 
                s.n_surface_edges_max * s.n_surface_edges_max,
                x,
                s.particle_q_rest,
                s.n_surface_edges,
                s.surface_edges,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                s.particle_component,
                s.particle_scene,
                s.component_removed,
                s.component_friction,
                s.component_group,
                c.d_hat,
                c.time_step,
                c.v_max,
                c.ee_classify_thres,
                c.allow_self_collision,
                s.block_type,
                s.contact_friction,
            ],
            device=c.device,
        )
        
        n_contact = s.contact_counter.numpy()[0]
        s.n_blocks_this_step = int(s.n_static_blocks + n_contact)
        ipc_logger.debug(f"contact_counter: {n_contact}")
        
        assert s.n_blocks_this_step <= s.config.max_blocks, f"Too many Hessian blocks! ({s.n_blocks_this_step} > {s.config.max_blocks})"
