from __future__ import annotations
from typing import TYPE_CHECKING

from sapienipc.ipc_utils.warp_types import wp_slice
from sapienipc.ipc_utils.logging_utils import ipc_logger

if TYPE_CHECKING:
    from .ipc_system import IPCSystem

import warp as wp

from .ipc_kernels.ccd_kernels import *
from .ipc_kernels.utils_kernels import *


class IPCCCD:
    def __init__(self, system: IPCSystem):
        self.system = system

    def compute_step(self, x, v):
        s = self.system
        c = s.config

        wp_slice(s.ccd_step, 0, s.n_scenes).fill_(1.0)

        wp.launch(
            kernel=ccd_kernel,
            dim=s.n_blocks_this_step,
            inputs=[
                x,
                v,
                s.particle_scene,
                s.n_static_blocks,
                s.block_type,
                s.particle_q_rest,
                s.block_indices,
                s.affine_block_ids,
                s.particle_affine,
                # c.ground_altitude,
                s.plane_normals,
                s.plane_offsets,
                c.ccd_slackness,
                c.ccd_thickness,
                c.ccd_max_iters,
                c.ccd_tet_inversion_thres,
                c.ee_classify_thres,
            ],
            outputs=[s.ccd_step],
            device=c.device,
        )

        if c.debug:
            ipc_logger.debug(f"CCD step: {s.ccd_step.numpy()[:s.n_scenes]}")
