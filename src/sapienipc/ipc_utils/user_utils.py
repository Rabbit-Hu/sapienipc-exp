import sapien
import warp as wp
from sapien.render import RenderCudaMeshComponent

from ..ipc_component import IPCBaseComponent
from ..ipc_system import IPCSystem
from .warp_types import wp_slice
from .logging_utils import ipc_logger


@wp.kernel
def copy_positions_to_render(
    dst_vertices: wp.array2d(dtype=wp.float32),
    src_positions: wp.array(dtype=wp.vec3),
):
    i, j = wp.tid()
    dst_vertices[i, j] = src_positions[i][j]


def ipc_update_render(ipc_component: IPCBaseComponent, render_component: RenderCudaMeshComponent):
    """Call this after system.step() and before scene.update_render()
    to copy the vertex positions from the simulation to the render component
    and notify the render component that the vertex positions have been updated.

    :param ipc_component: the sapienipc component with mesh
    :type ipc_component: IPCBaseComponent
    :param render_component: the sapien render component (triangles already set)
    :type render_component: RenderCudaMeshComponent
    """
    system: IPCSystem = ipc_component.entity.scene.get_system("ipc")
    device = system.config.device
    # change ptr to that one from interface dict
    interface = render_component.cuda_vertices.__cuda_array_interface__
    dst = wp.array(
        ptr=interface['data'][0],
        dtype=wp.float32,
        shape=interface['shape'],
        strides=interface['strides'],
        owner=False,
    )
    src = wp_slice(
        system.particle_q,
        ipc_component.particle_begin_index,
        ipc_component.particle_end_index,
    )
    wp.launch(
        kernel=copy_positions_to_render,
        dim=(ipc_component.particle_end_index - ipc_component.particle_begin_index, 3),
        inputs=[dst, src],
        device=device
    )
    render_component.notify_vertex_updated(
        wp.get_stream(device).cuda_stream
    )


def ipc_update_render_all(scene: sapien.Scene):
    """Call this after system.step() and before scene.update_render()
    to copy the vertex positions from the simulation to the render component
    and notify the render component that the vertex positions have been updated.

    :param ipc_system: the sapienipc system
    :type ipc_system: IPCSystem
    """
    for entity in scene.get_entities():
        ipc_component = entity.find_component_by_type(IPCBaseComponent)
        render_component = entity.find_component_by_type(RenderCudaMeshComponent)
        if ipc_component is None:
            continue
        if render_component is None:
            ipc_logger.info(f"Entity \"{entity.name}\" has IPC component but no render component")
            continue
        ipc_update_render(ipc_component, render_component)
