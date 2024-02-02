import os
import time

import cv2

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from sapienipc.ipc_utils.user_utils import ipc_update_render_all
import warp as wp
import sapien

from sapien.utils import Viewer

from sapienipc.ipc_utils.ipc_mesh import IPCTetMesh, IPCTriMesh
from sapienipc.ipc_component import IPCFEMComponent, IPCABDComponent, IPCPlaneComponent
from sapienipc.ipc_system import IPCSystem, IPCSystemConfig
from sapienipc.ipc_utils.logging_utils import ipc_logger


ipc_logger.setLevel("DEBUG")

wp.config.kernel_cache_dir = "./build"
wp.init()
np.random.seed(0)

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))

wp_device = wp.get_preferred_device()
torch_device = wp.device_to_torch(wp_device)
# torch.cuda.set_stream(torch.cuda.Stream())

print(f"wp_device={wp_device}, torch_device={torch_device}")
print(f"wp_stream={wp.get_stream().cuda_stream}, torch_stream={torch.cuda.current_stream().cuda_stream}")


######## Example scene parameters ########

sensor_friction = 2.0
abd_friction = 0.0

sensor_color = [0.3, 1.0, 1.0, 1.0]  # cyan
hole_color = [1.0, 1.0, 0.3, 1.0]  # yellow
peg_colors = [
    [1.0, 0.3, 0.3, 1.0],  # red
    [0.3, 1.0, 0.3, 1.0],  # green
    [0.3, 0.3, 1.0, 1.0],  # blue
]

n_scenes = 64
n_steps = 1000
# offsets = np.random.randn(n_scenes, 2) * 1e-2 + np.array([0.0, 0.015])
# offsets = np.array([[0.0, 0.015]] * n_scenes)
offsets = np.random.randn(n_scenes, 3) * np.array([0.0, 1e-2, 0.0]) + np.array(
    [0.0, 0.005, 0.021]
)
# offsets = np.random.randn(64, 3)[49:50] * np.array([0.0, 1e-2, 0.0]) + np.array(
#     [0.0, 0.005, 0.021]
# )

sensor_1_keyframes = [
    (0.0, [-0.0171, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (0.1, [-0.0165, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (2.0, [-0.0165, -0.006, 0.068], [0.70710677, 0.0, 0.70710677, 0.0]),
    (4.0, [-0.0165, -0.006, 0.05], [0.70710677, 0.0, 0.70710677, 0.0]),
]
sensor_2_keyframes = [
    (0.0, [0.0171, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (0.1, [0.0165, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (2.0, [0.0165, -0.006, 0.068], [0.0, -0.70710677, 0.0, 0.70710677]),
    (4.0, [0.0165, -0.006, 0.05], [0.0, -0.70710677, 0.0, 0.70710677]),
]


def interpolate_keyframes(keyframes, time_step):
    p_traj = []
    q_traj = []
    T_traj = []
    for i in range(len(keyframes) - 1):
        t0, p0, q0 = keyframes[i]
        t1, p1, q1 = keyframes[i + 1]

        assert abs(t0 - round(t0 / time_step) * time_step) < 1e-6
        assert abs(t1 - round(t1 / time_step) * time_step) < 1e-6

        step0 = int(round(t0 / time_step))
        step1 = int(round(t1 / time_step))

        if i > 0:
            p_traj = p_traj[:-1]
            q_traj = q_traj[:-1]

        p_traj += np.linspace(p0, p1, int(step1 - step0 + 1)).tolist()
        slerp = Slerp([t0, t1], Rotation.from_quat([q0, q1]))
        q_traj += slerp(np.linspace(t0, t1, int(step1 - step0 + 1))).as_quat().tolist()

    if len(p_traj) == 0:
        p_traj = [keyframes[0][1]]
        q_traj = [keyframes[0][2]]

    for i in range(len(p_traj)):
        T_traj.append(
            torch.from_numpy(
                sapien.Pose(p=p_traj[i], q=q_traj[i]).to_transformation_matrix()
            ).to(torch_device)
        )

    return np.array(p_traj), np.array(q_traj), torch.stack(T_traj)


######## Create system ########

ipc_system_config = IPCSystemConfig()
ipc_system_config.d_hat = 1e-4
ipc_system_config.eps_d = 1e-6
ipc_system_config.v_max = 1e-1
ipc_system_config.kappa = 1e2
ipc_system_config.kappa_affine = 1e3
ipc_system_config.kappa_con = 1e3
ipc_system_config.newton_max_iters = 4
ipc_system_config.cg_max_iters = 40
ipc_system_config.cg_error_frequency = 40
ipc_system_config.cg_error_tolerance = 1e-9
ipc_system_config.ee_classify_thres = 1e-3
ipc_system_config.ee_mollifier_thres = 1e-3
ipc_system_config.ccd_slackness = 0.7
ipc_system_config.ccd_thickness = 1e-8
# ipc_system_config.ccd_tet_inversion_thres = 1e-6
ipc_system_config.allow_self_collision = False
# ipc_system_config.debug = True
# ipc_system_config.use_graph = False
ipc_system_config.max_blocks = 1000000

ipc_system = IPCSystem(ipc_system_config)


sensor_1_p_traj, sensor_1_q_traj, sensor_1_T_traj = interpolate_keyframes(
    sensor_1_keyframes, ipc_system_config.time_step
)
sensor_2_p_traj, sensor_2_q_traj, sensor_2_T_traj = interpolate_keyframes(
    sensor_2_keyframes, ipc_system_config.time_step
)

######## Create multiple scenes ########

scenes = []

hole_components = []

sensor_1_components = []
sensor_1_dbc_indices_list = []
sensor_1_dbc_vertices_list = []

sensor_2_components = []
sensor_2_dbc_indices_list = []
sensor_2_dbc_vertices_list = []

for i in range(n_scenes):
    scene = sapien.Scene()
    scene.add_system(ipc_system)
    scenes.append(scene)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, -1, -1], [0.5, 0.5, 0.5], True)
    scene.set_environment_map(os.path.join(assets_dir, "env.ktx"))

    # add a camera to indicate shader
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(512, 512)
    cam_entity.add_component(cam)
    cam_entity.name = "camera"
    cam_entity.set_pose(
        sapien.Pose(
            [-0.0379048, -0.0170053, 0.0775995],
            [0.960526, -0.0394655, 0.19271, 0.196709],
        )
    )
    scene.add_entity(cam_entity)

    ######## Ground ########

    # ground_component = IPCPlaneComponent()
    # ground_component.set_friction(0.5)
    # ground_render_component = sapien.render.RenderBodyComponent()
    # ground_render_component.attach(
    #     sapien.render.RenderShapePlane(
    #         np.array([1.0, 1.0, 1.0]),
    #         sapien.render.RenderMaterial(base_color=[1.0, 1.0, 1.0, 1.0]),
    #     )
    # )
    # ground_entity = sapien.Entity()
    # ground_entity.add_component(ground_component)
    # ground_entity.add_component(ground_render_component)
    # ground_entity.set_pose(sapien.Pose(p=[0, 0, -0.02], q=[0.7071068, 0, -0.7071068, 0]))
    # if i == 0:
    #     scene.add_entity(ground_entity)

    ######## Load components ########

    peg_component = IPCABDComponent()
    peg_component.set_tet_mesh(
        IPCTetMesh(filename=os.path.join(assets_dir, "concave1_peg_tet.msh"))
    )
    peg_component.set_density(500.0)
    peg_component.set_friction(abd_friction)
    peg_entity = sapien.Entity()
    peg_entity.add_component(peg_component)

    peg_render_component = sapien.render.RenderCudaMeshComponent(
        peg_component.tet_mesh.n_vertices, peg_component.tet_mesh.n_surface_triangles
    )
    peg_render_component.set_vertex_count(peg_component.tet_mesh.n_vertices)
    peg_render_component.set_triangle_count(peg_component.tet_mesh.n_surface_triangles)
    peg_render_component.set_triangles(peg_component.tet_mesh.surface_triangles)
    peg_render_component.set_material(
        sapien.render.RenderMaterial(base_color=peg_colors[i % len(peg_colors)])
    )
#     peg_render_component.set_data_source(peg_component)
    peg_entity.add_component(peg_render_component)

    hole_component = IPCABDComponent()
    hole_component.set_tet_mesh(
        IPCTetMesh(filename=os.path.join(assets_dir, "concave1_hole_2.0mm_tet.msh"))
    )
    hole_component.set_density(500.0)
    hole_component.set_friction(abd_friction)
    hole_components.append(hole_component)
    hole_entity = sapien.Entity()
    hole_entity.add_component(hole_component)

    hole_render_component = sapien.render.RenderCudaMeshComponent(
        hole_component.tet_mesh.n_vertices, hole_component.tet_mesh.n_surface_triangles
    )
    hole_render_component.set_vertex_count(hole_component.tet_mesh.n_vertices)
    hole_render_component.set_triangle_count(
        hole_component.tet_mesh.n_surface_triangles
    )
    hole_render_component.set_triangles(hole_component.tet_mesh.surface_triangles)
    hole_render_component.set_material(
        sapien.render.RenderMaterial(base_color=hole_color)
    )
#     hole_render_component.set_data_source(hole_component)
    hole_entity.add_component(hole_render_component)

    scene.add_entity(hole_entity)
    # hole_component.load_state_dict(hole_dict)

    hole_height = np.max(hole_component.tet_mesh.vertices[:, 2]) - np.min(
        hole_component.tet_mesh.vertices[:, 2]
    )
    peg_height = np.max(peg_component.tet_mesh.vertices[:, 2]) - np.min(
        peg_component.tet_mesh.vertices[:, 2]
    )
    x_offset = 0.0
    y_offset = 0.015

    # peg_entity.set_pose(sapien.Pose(p=[x_offset, y_offset, hole_height + 1e-3]))
    peg_entity.set_pose(sapien.Pose(p=offsets[i]))

    scene.add_entity(peg_entity)
    # peg_component.load_state_dict(peg_dict)

    ######## Tactile Sensor 1 ########

    sensor_1_cpnt = IPCFEMComponent()
    sensor_1_cpnt.set_tet_mesh(
        IPCTetMesh(filename=os.path.join(assets_dir, "gel.msh"))
    )
    sensor_1_cpnt.set_material(1e3, 1e5, 0.4)
    sensor_1_cpnt.set_friction(sensor_friction)

    sensor_1_r_cpnt = sapien.render.RenderCudaMeshComponent(
        sensor_1_cpnt.tet_mesh.n_vertices, sensor_1_cpnt.tet_mesh.n_surface_triangles
    )
    sensor_1_r_cpnt.set_vertex_count(sensor_1_cpnt.tet_mesh.n_vertices)
    sensor_1_r_cpnt.set_triangle_count(sensor_1_cpnt.tet_mesh.n_surface_triangles)
    sensor_1_r_cpnt.set_triangles(sensor_1_cpnt.tet_mesh.surface_triangles)
    sensor_1_r_cpnt.set_material(sapien.render.RenderMaterial(base_color=sensor_color))
#     sensor_1_r_cpnt.set_data_source(sensor_1_cpnt)

    sensor_1_components.append(sensor_1_cpnt)

    sensor_1_etty = sapien.Entity()
    sensor_1_etty.add_component(sensor_1_r_cpnt)
    sensor_1_etty.add_component(sensor_1_cpnt)
    sensor_1_etty.set_pose(
        sapien.Pose(p=sensor_1_keyframes[0][1], q=sensor_1_keyframes[0][2])
    )
    scene.add_entity(sensor_1_etty)

    sensor_1_dbc_indices = sensor_1_cpnt.select_box(
        [-1.0, -1.0, -2.0e-3 - 1e-5], [1.0, 1.0, -2.0e-3 + 1e-5]
    )
    # sensor_1_dbc_q = sensor_1_cpnt.tensor_slice.cpu().numpy()[sensor_1_dbc_indices]
    sensor_1_dbc_vertices = sensor_1_cpnt.tet_mesh.vertices[sensor_1_dbc_indices]

    sensor_1_dbc_indices_list.append(
        torch.from_numpy(sensor_1_dbc_indices).to(torch_device)
    )
    sensor_1_dbc_vertices_list.append(
        torch.from_numpy(sensor_1_dbc_vertices).to(torch_device)
    )

    ######## Tactile Sensor 2 ########

    sensor_2_cpnt = IPCFEMComponent()
    sensor_2_cpnt.set_tet_mesh(
        IPCTetMesh(filename=os.path.join(assets_dir, "gel.msh"))
    )
    sensor_2_cpnt.set_material(1e3, 1e5, 0.4)
    sensor_2_cpnt.set_friction(sensor_friction)

    sensor_2_r_cpnt = sapien.render.RenderCudaMeshComponent(
        sensor_2_cpnt.tet_mesh.n_vertices, sensor_2_cpnt.tet_mesh.n_surface_triangles
    )
    sensor_2_r_cpnt.set_vertex_count(sensor_2_cpnt.tet_mesh.n_vertices)
    sensor_2_r_cpnt.set_triangle_count(sensor_2_cpnt.tet_mesh.n_surface_triangles)
    sensor_2_r_cpnt.set_triangles(sensor_2_cpnt.tet_mesh.surface_triangles)
    sensor_2_r_cpnt.set_material(sapien.render.RenderMaterial(base_color=sensor_color))
#     sensor_2_r_cpnt.set_data_source(sensor_2_cpnt)

    sensor_2_components.append(sensor_2_cpnt)

    sensor_2_etty = sapien.Entity()
    sensor_2_etty.add_component(sensor_2_r_cpnt)
    sensor_2_etty.add_component(sensor_2_cpnt)
    sensor_2_etty.set_pose(
        sapien.Pose(p=sensor_2_keyframes[0][1], q=sensor_2_keyframes[0][2])
    )
    scene.add_entity(sensor_2_etty)

    sensor_2_dbc_indices = sensor_2_cpnt.select_box(
        [-1.0, -1.0, -2.0e-3 - 1e-5], [1.0, 1.0, -2.0e-3 + 1e-5]
    )
    # sensor_2_dbc_q = sensor_2_cpnt.tensor_slice.cpu().numpy()[sensor_2_dbc_indices]
    sensor_2_dbc_vertices = sensor_2_cpnt.tet_mesh.vertices[sensor_2_dbc_indices]

    sensor_2_dbc_indices_list.append(
        torch.from_numpy(sensor_2_dbc_indices).to(torch_device)
    )
    sensor_2_dbc_vertices_list.append(
        torch.from_numpy(sensor_2_dbc_vertices).to(torch_device)
    )

# precompute the dbc vertices
sensor_1_dbc_vertices_traj = [[]] * n_scenes
sensor_2_dbc_vertices_traj = [[]] * n_scenes
for i in range(n_scenes):
    for step in range(n_steps):
        T_1 = sensor_1_T_traj[min(step, len(sensor_1_T_traj) - 1)]
        T_2 = sensor_2_T_traj[min(step, len(sensor_2_T_traj) - 1)]

        c, dbc_indices, dbc_vertices = (
            sensor_1_components[i],
            sensor_1_dbc_indices_list[i],
            sensor_1_dbc_vertices_list[i],
        )
        sensor_1_dbc_vertices_traj[i].append(
            torch.mm(dbc_vertices, T_1[:3, :3].T) + T_1[:3, 3]
        )
        c, dbc_indices, dbc_vertices = (
            sensor_2_components[i],
            sensor_2_dbc_indices_list[i],
            sensor_2_dbc_vertices_list[i],
        )
        sensor_2_dbc_vertices_traj[i].append(
            torch.mm(dbc_vertices, T_2[:3, :3].T) + T_2[:3, 3]
        )

print(f"sensor_1_dbc_vertices_traj[0][0].dtype={sensor_1_dbc_vertices_traj[0][0].dtype}")

hole_proxy_positions = hole_component.get_proxy_positions()
print(f"hole_proxy_positions={hole_proxy_positions}")

######## Render ########

viewer = Viewer()
viewing_scene = 0
viewer.set_scene(scenes[viewing_scene])
viewer.set_camera_pose(
    sapien.Pose(
        [-0.0669768, -0.0243881, 0.0526962], [0.95626, -0.028899, 0.101262, 0.272907]
    )
)
viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)

print("Press 'c' to start simulation")
print("Press 0-9 to switch scene")


def switch_scene(scene_id):
    global viewing_scene
    global last_switch_time
    viewing_scene = scene_id
    viewer.set_scene(scenes[viewing_scene])
    viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
    ipc_logger.info(f"Switch to scene {viewing_scene}")
    last_switch_time = time.time()


def check_switch_scene():
    if time.time() - last_switch_time > 0.2:
        for i in range(min(n_scenes, 10)):
            if viewer.window.key_down(f"{i}"):  # press 0-9 to switch scene
                switch_scene(i)
        if viewer.window.key_down("left") and viewing_scene > 0:
            switch_scene((viewing_scene - 1) % n_scenes)
        if viewer.window.key_down("right") and viewing_scene < n_scenes - 1:
            switch_scene((viewing_scene + 1) % n_scenes)
        if viewer.window.key_down("up") and viewing_scene >= 10:
            switch_scene((viewing_scene - 10) % n_scenes)
        if viewer.window.key_down("down") and viewing_scene < n_scenes - 10:
            switch_scene((viewing_scene + 10) % n_scenes)


paused = True
last_switch_time = time.time()
while paused:
    scenes[viewing_scene].update_render()
    ipc_update_render_all(scenes[viewing_scene])
    viewer.render()
    if viewer.window.key_down("c"):
        paused = False
    check_switch_scene()

# output_dir = os.path.join(
#     os.path.dirname(__file__), "output/example_peg_multiple_scenes_1/newton_40"
# )
# os.makedirs(output_dir, exist_ok=True)

step = 0

remove_steps = np.arange(0, n_scenes) * 25 + 100
# remove_steps = np.arange(0, n_scenes) * 25 + 10000

# while not viewer.closed:
for _ in range(n_steps):
    check_switch_scene()

    ipc_logger.info(f"Visualize scene {viewing_scene}, step {step}")

    for i in range(n_scenes):
        if step < remove_steps[i]:
            c, dbc_indices, dbc_vertices = (
                sensor_1_components[i],
                sensor_1_dbc_indices_list[i],
                sensor_1_dbc_vertices_traj[i][step],
            )
            c.set_kinematic_target(dbc_indices, dbc_vertices)
            c, dbc_indices, dbc_vertices = (
                sensor_2_components[i],
                sensor_2_dbc_indices_list[i],
                sensor_2_dbc_vertices_traj[i][step],
            )
            c.set_kinematic_target(dbc_indices, dbc_vertices)
        elif step == remove_steps[i]:
            sensor_1_components[i].entity.remove_from_scene()
            sensor_2_components[i].entity.remove_from_scene()
            ipc_system.rebuild()

    for hole_component in hole_components:
        # hole_component.set_kinematic_target_pose(sapien.Pose())
        hole_component.set_kinematic_target(hole_proxy_positions)

    ipc_system.step()
    scenes[viewing_scene].update_render()
    ipc_update_render_all(scenes[viewing_scene])
    viewer.render()

    step += 1

    # image = viewer.window.get_picture(viewer.render_target)
    # cv2.imwrite(
    #     os.path.join(output_dir, f"step_{step:04d}.png"), image[:, :, 2::-1] * 255
    # )
