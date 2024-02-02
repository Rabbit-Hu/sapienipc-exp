import os
import time

import cv2
from PIL import Image

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sapienipc.ipc_utils.user_utils import ipc_update_render_all
import warp as wp
import sapien

from sapien.utils import Viewer

from sapienipc.ipc_utils.ipc_mesh import IPCTetMesh, IPCTriMesh
from sapienipc.ipc_component import IPCFEMComponent, IPCABDComponent, IPCPlaneComponent
from sapienipc.ipc_system import IPCSystem, IPCSystemConfig
from sapienipc.ipc_utils.logging_utils import ipc_logger


# sapien.render.set_viewer_shader_dir("rt")
# sapien.render.set_camera_shader_dir("rt")
# sapien.render.set_ray_tracing_samples_per_pixel(256)

ipc_logger.setLevel("DEBUG")

wp.config.kernel_cache_dir = "./build"
wp.init()

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))

wp_device = wp.get_preferred_device()

######## Create scene, and camera entity ########

scene = sapien.Scene()

scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, -1, -1], [0.5, 0.5, 0.5], True)
scene.set_environment_map(os.path.join(assets_dir, "env.ktx"))
# scene.add_ground(0)

# add a camera to indicate shader
cam_entity = sapien.Entity()
cam = sapien.render.RenderCameraComponent(512, 512)
cam.set_near(0.001)
cam.set_far(100)
cam_entity.add_component(cam)
cam_entity.name = "camera"
cam_entity.set_pose(
    sapien.Pose(
        [-0.0379048, -0.0170053, 0.0775995], [0.960526, -0.0394655, 0.19271, 0.196709]
    )
)
scene.add_entity(cam_entity)

######## Create system ########

ipc_system_config = IPCSystemConfig()
# ipc_system_config.time_step = 1e-4
ipc_system_config.time_step = 0.025
ipc_system_config.d_hat = 1e-4
ipc_system_config.eps_d = 0
ipc_system_config.preprocess_every_newton_step = True
ipc_system_config.v_max = 1e-1
ipc_system_config.clip_velocity = True
ipc_system_config.kappa = 1e3
ipc_system_config.kappa_affine = 1e3
ipc_system_config.kappa_con = 1e5
ipc_system_config.precondition = "jacobi"
ipc_system_config.newton_max_iters = 10
ipc_system_config.cg_max_iters = 40
ipc_system_config.cg_error_frequency = 40
ipc_system_config.cg_error_tolerance = 0
ipc_system_config.ccd_slackness = 0.7
ipc_system_config.ccd_thickness = 0.0
ipc_system_config.ccd_tet_inversion_thres = 0.0
ipc_system_config.allow_self_collision = False
ipc_system_config.max_blocks = 10000
ipc_system_config.debug = True
# ipc_system_config.use_graph = False
ipc_system_config.device = wp_device
# ipc_system_config.spd_project_max = 0

# ipc_system_config.enable_friction = False
# ipc_system_config.enable_collision = False

ipc_system = IPCSystem(ipc_system_config)
scene.add_system(ipc_system)

sensor_friction = 4.0
abd_friction = 0.0

sensor_color = [0.3, 0.9, 0.9, 1.0]  # cyan
hole_color = [1.0, 1.0, 0.3, 1.0]  # yellow
peg_color = [1.0, 0.3, 1.0, 1.0]


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
    sapien.render.RenderMaterial(base_color=peg_color)
)
# peg_render_component.set_data_source(peg_component)
peg_entity.add_component(peg_render_component)

hole_component = IPCABDComponent()
hole_component.set_tet_mesh(
    IPCTetMesh(filename=os.path.join(assets_dir, "concave1_hole_2.0mm_tet.msh"))
)
hole_component.set_density(500.0)
hole_component.set_friction(abd_friction)
hole_entity = sapien.Entity()
hole_entity.add_component(hole_component)

hole_render_component = sapien.render.RenderCudaMeshComponent(
    hole_component.tet_mesh.n_vertices, hole_component.tet_mesh.n_surface_triangles
)
hole_render_component.set_vertex_count(hole_component.tet_mesh.n_vertices)
hole_render_component.set_triangle_count(hole_component.tet_mesh.n_surface_triangles)
hole_render_component.set_triangles(hole_component.tet_mesh.surface_triangles)
hole_render_component.set_material(
    sapien.render.RenderMaterial(base_color=hole_color)
)
# hole_render_component.set_data_source(hole_component)
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
peg_entity.set_pose(sapien.Pose(p=[-0.0, 0.005, 0.021], q=[1.0, 0.0, 0.0, 0.0]))

scene.add_entity(peg_entity)
# peg_component.load_state_dict(peg_dict)


######## Add Sensors ########

# p10 = [x_offset - 20e-3, y_offset, hole_height + peg_height + 1e-3 - 5e-3]
# p20 = [x_offset - 17.5e-3, y_offset, hole_height + peg_height + 1e-3 - 5e-3]
# p30 = [x_offset - 17.5e-3, 5e-3, hole_height + peg_height + 1e-3 - 5e-3]
# p40 = [x_offset - 17.5e-3, 0., hole_height + peg_height + 1e-3 - 20e-3]

# sensor_1_keyframes = [
#     (0.0, p10, [ 0.5, 0.5, 0.5, 0.5 ]),
#     (0.5, p20, [ 0.5, 0.5, 0.5, 0.5 ]),
#     (2.0, p30, [ 0.5, 0.5, 0.5, 0.5 ]),
#     (3.5, p40, [ 0.5, 0.5, 0.5, 0.5 ]),
# ]

# sensor_2_keyframes = [
#     (0.0, [x_offset * 2 - p10[0], p10[1], p10[2]], [ 0.5, 0.5, -0.5, -0.5 ]),
#     (0.5, [x_offset * 2 - p20[0], p20[1], p20[2]], [ 0.5, 0.5, -0.5, -0.5 ]),
#     (2.0, [x_offset * 2 - p30[0], p30[1], p30[2]], [ 0.5, 0.5, -0.5, -0.5 ]),
#     (3.5, [x_offset * 2 - p40[0], p40[1], p40[2]], [ 0.5, 0.5, -0.5, -0.5 ]),
# ]

sensor_1_keyframes = [
    (0.0, [-0.0171, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (0.1, [-0.0165, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (2.0, [-0.0165, -0.006, 0.066], [0.70710677, 0.0, 0.70710677, 0.0]),
    (4.0, [-0.0165, -0.006, 0.05], [0.70710677, 0.0, 0.70710677, 0.0]),
]

sensor_2_keyframes = [
    (0.0, [0.0171, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (0.1, [0.0165, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (2.0, [0.0165, -0.006, 0.066], [0.0, -0.70710677, 0.0, 0.70710677]),
    (4.0, [0.0165, -0.006, 0.05], [0.0, -0.70710677, 0.0, 0.70710677]),
]

# generate trajectory


def interpolate_keyframes(keyframes, time_step):
    p_traj = []
    q_traj = []
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

    return np.array(p_traj), np.array(q_traj)


sensor_1_p_traj, sensor_1_q_traj = interpolate_keyframes(
    sensor_1_keyframes, ipc_system_config.time_step
)
sensor_2_p_traj, sensor_2_q_traj = interpolate_keyframes(
    sensor_2_keyframes, ipc_system_config.time_step
)

######## Tactile Sensor 1 ########

sensor_1_cpnt = IPCFEMComponent()
sensor_1_cpnt.set_tet_mesh(IPCTetMesh(filename=os.path.join(assets_dir, "gel.msh")))
sensor_1_cpnt.set_material(1e3, 1e5, 0.4)
sensor_1_cpnt.set_friction(sensor_friction)

sensor_1_r_cpnt = sapien.render.RenderCudaMeshComponent(
    sensor_1_cpnt.tet_mesh.n_vertices, sensor_1_cpnt.tet_mesh.n_surface_triangles
)
sensor_1_r_cpnt.set_vertex_count(sensor_1_cpnt.tet_mesh.n_vertices)
sensor_1_r_cpnt.set_triangle_count(sensor_1_cpnt.tet_mesh.n_surface_triangles)
sensor_1_r_cpnt.set_triangles(sensor_1_cpnt.tet_mesh.surface_triangles)
sensor_1_r_cpnt.set_material(
    sapien.render.RenderMaterial(base_color=sensor_color)
)
# sensor_1_r_cpnt.set_data_source(sensor_1_cpnt)

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


######## Tactile Sensor 2 ########

sensor_2_cpnt = IPCFEMComponent()
sensor_2_cpnt.set_tet_mesh(IPCTetMesh(filename=os.path.join(assets_dir, "gel.msh")))
sensor_2_cpnt.set_material(1e3, 1e5, 0.4)
sensor_2_cpnt.set_friction(sensor_friction)

sensor_2_r_cpnt = sapien.render.RenderCudaMeshComponent(
    sensor_2_cpnt.tet_mesh.n_vertices, sensor_2_cpnt.tet_mesh.n_surface_triangles
)
sensor_2_r_cpnt.set_vertex_count(sensor_2_cpnt.tet_mesh.n_vertices)
sensor_2_r_cpnt.set_triangle_count(sensor_2_cpnt.tet_mesh.n_surface_triangles)
sensor_2_r_cpnt.set_triangles(sensor_2_cpnt.tet_mesh.surface_triangles)
sensor_2_r_cpnt.set_material(
    sapien.render.RenderMaterial(base_color=sensor_color)
)
# sensor_2_r_cpnt.set_data_source(sensor_2_cpnt)

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


######## Render ########

viewer = Viewer()
viewer.set_scene(scene)
viewer.set_camera_pose(
    sapien.Pose(
        [-0.0669768, -0.0243881, 0.0526962], [0.95626, -0.028899, 0.101262, 0.272907]
    )
)
viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)

# input("Press Enter to start simulation...")

output_dir = os.path.join(os.path.dirname(__file__), "output/example_peg")
os.makedirs(output_dir, exist_ok=True)

step = 0
viewer.paused = True
scene.update_render()
ipc_update_render_all(scene)
viewer.render()

while not viewer.closed:
# for _ in range(1000):

    T_1 = sapien.Pose(
        p=sensor_1_p_traj[min(step, len(sensor_1_p_traj) - 1)],
        q=sensor_1_q_traj[min(step, len(sensor_1_p_traj) - 1)],
    ).to_transformation_matrix()
    T_2 = sapien.Pose(
        p=sensor_2_p_traj[min(step, len(sensor_2_p_traj) - 1)],
        q=sensor_2_q_traj[min(step, len(sensor_2_p_traj) - 1)],
    ).to_transformation_matrix()

    sensor_1_cpnt.set_kinematic_target(
        sensor_1_dbc_indices, sensor_1_dbc_vertices @ T_1[:3, :3].T + T_1[:3, 3]
    )
    sensor_2_cpnt.set_kinematic_target(
        sensor_2_dbc_indices, sensor_2_dbc_vertices @ T_2[:3, :3].T + T_2[:3, 3]
    )

    hole_component.set_kinematic_target_pose(sapien.Pose())

    ipc_system.step()
    scene.update_render()
    ipc_update_render_all(scene)
    viewer.render()

    # cam.take_picture()
    # rgba = cam.get_picture("Color")
    # rgba = np.clip(rgba, 0, 1)[:, :, :3]
    # rgba = Image.fromarray((rgba * 255).astype(np.uint8))
    # rgba.save(os.path.join(output_dir, f'step_{step:04d}.png'))

    step += 1

    # image = viewer.window.get_picture(viewer.render_target)
    # cv2.imwrite(
    #     os.path.join(output_dir, f"step_{step:04d}.png"), image[:, :, 2::-1] * 255
    # )

    # time.sleep(0.1)

    # np.set_printoptions(threshold=np.inf)
    # print(f"system._get_dense_matrix_brute_force().diagonal() =\n" + str(ipc_system._get_dense_matrix_brute_force().diagonal()))
    # print(f"system.cg_solver.A_diag =\n" + str(1.0 / ipc_system.cg_solver.A_diag.numpy()[:ipc_system.n_particles].reshape(-1).astype(np.float64)))
    # exit(0)
