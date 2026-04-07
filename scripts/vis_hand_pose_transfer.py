"""
Visualize hand pose transfer frames for a robot hand configuration.

This script:
- loads the robot mesh and the hand pose transfer YAML
- visualizes the hand base frame(s) named in the hand pose transfer YAML and the transferred palm frame(s)
- applies a hand-specific frame adjustment through `adjust_palm_frame`
- prints the adjusted palm rotation matrix and translation vector
- saves the adjusted hand base link parameters back to the YAML when `s` is pressed in the viewer
"""

import os

import numpy as np
import torch
import trimesh as tm
import yaml
from scipy.spatial.transform import Rotation as R
from trimesh.viewer import SceneViewer
from trimesh.viewer.windowed import pyglet

from visualizer import Visualizer


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def make_axis_mesh(transform, axis_length=0.04, origin_size=0.006):
    axis_mesh = tm.creation.axis(origin_size=origin_size, axis_length=axis_length)
    axis_mesh.apply_transform(transform)
    return axis_mesh


def rotation_from_euler_y(angle_deg):
    return R.from_euler("y", angle_deg, degrees=True).as_matrix()


def adjust_palm_frame(rotation, translation, link_name=None):
    """
    Adjust the loaded palm frame for visualization or saving.

    Update this function for hand-specific frame conventions.
    """
    del link_name
    adjusted_rotation = rotation
    adjusted_translation = translation
    return adjusted_rotation, adjusted_translation


def format_number(value):
    return str(float(value))


def format_vector(values):
    return "[" + ", ".join(format_number(v) for v in values) + "]"


def format_matrix(rows):
    return "[" + ", ".join(format_vector(row) for row in rows) + "]"


def dump_hand_pose_transfer_inline(path, hand_pose_transfer):
    header = [
        "# transfer_trans defines the transformation from palm frame to hand base frame",
        "#",
        "# Palm frame (sampled from object surface):",
        "#   - x-axis: points towards object (from palm)",
        "#   - y-axis: points towards thumb (right hand) or away from thumb (left hand)",
        "#   - z-axis: points towards four primary fingers",
        "#",
        "# Hand base frame: defined by the hand URDF",
        "",
    ]
    body = []
    for link_name, transfer_cfg in hand_pose_transfer.items():
        body.append(f"{link_name}:")
        body.append(f"  r: {format_matrix(transfer_cfg['r'])}")
        body.append(f"  t: {format_vector(transfer_cfg['t'])}")
        body.append("")

    text = "\n".join(header + body).rstrip() + "\n"
    with open(path, "w") as file:
        file.write(text)


if __name__ == "__main__":
    """
    Visualize the hand pose transfer frames of the robot.
    """

    # ------------- create visualizer -------------
    # robot_urdf_path = "src/curobo/content/assets/robot/leap_hand/leap.urdf"
    # mesh_dir_path = "src/curobo/content/assets/robot/leap_hand"
    # robot_config_path = "src/curobo/content/configs/robot/leap_hand.yml"
    # hand_pose = torch.zeros((1, 3 + 4 + 16))

    robot_urdf_path = "src/curobo/content/assets/robot/leap_hand/dual_dummy_arm_leap.urdf"
    mesh_dir_path = "src/curobo/content/assets/robot/leap_hand"
    robot_config_path = "src/curobo/content/configs/robot/dual_dummy_arm_leap.yml"
    hand_pose = torch.zeros((1, 3 + 4 + 6 + 6 + 16 + 16))

    visualize = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    hand_pose[:, 3] = 1.0  # quat w
    # hand_pose[:, 7:] = torch.rand_like(hand_pose[:, 7:])

    visualize.set_robot_parameters(hand_pose)
    robot_mesh = visualize.get_robot_trimesh_data(i=0, color=[0, 255, 0, 100])
    scene_caption = (
        "Large axes: hand base frames from hand_pose_transfer | "
        "Smaller axes: palm frames | "
        "Press 's' to save adjusted hand base link parameters."
    )

    with open(robot_config_path, "r") as file:
        robot_config = yaml.safe_load(file)

    kinematics_cfg = robot_config["robot_cfg"]["kinematics"]
    world_t_robot = hand_pose[0, 0:3].cpu().numpy()
    world_r_robot = visualize.global_rotation[0].cpu().numpy()
    world_t_robot_tf = make_transform(world_r_robot, world_t_robot)
    axis_meshes = []

    hand_pose_transfer_path = os.path.join(
        "src/curobo/content/configs/robot", kinematics_cfg["hand_pose_transfer_path"]
    )
    with open(hand_pose_transfer_path, "r") as file:
        hand_pose_transfer = yaml.safe_load(file)

    adjusted_hand_base_links = {}

    # Each entry stores the palm frame pose in the corresponding hand base frame.
    for link_name, transfer_cfg in hand_pose_transfer.items():
        hand_base_tf = visualize.current_status[link_name].get_matrix()[0].cpu().numpy()
        world_t_hand_base = world_t_robot_tf @ hand_base_tf
        axis_meshes.append(make_axis_mesh(world_t_hand_base, axis_length=0.06, origin_size=0.008))

        palm_r_in_base = np.asarray(transfer_cfg["r"], dtype=np.float64)
        palm_t_in_base = np.asarray(transfer_cfg["t"], dtype=np.float64)
        adjusted_palm_r_in_base, adjusted_palm_t_in_base = adjust_palm_frame(
            palm_r_in_base, palm_t_in_base, link_name=link_name
        )
        palm_tf_in_base = make_transform(adjusted_palm_r_in_base, adjusted_palm_t_in_base)
        world_t_palm = world_t_hand_base @ palm_tf_in_base

        print(f"{link_name} adjusted palm rotation matrix:")
        print(adjusted_palm_r_in_base)
        print(f"{link_name} adjusted palm translation vector:")
        print(adjusted_palm_t_in_base)

        axis_meshes.append(make_axis_mesh(world_t_palm, axis_length=0.04, origin_size=0.006))

        palm_origin = tm.creation.icosphere(subdivisions=3, radius=0.004)
        palm_origin.apply_translation(world_t_palm[:3, 3])
        palm_origin.visual.face_colors = [255, 255, 0, 255]
        axis_meshes.append(palm_origin)

        adjusted_hand_base_links[link_name] = {
            "r": adjusted_palm_r_in_base.tolist(),
            "t": adjusted_palm_t_in_base.tolist(),
        }

    scene = tm.Scene(geometry=[robot_mesh] + axis_meshes)

    viewer = SceneViewer(scene, caption=scene_caption, start_loop=False)
    original_on_key_press = viewer.on_key_press

    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.S and adjusted_hand_base_links:
            hand_pose_transfer.update(adjusted_hand_base_links)
            dump_hand_pose_transfer_inline(hand_pose_transfer_path, hand_pose_transfer)
            print(f"Saved adjusted hand base link parameters to {hand_pose_transfer_path}")
            return
        original_on_key_press(symbol, modifiers)

    viewer.on_key_press = on_key_press
    pyglet.app.run()
