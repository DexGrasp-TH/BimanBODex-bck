import numpy as np

import torch
import trimesh as tm
import yaml
import os
import json
import sys
from pathlib import Path
from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

# Get parent of parent
parent_parent = Path(__file__).resolve().parents[2]
sys.path.append(str(parent_parent))
from visualizer import Visualizer


if __name__ == "__main__":
    robot_urdf_path = "src/curobo/content/assets/robot/shadow_hand/dual_dummy_arm_shadow.urdf"
    mesh_dir_path = os.path.dirname(robot_urdf_path)
    visualizer = Visualizer(robot_urdf_path=robot_urdf_path, mesh_dir_path=mesh_dir_path)

    grasp_file_path = "src/curobo/content/assets/output/sim_dual_dummy_arm_shadow/tabletop_full/local_debug5/graspdata/mujoco_Perricone_MD_Chia_Serum/tabletop_ur10e/scale020_pose004_0_grasp.npy"

    grasp_data = np.load(os.path.join(grasp_file_path), allow_pickle=True).item()
    scene_path = str(grasp_data["scene_path"][0])
    scene_data = np.load(scene_path, allow_pickle=True).item()

    joint_names = grasp_data["joint_names"]
    obj_name = scene_data["task"]["obj_name"]
    obj_pose = scene_data["scene"][obj_name]["pose"]
    obj_scale = scene_data["scene"][obj_name]["scale"]
    obj_mesh_path = scene_data["scene"][obj_name]["file_path"]
    obj_mesh_path = os.path.abspath(os.path.join(os.path.dirname(scene_path), obj_mesh_path))

    # object mesh
    obj_transform = posQuat2Isometry3d(obj_pose[:3], quatWXYZ2XYZW(obj_pose[3:]))
    obj_mesh = tm.load_mesh(obj_mesh_path, process=False)
    obj_mesh = obj_mesh.copy().apply_scale(obj_scale)
    obj_mesh.apply_transform(obj_transform)

    # ################### Visualize pregrasp, grasp, and squeeze pose ###################
    # for grasp_idx in range(20):
    #     n_dof = grasp_data["robot_pose"].shape[-1]
    #     n_step = grasp_data["robot_pose"].shape[-2]
    #     n_grasp = grasp_data["robot_pose"].shape[-3]
    #     grasp_qpos = grasp_data["robot_pose"][:, grasp_idx, :, :].reshape(n_step, n_dof)
    #     robot_pose = torch.cat(
    #         [torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).repeat(n_step, 1), torch.tensor(grasp_qpos)], dim=-1
    #     )
    #     visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)

    #     robot_mesh_0 = visualizer.get_robot_trimesh_data(i=0, color=[30, 119, 179, 150])
    #     robot_mesh_1 = visualizer.get_robot_trimesh_data(i=1, color=[255, 127, 13, 255])
    #     robot_mesh_2 = visualizer.get_robot_trimesh_data(i=2, color=[44, 160, 44, 150])

    #     axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
    #     scene = tm.Scene(geometry=[robot_mesh_0, robot_mesh_1, robot_mesh_2, obj_mesh, axis])
    #     scene.show(smooth=False)

    ################# Visualize optimization process ###################

    # for grasp_idx in [1]:
    for grasp_idx in range(grasp_data["robot_pose"].shape[1]):
        n_dof = grasp_data["robot_pose"].shape[-1]
        n_step = grasp_data["robot_pose"].shape[-2]
        n_grasp = grasp_data["robot_pose"].shape[-3]
        grasp_qpos = grasp_data["robot_pose"][:, grasp_idx, :, :].reshape(n_step, n_dof)
        robot_pose = torch.cat(
            [torch.tensor([[0, 0, 0, 1, 0, 0, 0]]).repeat(n_step, 1), torch.tensor(grasp_qpos)], dim=-1
        )
        visualizer.set_robot_parameters(robot_pose, joint_names=joint_names)

        geometry_lst = []
        axis = tm.creation.axis(origin_size=0.01, axis_length=1.0)
        geometry_lst.append(axis)
        geometry_lst.append(obj_mesh)

        # Find the contact stage switching steps
        contact_stage = grasp_data["debug_info"]["contact_stage"].reshape(n_grasp, n_step)[grasp_idx, :]
        diff = np.diff(contact_stage)
        change_indices = np.where(diff != 0)[0]
        opt_step_lst = [0] + change_indices.tolist() + [n_step - 1]

        # opt_step_lst = np.asarray(opt_step_lst)[[-1, -2]]
        opt_step_lst = np.asarray(opt_step_lst)[:-1]
        # opt_step_lst = np.asarray(opt_step_lst)[[-2, -1]]

        for i_opt, opt_step in enumerate(opt_step_lst):
            alpha = 1.0 * (i_opt + 1) / len(opt_step_lst)
            robot_mesh = visualizer.get_robot_trimesh_data(i=opt_step, color=[0.941, 0.502, 0.502, alpha])
            geometry_lst.append(robot_mesh)

        scene = tm.Scene(geometry=geometry_lst)
        scene.show(smooth=False)
