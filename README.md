# BimanBODex

A GPU-based efficient pipeline for dexterous grasp synthesis built on [cuRobo](https://github.com/NVlabs/curobo/tree/main), proposed in *[ICRA 2025] BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization*.

[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490) ｜ [Dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) ｜ [Benchmark code](https://github.com/JYChen18/DexGraspBench) ｜ [Learning code](https://github.com/JYChen18/DexLearn)

## Introduction
### Main Features
- **Grasp Synthesis**: Generate force-closure grasps for floating dexterous hands, such as the Shadow, Allegro, and Leap Hand.
- **Trajectory Planning**: Plan collision-free approaching trajectories with the table for hands mounted on robotic arms, e.g., UR10e + Shadow Hand systems.

### Highlights
- **Efficient**: Capable of synthesizing millions of grasps per day using a single NVIDIA 3090 GPU.
- **Generalizable**: Supports different hands and a wide range of objects.

### Follow-up Work
Some projects *make modifications* on our BODex pipeline to synthesize large-scale datasets of grasping poses, such as [DexGraspNet 2.0](https://pku-epic.github.io/DexGraspNet2.0/) and [GraspVLA](https://pku-epic.github.io/GraspVLA-web/).

Looking for more diverse and higher-quality dexterous grasps? Check out [Dexonomy](https://pku-epic.github.io/Dexonomy).



## Getting Started
1. **Install git lfs**: Before `git clone` this repository, please make sure that the git lfs has been installed by `sudo apt install git-lfs`.

2. **Clone with submodules**:
```bash
git clone --recurse-submodules git@github.com:DexGrasp-TH/BimanBODex.git
# Or if already cloned:
git submodule update --init --recursive
```

3. **Install the Python environment**:
```bash
conda create -n bibodex python=3.10
conda activate bibodex

conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

pip install mkl==2024.0.0

pip install -e . --no-build-isolation  

pip install usd-core 
conda install pytorch-scatter -c pyg -y
conda install coal==3.0.1 -c conda-forge -y   # https://github.com/coal-library/coal

pip uninstall numpy
pip install numpy==1.26.4

cd src/curobo/geom/cpp
python setup.py install    # install coal_openmp_wrapper

# by mingrui
pip install 'pyglet<2'
pip install hydra-core
pip install pyrender
pip install mujoco

cd third_party/pytorch_kinematics
pip install -e .

cd ../utils_python
pip install -e .
```

3. **Prepare object assets**: Download our pre-processed object assets `DGN_2k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex) and organize the unzipped folders as below. 
```
src/curobo/content/assets/object/DGN_2k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```
Alternatively, new object assets can be pre-processed using [MeshProcess](https://github.com/JYChen18/MeshProcess).
   
4. **Synthesize grasp poses**: Each synthesized grasping data point includes a pre-grasp, a grasp, and a squeeze pose. 
```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40 

# Debugging. The saved USD file has the whole optimization process, while the intermediate gradient on each contact point is denoted by the purple line.
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 1 -m usd -debug -d all -i 0 1

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/fc.yml -t grasp -g 0 1 2 3 
```
We can also **synthesize approaching trajectories** that are collision-free with the table for hands mounted on arms, e.g., UR10e+Shadow Hand systems.
 ```
# Single GPU version
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_mogen_batch.py -c sim_shadow/tabletop.yml -t grasp_and_mogen

# Multiple GPU version
python example_grasp/multi_gpu.py -c sim_shadow/tabletop.yml -t grasp_and_mogen -g 0 1 2 3 
```
On a single GPU, the grasp synthesis supports parallizing different objects, but the motion planning only supports parallizing different trajectories for the same object.

5. **(Optional) Visualize synthesized poses**:
```
python example_grasp/visualize_npy.py -c sim_shadow/fc.yml -p debug -m grasp
```

6. **Evaluate grasp poses and filter out bad ones**: please see [DexGraspBench](https://github.com/JYChen18/DexGraspBench).

## Mingrui Usage

### Robot Asserts Preparation

1. Assemble the `<hand>.urdf` and `dual_dummy_arm_<hand>.urdf` via `robot_assets`.
1. Copy the urdf and meshes files into `src/curobo/content/assets/robot`.
1. Create a robot config.yml in `src/curobo/content/configs/robot`.
    1. When setting the collision_spheres, you can use `scripts/vis_collision_spheres.py` to help visualize and manually adjust the collision spheres.
    1. When setting the `hand_pose_transfer` parameters, you can use `scripts/vis_hand_pose_transfer.py` to visualize the hand base frame and palm frame. The frame adjustment is centralized in `adjust_palm_frame`, and pressing `s` in the trimesh viewer will save the adjusted `hand_base_link` parameters back to the YAML file.


### Synthesis of Five Grasp Types

Synthesis:
```bash
CUDA_VISIBLE_DEVICES=7 python example_grasp/plan_batch_env.py -c <ROBOT>/<GRASP_TYPE>.yml -w 100 -k --exp_name <NAME>
```

Multi-gpu synthesis:
```bash
$ CUDA_VISIBLE_DEVICES=7 python example_grasp/multi_gpu.py -c <ROBOT>/<GRASP_TYPE>.yml -w 100 -t grasp -g <GPU_ID...> --exp_name <NAME>

# Example:
# $ python example_grasp/multi_gpu.py -c sim_shadow/tabletop_full.yml -w 10 -t grasp -g 0 1 2 3 --exp_name minitest_multigpu --start 0 --end 10
```

Render:
```bash
python example_grasp/main.py task=render manip_cfg_file=<ROBOT>/<GRASP_TYPE>.yml n_worker=96 task.debug=False task.b_opt_process=<True/False> name=<EXP_NAME> 
```

* ROBOT/GRASP_TYPE:
  * sim_shadow/tabletop_two
  * sim_shadow/tabletop_three
  * sim_shadow/tabletop_full
  * sim_dual_dummy_arm_shadow/tabletop_three
  * sim_dual_dummy_arm_shadow/tabletop_full
  * sim_leap/tabletop_two
  * sim_leap/tabletop_three
  * sim_leap/tabletop_full
  * sim_dual_dummy_arm_leap/tabletop_three
  * sim_dual_dummy_arm_leap/tabletop_full


Batch run all grasp types on multiple GPUs:
```bash
./scripts/run_all_grasps_multi_gpu.sh --hand <shadow|leap> --exp-name <EXP_NAME> --parallel-env <NUM_PARALLEL_ENV> --gpus <GPU_ID...> [-k] [--start START] [--end END]
# Examples:
# ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name dataset_1k --parallel-env 40 --gpus 4 5 6 7 --start 0 --end 1000
# ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name minitest --parallel-env 10 --gpus 0 1 2 3 -k --start 0 --end 10
```

* `shadow` and `leap` select the config family to run. 
* `-k` disables skipping existing files, matching `example_grasp/multi_gpu.py`.
* `START` and `END` are optional overrides for `world.start` and `world.end` in the manipulation YAML files such as `src/curobo/content/configs/manip/sim_shadow/tabletop_full.yml`. If omitted, the values from each config file are used.
