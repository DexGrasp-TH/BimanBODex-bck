# BimanBODex

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
    pip install mujoco==3.3.2

    pip install -e ./third_party/pytorch_kinematics
    pip install -e ./third_party/utils_python
    ```

3. **Prepare object assets**: Download our pre-processed object assets `DGN_2k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex) and organize the unzipped folders as below.  (A symbolic link is recommended.)
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
   


## Usage

### Robot Asserts Preparation

1. Assemble the `<hand>.urdf` and `dual_dummy_arm_<hand>.urdf` via `robot_assets`.
1. Copy the urdf and meshes files into `src/curobo/content/assets/robot`.
1. Create a robot config.yml in `src/curobo/content/configs/robot`.
    1. When setting the collision_spheres, you can use `scripts/vis_collision_spheres.py` to help visualize and manually adjust the collision spheres.
    1. When setting the `hand_pose_transfer` parameters, you can use `scripts/vis_hand_pose_transfer.py` to visualize the hand base frame and palm frame. The frame adjustment is centralized in `adjust_palm_frame()`, and pressing Key `s` in the trimesh viewer will save the adjusted `hand_base_link` parameters back to the YAML file.


### Single-Type Synthesis

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

### Multi-Type Synthesis of One Hand

Batch run all grasp types on multiple GPUs:
```bash
./scripts/run_all_grasps_multi_gpu.sh --hand <shadow|leap> --exp-name <EXP_NAME> --parallel-env <NUM_PARALLEL_ENV> --gpus <GPU_ID...> [-k] [--start START] [--end END]
# Examples:
# ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name minitest --parallel-env 10 --gpus 0 1 2 3 -k --start 0 --end 10
# ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name dataset_1k --parallel-env 40 --gpus 4 5 6 7 --start 0 --end 1000
# ./scripts/run_all_grasps_multi_gpu.sh --hand leap --exp-name dataset_full --parallel-env 300 --gpus 0 1 2 3
```

* `shadow` and `leap` select the config family to run. 
* `-k` disables skipping existing files, matching `example_grasp/multi_gpu.py`.
* `START` and `END` are optional overrides for `world.start` and `world.end` in the manipulation YAML files such as `src/curobo/content/configs/manip/sim_shadow/tabletop_full.yml`. If omitted, the values from each config file are used.

Analyze grasp counts produced by the multi-GPU batch run:
```bash
python scripts/analyze_multi_gpu_grasp_counts.py --hand <shadow|leap> --exp-name <EXP_NAME> [--start START] [--end END] [--show-missing N]
# Examples:
# python scripts/analyze_multi_gpu_grasp_counts.py --hand leap --exp-name minitest --start 0 --end 10
```

* This checks expected output files under `src/curobo/content/assets/output/.../graspdata` for the same config family used by `scripts/run_all_grasps_multi_gpu.sh`.
* `--start` and `--end` should match any range override used during generation so the expected file count lines up with the run.
* `--show-missing N` prints up to `N` missing output paths per config to help identify unfinished scenes.
