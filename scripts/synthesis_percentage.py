#!/usr/bin/env python3
import sys
import yaml
from pathlib import Path
from datetime import datetime

# 获取当前时间
now = datetime.now()
# 或格式化输出，例如：2025-08-11 11:15:30
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# 从命令行参数获取路径
if len(sys.argv) < 3:
    raise ValueError("Usage: python synthesis_percentage.py <config_path> <exp_name>")
config_path = sys.argv[1]
exp_name = sys.argv[2]

# 加载对应的yaml配置文件
yaml_path = Path(f"src/curobo/content/configs/manip/{config_path}.yml")
if yaml_path.exists():
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    object_scale_list = config.get("world", {}).get("object_scale_list")
else:
    object_scale_list = None

# 输入和输出根目录
input_root = Path("src/curobo/content/assets/object/DGN_2k/scene_cfg")
output_root = Path(f"src/curobo/content/assets/output/{config_path}/{exp_name}/graspdata")

# 找到所有输入文件
input_files = list(input_root.rglob("*.npy"))

# 根据 object_scale_list 过滤
if object_scale_list is not None:
    scale_patterns = [f"scale{int(s * 100):03d}_" for s in object_scale_list]
    input_files = [f for f in input_files if any(sp in f.name for sp in scale_patterns)]

total = len(input_files)
done = 0
missing_files = []

for input_file in input_files:
    # 相对路径（去掉 input_root 前缀）
    rel_path = input_file.relative_to(input_root)
    
    # 输出文件路径
    output_file = output_root / rel_path.parent / f"{input_file.stem}_grasp.npy"
    
    if output_file.is_file():
        done += 1
    else:
        missing_files.append(output_file)

# 打印统计结果
if total > 0:
    percent = (done / total) * 100
    print(f"总文件数: {total}")
    print(f"已完成: {done}")
    print(f"完成百分比: {percent:.2f}%")
else:
    print("未找到输入文件")

# # 如果有缺失文件，可以打印出来
# if missing_files:
#     print("\n缺失文件清单（前20个示例）:")
#     for f in missing_files[:20]:
#         print(f"  {f}")

# ------------------------------------------------------------------------------------------
# 找到所有输入文件
input_files = list(input_root.rglob("*.npy"))

# 提取 object 名称集合
objects = {f.parts[len(input_root.parts)] for f in input_files}
total_objects = len(objects)

done_objects = set()

for obj in objects:
    output_obj_dir = output_root / obj
    if output_obj_dir.exists():
        # 检查该 object 是否有至少一个文件
        if any(output_obj_dir.rglob("*.npy")):
            done_objects.add(obj)

done_count = len(done_objects)

print("--------------------")
# 统计结果
if total_objects > 0:
    percent = (done_count / total_objects) * 100
    print(f"总 Object 数量: {total_objects}")
    print(f"已完成: {done_count}")
    print(f"完成百分比: {percent:.2f}%")
else:
    print("未找到输入文件")

# # 输出未完成的 object
# missing_objects = objects - done_objects
# if missing_objects:
#     print("\n未完成的 Object（前20个示例）:")
#     for obj in sorted(missing_objects)[:20]:
#         print(f"  {obj}")
