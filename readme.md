

---

# README - 不确定性校准仓库

本仓库包含了一些用于**不确定性校准**的脚本，主要应用于**缺陷检测**和**分类任务**。这些方法利用了**集成模型**、**贝叶斯优化**、**MC Dropout**、**Fusion版本**和**温度缩放（TS）**等技术来提高模型性能，尤其是在软标签场景下的效果。

### 项目概述

该仓库包含多个用于**软标签生成**和**模型评估**的组件，适用于黑盒和软标签测试集的推理与模型校准。它还提供了基于**集成方法**和**贝叶斯加权策略**来优化和校准模型预测的工具。

### 关键组件

* **annotate_pairs_gradio.py**: 使用 Gradio 构建的一个网页应用，用于标注软标签数据对。
* **baseline_infer_singlepad.py**: 在黑盒测试集上使用原始基线模型进行推理。
* **deepensemble_infer_singlepad.py**: 使用深度集成方法（多个模型进行预测）进行推理。
* **laplace_infer_singlepad.py**: 使用拉普拉斯近似方法进行推理。
* **mcdropout_infer_singlepad.py**: 使用蒙特卡洛（MC）Dropout方法进行不确定性估计并进行推理。
* **fusion_version_singlepad.py**: 使用不同模型版本的融合方法进行推理。
* **ts_gridsearch_singlepad.py**: 在黑盒测试集上和软标签测试集上执行网格搜索，找到最佳的全局温度和按类温度。
* **checked_sample.csv/calibrate_withsoftlabels.csv**: 包含需要标注的图像对的 CSV 文件。

### 使用说明

#### 1. 修改模型配置

每个脚本中需要修改的配置文件路径不同，用户需要根据自己的环境来修改。以下是各个脚本中需要特别配置的部分：

##### 1.1 **annotate_pairs_gradio.py**

此脚本用于标注软标签数据对。


```

##### 1.2 **baseline_infer_singlepad.py**

此脚本在黑盒测试集上使用原始模型进行推理。修改以下路径来确保正确加载模型和数据：

```python
# 模型检查点路径
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

# 数据路径
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    # 如需固定为某个 ckpt，可以在这里覆盖：
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_v0.18.9lhf1certainlut05cp05clean20.0j0.4lr0.025nb256nm256dual2top2.pth.tar"
    )

```

##### 1.3 **deepensemble_infer_singlepad.py**

此脚本使用深度集成方法进行推理。需要配置模型检查点路径：

```python
# 模型检查点路径
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

# 数据路径
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

# 集成模型检查点
ENSEMBLE_CKPT_PATHS = [
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestacc.pth.tar",
    # 其他模型检查点...
]
```

##### 1.4 **laplace_infer_singlepad.py**

此脚本使用拉普拉斯近似方法进行推理。配置与其他推理脚本相似：

```python
# 模型检查点路径
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

# 数据路径
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    # 如需固定为某个 ckpt，可以在这里覆盖：
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_v0.18.9lhf1certainlut05cp05clean20.0j0.4lr0.025nb256nm256dual2top2.pth.tar"
    )

```

##### 1.5 **mcdropout_infer_singlepad.py**

此脚本使用MC Dropout方法进行推理。配置与其他推理脚本类似：

```python
# 模型检查点路径
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

# 数据路径
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

# MCDropout的检查点路径
def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/bayesian_weighted_ensemble_acc0.9028.pth.tar"
    )
    print(f"=> 为 {part_name} 加载 CNN 模型: {ckpt_path}")

    model = load_mpb3_ckpt_auto(
        ckpt_path=ckpt_path,
        backbone_arch=cfg["backbone"],
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_type="dual2",
        device=DEVICE,
    )
    return model

```

##### 1.6 **fusion_version_singlepad.py**

此脚本使用融合版本方法进行推理。你需要修改以下路径来配置模型检查点列表：

```python
# 手动配置的模型检查点列表（如果想用模式搜索，替换此列表）
ENSEMBLE_CKPT_PATHS = [
    "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestacc.pth.tar",
    # 其他模型检查点...
]

```

##### 1.7 **ts_gridsearch_singlepad.py**

此脚本用于在黑盒和软标签测试集上执行温度缩放的网格搜索。确保设置以下路径：

```python
# 模型检查点路径
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

# 数据路径
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")
def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/bayesian_weighted_ensemble_acc0.9028.pth.tar"
    )
    print(f"=> 为 {part_name} 加载 CNN 模型: {ckpt_path}")

    model = load_mpb3_ckpt_auto(
        ckpt_path=ckpt_path,
        backbone_arch=cfg["backbone"],
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_type="dual2",
        device=DEVICE,
    )
    return model

```

### 配置模型和数据路径

所有脚本中，你需要修改以下路径配置，确保它们指向正确的位置：

```python
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"
ROOT_DIR = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")
```

### 其他模型特定配置

* **集成模型的检查点路径（Deep Ensemble）**:

  ```python
  ENSEMBLE_CKPT_PATHS = [
      "/home/cat/workspace/vlm/scripts/models/checkpoints/p2/singlepadfcdropoutmobilenetv3largers6464s8c3val0.1b256_ckp_v0.18.9lhf1certainretrainlut05cp05clean20.0j0.4lr0.1nb256nm256dual2bestacc.pth.tar",
      # 其他集成模型检查点...
  ]
  ```

* **MCDropout模型检查点路径**:

  ```python
  ckpt_path = "/home/cat/workspace/vlm/scripts/models/checkpoints/bayesian_weighted_ensemble_acc0.9028.pth.tar"
  ```

### 安装依赖

1. Python 3.x
2. 使用 `pip` 安装所需的依赖：

   ```bash
   pip install -r requirements.txt
   ```

### 结论

该仓库提供了几种用于不确定性校准和软标签生成的方法，使用不同的模型架构。重点是通过集成学习、蒙特卡洛 Dropout、拉普拉斯近似和温度缩放来提高模型性能。在运行脚本之前，请确保正确配置您的模型和数据路径。

