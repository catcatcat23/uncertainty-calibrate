import os
import glob
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from torch.optim.swa_utils import AveragedModel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.utilities import TransformImage
from models.MPB3 import MPB3net

# ==== 可视化 & F1 曲线 ====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 这个目录你已经有了
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

N_CLASS = 3
N_CLASSES = 3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR = (
    "/home/cat/workspace/defect_data/"
    "defect_DA758_black_uuid_250310/send2terminal/250310"
)
CSV_PATH = os.path.join(ROOT_DIR, "checked_samples.csv")

# 顺序和 defect_code.keys() 一致：
# defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
CLASS_ORDER = ["ok", "undersolder", "pseudosolder"]

PART_CONFIG = {
    "singlepad": {
        "ckpt_pattern": "singlepadfcdropoutmobilenetv3largers6464*dual2*.pth.tar",
        "backbone": "fcdropoutmobilenetv3large",
        "n_units": [256, 256],
        "n_class": N_CLASS,
        "img_h": 64,
        "img_w": 64,
    },
    "singlepinpad": {
        "ckpt_pattern": "singlepinpadmobilenetv3smallrs12832*dual2*.pth.tar",
        "backbone": "mobilenetv3small",
        "n_units": [256, 256],
        "n_class": N_CLASS,
        "img_h": 32,
        "img_w": 128,
    },
}

# F1 曲线采样点
N_THRESH = 101
# 需要打印详细指标的阈值
THRESH_LIST = [0.5, 0.8]
# ECE bin 数
N_BINS_ECE = 15

# 评估图保存目录
SAVE_DIR = os.path.join(ROOT_DIR, "mcdropout_eval")
os.makedirs(SAVE_DIR, exist_ok=True)

# 温度缩放配置
# 全局温度搜索范围和步长
T_GLOBAL_SEARCH = np.linspace(0.1, 5.0, 50)
# 按类温度搜索范围（每个类别独立搜索）
T_PER_CLASS_SEARCH = np.linspace(0.1, 5.0, 50)
# 验证集比例（用于温度优化）
VAL_RATIO = 0.2

# 使用英文默认字体
plt.rcParams["font.family"] = "DejaVu Sans"


# ==============================
# 新增：温度缩放相关函数
# ==============================
def compute_ece_for_temperature(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = N_BINS_ECE
) -> float:
    """计算ECE，用于温度优化的目标函数"""
    y_pred = probs.argmax(axis=1)
    max_conf = probs[np.arange(len(probs)), y_pred]
    correctness = (y_pred == y_true).astype(np.float64)
    
    ece, _, _, _, _ = compute_ece(max_conf, correctness, n_bins)
    return ece

def temperature_scaling(
    logits: np.ndarray,
    temperature: float
) -> np.ndarray:
    """全局温度缩放：logits / T 后做softmax"""
    logits = np.asarray(logits, dtype=np.float64)
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))  # 防止溢出
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

def per_class_temperature_scaling(
    logits: np.ndarray,
    temperatures: np.ndarray
) -> np.ndarray:
    """按类温度缩放：每个类别使用独立的温度系数"""
    logits = np.asarray(logits, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    assert len(temperatures) == logits.shape[1], "温度系数数量必须等于类别数"
    
    scaled_logits = logits / temperatures[np.newaxis, :]
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

def grid_search_best_global_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    val_ratio: float = VAL_RATIO
) -> Tuple[float, float]:
    """网格搜索最优全局温度"""
    # 划分训练/验证集（用于温度优化）
    n_samples = len(logits)
    val_size = int(n_samples * val_ratio)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    logits_val = logits[val_indices]
    y_true_val = y_true[val_indices]
    
    best_ece = float('inf')
    best_t = 1.0
    
    print(f"\n开始全局温度网格搜索 (范围: {T_GLOBAL_SEARCH.min():.2f} ~ {T_GLOBAL_SEARCH.max():.2f})")
    for t in tqdm(T_GLOBAL_SEARCH, desc="全局温度搜索"):
        probs = temperature_scaling(logits_val, t)
        ece = compute_ece_for_temperature(probs, y_true_val)
        
        if ece < best_ece:
            best_ece = ece
            best_t = t
    
    # 用最优温度计算训练集ECE
    probs_train = temperature_scaling(logits[train_indices], best_t)
    train_ece = compute_ece_for_temperature(probs_train, y_true[train_indices])
    
    print(f"最优全局温度: T={best_t:.3f}, 验证集ECE={best_ece:.4f}, 训练集ECE={train_ece:.4f}")
    return best_t, best_ece

def grid_search_best_per_class_temperatures(
    logits: np.ndarray,
    y_true: np.ndarray,
    val_ratio: float = VAL_RATIO
) -> Tuple[np.ndarray, float]:
    """网格搜索最优按类温度（ETS）"""
    # 划分训练/验证集
    n_samples = len(logits)
    val_size = int(n_samples * val_ratio)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:val_size]
    y_true_val = y_true[val_indices]
    logits_val = logits[val_indices]
    
    n_classes = logits.shape[1]
    best_t_per_class = np.ones(n_classes)  # 初始温度为1.0
    best_ece = float('inf')
    
    print(f"\n开始按类温度网格搜索 (每个类别范围: {T_PER_CLASS_SEARCH.min():.2f} ~ {T_PER_CLASS_SEARCH.max():.2f})")
    
    # 逐个类别优化（贪心策略）
    for c in range(n_classes):
        current_best_t = 1.0
        current_best_ece = float('inf')
        
        for t in tqdm(T_PER_CLASS_SEARCH, desc=f"优化类别{c} ({CLASS_ORDER[c]})"):
            temp_t = best_t_per_class.copy()
            temp_t[c] = t
            
            probs = per_class_temperature_scaling(logits_val, temp_t)
            ece = compute_ece_for_temperature(probs, y_true_val)
            
            if ece < current_best_ece:
                current_best_ece = ece
                current_best_t = t
        
        best_t_per_class[c] = current_best_t
        best_ece = current_best_ece
        print(f"类别{c} ({CLASS_ORDER[c]}) 最优温度: {current_best_t:.3f}, ECE={current_best_ece:.4f}")
    
    # 最终验证
    final_probs = per_class_temperature_scaling(logits_val, best_t_per_class)
    final_ece = compute_ece_for_temperature(final_probs, y_true_val)
    print(f"按类温度优化完成: {best_t_per_class}, 最终验证集ECE={final_ece:.4f}")
    
    return best_t_per_class, final_ece

def get_logits_from_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """从概率反推logits（用于温度缩放）"""
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs)
    return logits


# ==============================
# MC Dropout 推理
# ==============================
def load_mpb3_ckpt_auto(ckpt_path, backbone_arch, n_class, n_units, output_type, device):
    """
    自动适配加载：
      - 普通 model.state_dict():        key 可能是 'xxx' 或 'module.xxx'
      - SWA AveragedModel.state_dict(): key 通常包含 'n_averaged' + 'module.xxx'
      - 以及一些变体：module.module.xxx
    """
    def _extract_state_dict(obj):
        if not isinstance(obj, dict):
            return obj
        # 尽量覆盖你可能遇到的保存字段名
        cand_keys = [
            "swa_state_dict",
            "swa_model_state_dict",
            "averaged_model_state_dict",
            "avg_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
        ]
        for k in cand_keys:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # 有些人直接 torch.save(model.state_dict())
        return obj

    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        # 如果至少有一部分 key 有该 prefix，就尝试整体 strip（更稳妥：只 strip 那些真的带 prefix 的）
        has = any(k.startswith(prefix) for k in sd.keys())
        if not has:
            return sd
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    def _gen_variants(sd: Dict[str, torch.Tensor]):
        # 生成几种常见前缀形态的候选
        variants = []
        seen = set()

        def _add(d):
            # 用前几个 key 做 fingerprint，避免重复
            keys = tuple(list(d.keys())[:50])
            if keys in seen:
                return
            seen.add(keys)
            variants.append(d)

        _add(sd)
        # strip 一层/两层 module.
        sd1 = _strip_prefix(sd, "module.")
        _add(sd1)
        sd2 = _strip_prefix(sd1, "module.")
        _add(sd2)

        # 有些 ckpt 可能是 "model.module.xxx"（少见，但加一层容错）
        sd3 = _strip_prefix(sd, "model.")
        _add(sd3)
        sd4 = _strip_prefix(sd3, "module.")
        _add(sd4)

        return variants

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict_raw = _extract_state_dict(ckpt)

    # 先建 base 模型
    base_model = MPB3net(
        backbone=backbone_arch,
        pretrained=False,
        n_class=n_class,
        n_units=n_units,
        output_form=output_type,
    )

    fname = os.path.basename(ckpt_path)
    is_swa = ("swa" in fname) or fname.endswith("swa.pth.tar")

    if is_swa:
        print(f"=> [SWA] loading via AveragedModel: {ckpt_path}")
        swa_model = AveragedModel(base_model)

        # 1) 先尝试把它当成“真正的 AveragedModel.state_dict()”来加载
        #    （此时一般会有 n_averaged + module.xxx）
        loaded = False
        variants = _gen_variants(state_dict_raw)

        # 优先尝试：直接给 swa_model（需要 key 形如 module.xxx）
        for sd in variants:
            try:
                # strict=True 先试一次
                swa_model.load_state_dict(sd, strict=True)
                loaded = True
                print("[SWA] loaded as AveragedModel.state_dict() (strict=True).")
                break
            except Exception:
                pass

        if not loaded:
            for sd in variants:
                try:
                    # strict=False 再试一次（有时只差一个 n_averaged）
                    missing, unexpected = swa_model.load_state_dict(sd, strict=False)
                    # 判断是否“真的对得上”：至少要有大量 module.xxx 被加载到
                    # 否则就是完全错误的 dict，不要吞掉错误
                    expected = set(swa_model.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[SWA] loaded as AveragedModel.state_dict() (strict=False), "
                              f"common_ratio={ratio:.2f}, missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        # 2) 如果上面失败，说明 ckpt 更像“普通模型的 state_dict”
        #    那就加载到 swa_model.module（也就是 base_model）
        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = swa_model.module.load_state_dict(sd, strict=True)
                    loaded = True
                    print("[SWA] ckpt looks like normal model state_dict -> loaded into swa_model.module (strict=True).")
                    break
                except Exception:
                    pass

        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = swa_model.module.load_state_dict(sd, strict=False)
                    # 同样做一个覆盖率判断，避免 silent mismatch
                    expected = set(swa_model.module.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[SWA] ckpt looks like normal model state_dict -> loaded into swa_model.module (strict=False), "
                              f"common_ratio={ratio:.2f}, missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        if not loaded:
            # 如果仍然失败，把几个 key 打出来更好定位
            first_keys = list(state_dict_raw.keys())[:30] if isinstance(state_dict_raw, dict) else []
            raise RuntimeError(
                "[SWA] 无法加载该 ckpt 到 AveragedModel 或其 module。\n"
                f"ckpt_path={ckpt_path}\n"
                f"first_keys={first_keys}\n"
                "请检查 backbone/n_units/n_class/output_type 是否与训练一致。"
            )

        # n_averaged 缺失时，给一个合理默认值
        try:
            if not hasattr(swa_model, "n_averaged") or swa_model.n_averaged is None:
                swa_model.n_averaged = torch.tensor(1, dtype=torch.long)
            else:
                swa_model.n_averaged.copy_(torch.tensor(1, dtype=torch.long))
        except Exception:
            pass

        model = swa_model  # 你想直接用 swa_model 推理就保持这一行
        # 如果你后面代码强依赖 MPB3net 的属性（如 model.cnn_encoder），用这一行替代上一行：
        # model = swa_model.module

    else:
        print(f"=> loading normal ckpt: {ckpt_path}")

        loaded = False
        variants = _gen_variants(state_dict_raw)
        for sd in variants:
            try:
                base_model.load_state_dict(sd, strict=True)
                loaded = True
                break
            except Exception:
                pass

        if not loaded:
            for sd in variants:
                try:
                    missing, unexpected = base_model.load_state_dict(sd, strict=False)
                    expected = set(base_model.state_dict().keys())
                    common = len(expected.intersection(set(sd.keys())))
                    ratio = common / max(1, len(expected))
                    if ratio >= 0.60:
                        loaded = True
                        print(f"[WARN] normal ckpt loaded with strict=False, common_ratio={ratio:.2f}, "
                              f"missing={len(missing)}, unexpected={len(unexpected)}")
                        break
                except Exception:
                    pass

        if not loaded:
            first_keys = list(state_dict_raw.keys())[:30] if isinstance(state_dict_raw, dict) else []
            raise RuntimeError(
                "[NORMAL] 无法加载该 ckpt 到 base_model。\n"
                f"ckpt_path={ckpt_path}\n"
                f"first_keys={first_keys}\n"
                "请检查 backbone/n_units/n_class/output_type 是否与训练一致。"
            )

        model = base_model

    model = model.to(device).eval()
    return model


def enable_mc_dropout(model: torch.nn.Module):
    """
    只把模型里的 Dropout 打开（保持 BatchNorm 等仍然是 eval 状态）
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def set_dropout_p(model: torch.nn.Module, p: float):
    """
    把模型里所有 Dropout 的概率改成给定的 p
    （只改 nn.Dropout / nn.Dropout2d / nn.Dropout3d）
    """
    cnt = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.p = p
            cnt += 1
    print(f"[INFO] 已将 {cnt} 个 Dropout 层的 p 设置为 {p}")


@torch.no_grad()
def mc_predict_bos_bom(
    model: torch.nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    mc_times: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    对同一批输入做多次前向，返回 BOS/BOM 的均值和标准差
    - x1, x2: [B, C, H, W]
    - 返回 dict:
        bos_mean: [B, 2]
        bos_std:  [B, 2]
        bom_mean: [B, C]
        bom_std:  [B, C]
        bom_logits_mean: [B, C]  # 新增：返回logits均值，用于温度缩放
    """
    model.eval()              # BN / 其他保持 eval
    enable_mc_dropout(model)  # 只打开 Dropout

    bos_list, bom_list, bom_logits_list = [], [], []
    for _ in range(mc_times):
        logits_bos, logits_bom = model(x1, x2)
        bos_list.append(F.softmax(logits_bos, dim=-1))
        bom_list.append(F.softmax(logits_bom, dim=-1))
        bom_logits_list.append(logits_bom)  # 保存原始logits

    bos_arr = torch.stack(bos_list, dim=0)  # [T, B, 2]
    bom_arr = torch.stack(bom_list, dim=0)  # [T, B, C]
    bom_logits_arr = torch.stack(bom_logits_list, dim=0)  # [T, B, C]

    return {
        "bos_mean": bos_arr.mean(0),
        "bos_std":  bos_arr.std(0),
        "bom_mean": bom_arr.mean(0),
        "bom_std":  bom_arr.std(0),
        "bom_logits_mean": bom_logits_arr.mean(0),  # logits均值
    }


def load_model_by_values(ckp, model):
    model_state_dict = model.state_dict()
    model_state_dict_keys = list(model_state_dict.keys())
    index = 0
    new_state_dict = {}

    for key, value in ckp.items():
        if index >= len(model_state_dict_keys):
            print(f"[WARN] ckpt 多出来的参数：{key}")
            break

        model_key = model_state_dict_keys[index]

        if key == model_key and model_state_dict[model_key].shape == value.shape:
            new_state_dict[model_key] = value
        elif model_state_dict[model_key].shape == value.shape:
            print(f"ckp 的 {key} 对应 model 的 {model_key}")
            new_state_dict[model_key] = value
        else:
            print(
                f"Key 不匹配: ckpt {key} ({value.shape}) vs model {model_key} "
                f"({model_state_dict[model_key].shape})"
            )

        index += 1

    model.load_state_dict(new_state_dict, strict=True)


def _find_ckpt(pattern: str) -> str:
    """
    在 CKPT_DIR 下用通配符找 ckpt 文件。
    """
    full_pattern = os.path.join(CKPT_DIR, pattern)
    files = glob.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配 ckpt: {full_pattern}")
    if len(files) > 1:
        print(f"[WARN] 匹配到多个 ckpt，只使用第一个：{files[0]}")
    return files[0]


def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])
    ckpt_path = (
        "/home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464s42c3val0.1b256_ckp_v0.18.9lhf1certainlut05cp05clean20.0j0.4lr0.025nb256nm256dual2top2.pth.tar"
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


def infer_singlepad_batch(
    csv_path: str,
    root_path: str,
    model: MPB3net,
    part_name: str = "singlepad",
    batch_size: int = 32,
    mc_times: int = 20,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    对 singlepad 的 CSV 里所有样本做批量 + MC Dropout 推理。
    修改：返回原始logits和真实标签，用于温度缩放
    
    返回:
        result_df: 原始推理结果
        all_logits: 所有样本的logits均值 [N, C]
        y_true: 真实标签 [N]
    """

    cfg = PART_CONFIG[part_name]
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    # 获取真实标签
    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    if all(col in df.columns for col in gt_cols):
        soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
        y_true = soft_gt.argmax(axis=1).astype(np.int64)
    else:
        y_true = None
        print("[WARN] 未找到真实标签列，无法进行温度缩放优化")

    results = []
    all_logits = []

    num_samples = len(df)
    print(
        f"[*] 一共 {num_samples} 条样本，开始 MC Dropout 推理 "
        f"(batch_size={batch_size}, mc_times={mc_times})"
    )

    with tqdm(total=num_samples, desc=f"{part_name} MC Dropout", unit="img") as pbar:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_df = df.iloc[start:end]

            img1_list, img2_list = [], []
            meta_list = []

            for _, row in batch_df.iterrows():
                ref_rel = row["ref_image"]
                insp_rel = row["insp_image"]
                sample_id = row["id"]

                ref_path = os.path.join(root_path, ref_rel)
                insp_path = os.path.join(root_path, insp_rel)

                ref_img = TransformImage(
                    img_path=ref_path,
                    rs_img_size_h=cfg["img_h"],
                    rs_img_size_w=cfg["img_w"],
                ).transform()

                insp_img = TransformImage(
                    img_path=insp_path,
                    rs_img_size_h=cfg["img_h"],
                    rs_img_size_w=cfg["img_w"],
                ).transform()

                ref_tensor = torch.FloatTensor(ref_img)
                insp_tensor = torch.FloatTensor(insp_img)

                img1_list.append(ref_tensor)
                img2_list.append(insp_tensor)

                meta_list.append({
                    "id": sample_id,
                    "ref_path": ref_rel,
                    "insp_path": insp_rel,
                })

            x1 = torch.cat(img1_list, dim=0).to(DEVICE)
            x2 = torch.cat(img2_list, dim=0).to(DEVICE)

            mc_out = mc_predict_bos_bom(model, x1, x2, mc_times=mc_times)
            bom_mean = mc_out["bom_mean"]
            bom_std  = mc_out["bom_std"]
            bom_logits_mean = mc_out["bom_logits_mean"]  # 获取logits均值

            # 保存logits
            all_logits.append(bom_logits_mean.cpu().numpy())

            bom_mean_np = bom_mean.cpu().numpy()
            bom_std_np  = bom_std.cpu().numpy()

            for i, meta in enumerate(meta_list):
                probs = bom_mean_np[i]
                stds  = bom_std_np[i]

                row_result = {
                    **meta,
                    "NONE_CONF_DROPOUT":                     float(probs[0]),
                    "INSUFFICIENT_SOLDER_CONF_DROPOUT":      float(probs[1]),
                    "PSEUDO_SOLDER_CONF_DROPOUT":            float(probs[2]),
                    "NONE_CONF_DROPOUT_STD":                 float(stds[0]),
                    "INSUFFICIENT_SOLDER_CONF_DROPOUT_STD":  float(stds[1]),
                    "PSEUDO_SOLDER_CONF_DROPOUT_STD":        float(stds[2]),
                }
                results.append(row_result)

            pbar.update(len(batch_df))

    result_df = pd.DataFrame(results)
    all_logits = np.concatenate(all_logits, axis=0)

    cols = [
        "id",
        "ref_path", "insp_path",
        "NONE_CONF_DROPOUT",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT",
        "PSEUDO_SOLDER_CONF_DROPOUT",
        "NONE_CONF_DROPOUT_STD",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT_STD",
        "PSEUDO_SOLDER_CONF_DROPOUT_STD",
    ]
    result_df = result_df[cols]

    return result_df, all_logits, y_true


# ==============================
# 评估工具函数（混淆矩阵 & F1）
# ==============================

def confusion_matrix_3cls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_class: int = N_CLASSES,
):
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def calc_precision_recall_f1_binary(
    y_true_bin: np.ndarray,
    scores_pos: np.ndarray,
    thresholds: np.ndarray,
):
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        y_pred_bin = (scores_pos >= thr).astype(np.int64)

        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

        if tp + fp == 0:
            prec = 1.0
        else:
            prec = tp / (tp + fp)

        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)

        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return (
        np.array(precisions, dtype=np.float32),
        np.array(recalls, dtype=np.float32),
        np.array(f1s, dtype=np.float32),
        thresholds,
    )


def metrics_from_cm(cm: np.ndarray):
    """
    从混淆矩阵 cm 计算 per-class precision/recall/F1 +
    weighted F1（按 support 加权）
    """
    C = cm.shape[0]
    per_prec = np.zeros(C, dtype=float)
    per_rec = np.zeros(C, dtype=float)
    per_f1 = np.zeros(C, dtype=float)
    supports = cm.sum(axis=1)
    total_support = supports.sum()

    weighted_f1_num = 0.0

    for c in range(C):
        tp = cm[c, c]
        support = supports[c]
        pred_pos = cm[:, c].sum()

        if support == 0:
            rec = float("nan")
        else:
            rec = tp / support

        if pred_pos == 0:
            prec = float("nan")
        else:
            prec = tp / pred_pos

        if np.isnan(prec) or np.isnan(rec) or prec + rec == 0:
            f1 = float("nan")
        else:
            f1 = 2 * prec * rec / (prec + rec)

        per_prec[c], per_rec[c], per_f1[c] = prec, rec, f1

        if not np.isnan(f1):
            weighted_f1_num += f1 * support

    if total_support > 0:
        weighted_f1 = weighted_f1_num / total_support
    else:
        weighted_f1 = float("nan")

    return per_prec, per_rec, per_f1, supports, weighted_f1


# ==============================
# ECE & 可靠性柱状图（标量事件）
# ==============================

def compute_ece(confidences: np.ndarray,
                correctness: np.ndarray,
                n_bins: int = N_BINS_ECE):
    """
    标准 ECE 计算:
      - confidences: [N], 某个“事件”的预测概率（0~1）
      - correctness: [N], 0/1 该事件是否发生
    返回:
      ece, bin_edges, bin_conf_mean, bin_acc_mean, bin_counts
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)

    conf = np.clip(conf, 0.0, 1.0)
    assert conf.shape == corr.shape

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf = np.zeros(n_bins, dtype=float)
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    bin_ids = np.digitize(conf, bin_edges[1:-1], right=True)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt > 0:
            bin_conf[b] = float(conf[mask].mean())
            bin_acc[b] = float(corr[mask].mean())
        else:
            bin_conf[b] = 0.0
            bin_acc[b] = 0.0

    N = conf.shape[0]
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        weight = bin_counts[b] / N
        ece += weight * abs(bin_acc[b] - bin_conf[b])

    return ece, bin_edges, bin_conf, bin_acc, bin_counts


def plot_reliability_bars(
    bin_edges: np.ndarray,
    bin_acc: np.ndarray,
    title: str,
    out_path: str,
):
    """
    按 bin 画柱状可靠性图：
      - x 轴：bin 区间中心
      - y 轴：该 bin 内的 empirical accuracy
      - 叠加一条 y=x 虚线
    """
    n_bins = len(bin_acc)
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(5, 5))

    plt.bar(
        centers,
        bin_acc,
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label="Empirical accuracy",
    )

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] reliability diagram saved to: {out_path}")


# ==============================
# Multiclass “向量版” ECE 及绘图
# ==============================

def compute_multiclass_vector_ece(
    softmaxes: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS_ECE,
):
    """
    Multiclass 向量版 ECE（按 max 置信度分 bin + 按预测类别细分）:
    ...
    """
    probs = np.asarray(softmaxes, dtype=np.float64)   # [N, C]
    labels = np.asarray(labels, dtype=np.int64)       # [N]
    N, C = probs.shape

    pred = probs.argmax(axis=1)                       # [N]
    max_conf = probs[np.arange(N), pred]             # [N]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]

    acc_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    conf_means = np.full((C, n_bins), np.nan, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=int)
    gap_per_bin = np.zeros(n_bins, dtype=np.float64)

    ece = 0.0
    mce = 0.0

    for b, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (max_conf > lower) & (max_conf <= upper)
        count_b = int(in_bin.sum())
        bin_counts[b] = count_b
        if count_b == 0:
            continue

        g_b = 0.0

        for k in range(C):
            in_bin_k = in_bin & (pred == k)
            count_bk = int(in_bin_k.sum())
            if count_bk == 0:
                continue

            acc_k = (labels[in_bin_k] == k).mean()
            conf_k = probs[in_bin_k, k].mean()

            acc_means[k, b] = acc_k
            conf_means[k, b] = conf_k

            gap = abs(acc_k - conf_k)

            weight_in_bin = count_bk / count_b
            g_b += weight_in_bin * gap

            ece += (count_bk / N) * gap

        gap_per_bin[b] = g_b
        if g_b > mce:
            mce = g_b

    return ece, mce, bin_edges, bin_counts, gap_per_bin, acc_means, conf_means


def plot_multiclass_vector_gap(
    bin_edges: np.ndarray,
    gap_per_bin: np.ndarray,
    title: str,
    out_path: str,
):
    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    plt.figure(figsize=(6, 4))
    plt.bar(
        centers,
        gap_per_bin,
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label=r"$\sum_k |acc_k - conf_k|$",
    )

    plt.xlabel("Max predicted probability bin")
    plt.ylabel("Per-bin vector gap")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] multiclass vector gap diagram saved to: {out_path}")


def plot_multiclass_vector_classwise(
    bin_edges: np.ndarray,
    acc_means: np.ndarray,
    conf_means: np.ndarray,
    class_idx: int,
    class_name: str,
    title_prefix: str,
    out_path: str,
):
    acc = acc_means[class_idx]
    conf = conf_means[class_idx]
    gap = np.abs(acc - conf)

    width = bin_edges[1] - bin_edges[0]
    centers = bin_edges[:-1] + width / 2.0

    mask = ~np.isnan(gap)

    plt.figure(figsize=(5, 5))
    plt.bar(
        centers[mask],
        gap[mask],
        width=width * 0.9,
        align="center",
        alpha=0.7,
        edgecolor="black",
        label=r"|acc - conf| (class=%s)" % class_name,
    )

    plt.xlabel("Max predicted probability bin")
    plt.ylabel(r"Per-bin gap |acc - conf|")
    plt.title(f"{title_prefix} (class={class_name})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(
        f"[INFO] multiclass vector per-class gap "
        f"(class={class_name}) saved to: {out_path}"
    )


# ==============================
# 评估主函数：阈值 + ECE + 可靠性图 + F1 曲线
# ==============================

def eval_singlepad_mcdropout(
    csv_path: str,
    save_dir: str,
    part_name: str = "singlepad",
    temp_scaled_cols: Optional[Dict[str, list]] = None,
):
    """
    修改：支持评估温度缩放后的概率
    temp_scaled_cols: 温度缩放后的列名映射，如 {
        "global": ["NONE_CONF_DROPOUT_T_GLOBAL", ...],
        "per_class": ["NONE_CONF_DROPOUT_T_PER_CLASS", ...]
    }
    """
    df = pd.read_csv(csv_path)

    if "part_name" in df.columns:
        df = df[df["part_name"] == part_name].reset_index(drop=True)

    if len(df) == 0:
        print(f"[WARN] CSV 中没有 part_name={part_name} 的数据，跳过评估")
        return

    gt_cols = ["NONE_CONF", "INSUFFICIENT_SOLDER_CONF", "PSEUDO_SOLDER_CONF"]
    if not all(col in df.columns for col in gt_cols):
        raise ValueError(
            "CSV 缺少 ground truth 列："
            "NONE_CONF / INSUFFICIENT_SOLDER_CONF / PSEUDO_SOLDER_CONF"
        )

    soft_gt = df[gt_cols].to_numpy(dtype=np.float32)
    y_true_3 = soft_gt.argmax(axis=1).astype(np.int64)
    y_true_bin = (y_true_3 != 0).astype(np.int64)

    # 构建评测方法列表（原始 + 温度缩放）
    methods = {
        "MCDropout": {
            "cols": [
                "NONE_CONF_DROPOUT",
                "INSUFFICIENT_SOLDER_CONF_DROPOUT",
                "PSEUDO_SOLDER_CONF_DROPOUT",
            ],
            "color": "C1",
            "marker": "^",
        },
    }

    # 添加温度缩放后的评测方法
    if temp_scaled_cols is not None:
        if "global" in temp_scaled_cols:
            methods["MCDropout_T_Global"] = {
                "cols": temp_scaled_cols["global"],
                "color": "C2",
                "marker": "o",
            }
        if "per_class" in temp_scaled_cols:
            methods["MCDropout_T_PerClass"] = {
                "cols": temp_scaled_cols["per_class"],
                "color": "C3",
                "marker": "s",
            }

    thresholds_curve = np.linspace(0.0, 1.0, N_THRESH)
    f1_curves = {}

    print("\n===== singlepad MC Dropout evaluation (3-class & binary + calibration) =====")

    for method_name, cfg in methods.items():
        prob_cols = cfg["cols"]
        if not all(col in df.columns for col in prob_cols):
            print(f"[WARN] 跳过 {method_name}，缺少列: {prob_cols}")
            continue

        probs = df[prob_cols].to_numpy(dtype=np.float32)

        y_pred_3 = probs.argmax(axis=1)
        acc_mclass = (y_pred_3 == y_true_3).mean()

        cm_multi = confusion_matrix_3cls(y_true_3, y_pred_3, n_class=3)
        (
            per_prec_multi,
            per_rec_multi,
            per_f1_multi,
            support_multi,
            weighted_f1_multi,
        ) = metrics_from_cm(cm_multi)

        p_ok = probs[:, 0]
        p_defect = 1.0 - p_ok
        scores_defect = p_defect

        _, _, f1s_curve, _ = calc_precision_recall_f1_binary(
            y_true_bin, scores_defect, thresholds_curve
        )
        f1_curves[method_name] = {
            "f1": f1s_curve,
            "color": cfg["color"],
            "marker": cfg["marker"],
        }

        for thr in THRESH_LIST:
            print(f"\nThreshold = {thr:.2f}  --  [{method_name}]")
            print(f"---- threshold = {thr:.2f} ----")

            y_pred_bin = (scores_defect >= thr).astype(np.int64)
            acc_bin = (y_pred_bin == y_true_bin).mean()

            cm_bin = confusion_matrix_3cls(y_true_bin, y_pred_bin, n_class=2)
            (
                per_prec_bin,
                per_rec_bin,
                per_f1_bin,
                support_bin,
                weighted_f1_bin,
            ) = metrics_from_cm(cm_bin)

            print(f"binary accuracy = {acc_bin:.5f}")
            print(f"mclass  accuracy = {acc_mclass:.5f}")

            print("== Binary output ==")
            print(f"weighted f1 score = {weighted_f1_bin}")

            bin_names = ["ok", "defect"]
            for c in range(2):
                name = bin_names[c]
                n_c = int(support_bin[c])
                prec_c = per_prec_bin[c]
                rec_c = per_rec_bin[c]
                f1_c = per_f1_bin[c]

                if np.isnan(rec_c):
                    omission = 0.0
                else:
                    omission = 1.0 - rec_c

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

            print("== Multiclass output ==")
            print(f"weighted f1 score = {weighted_f1_multi}")

            for c in range(N_CLASSES):
                name = CLASS_ORDER[c]
                n_c = int(support_multi[c])
                prec_c = per_prec_multi[c]
                rec_c = per_rec_multi[c]
                f1_c = per_f1_multi[c]

                if np.isnan(rec_c):
                    omission = 0.0
                else:
                    omission = 1.0 - rec_c

                print(
                    f"{name} (n={n_c}): "
                    f"precision:{0.0 if np.isnan(prec_c) else prec_c:.3f}, "
                    f"recall:{0.0 if np.isnan(rec_c) else rec_c:.3f}, "
                    f"f1_score:{0.0 if np.isnan(f1_c) else f1_c:.3f}, "
                    f"omission_rate:{omission:.3f}"
                )

        y_pred_bin_ece = (p_defect >= 0.5).astype(np.int64)
        conf_bin_pred = np.where(y_pred_bin_ece == 1, p_defect, p_ok)
        correct_bin = (y_pred_bin_ece == y_true_bin).astype(np.int64)

        ece_bin_overall, bin_edges_b, _, bin_acc_b, _ = compute_ece(
            conf_bin_pred, correct_bin, n_bins=N_BINS_ECE
        )

        bin_overall_path = os.path.join(
            save_dir, f"singlepad_binary_reliability_overall_{method_name}.png"
        )
        plot_reliability_bars(
            bin_edges_b,
            bin_acc_b,
            title=f"Binary reliability overall (top1, {method_name})",
            out_path=bin_overall_path,
        )

        conf_ok = p_ok
        corr_ok = (y_true_bin == 0).astype(np.int64)
        ece_ok, bin_edges_ok, _, bin_acc_ok, _ = compute_ece(
            conf_ok, corr_ok, n_bins=N_BINS_ECE
        )
        path_ok = os.path.join(
            save_dir,
            f"singlepad_binary_reliability_true_ok_{method_name}.png",
        )
        plot_reliability_bars(
            bin_edges_ok,
            bin_acc_ok,
            title=f"Binary reliability (true=ok, {method_name})",
            out_path=path_ok,
        )

        conf_def = p_defect
        corr_def = (y_true_bin == 1).astype(np.int64)
        ece_def, bin_edges_def, _, bin_acc_def, _ = compute_ece(
            conf_def, corr_def, n_bins=N_BINS_ECE
        )
        path_def = os.path.join(
            save_dir,
            f"singlepad_binary_reliability_true_defect_{method_name}.png",
        )
        plot_reliability_bars(
            bin_edges_def,
            bin_acc_def,
            title=f"Binary reliability (true=defect, {method_name})",
            out_path=path_def,
        )

        print(f"\n[{method_name}] Binary calibration")
        print(f"  overall ECE (top1) = {ece_bin_overall:.4f}")
        print(
            "  classwise ECE (prob-wise, by true class): "
            f"ok={ece_ok:.4f}, defect={ece_def:.4f}"
        )

        conf_multi_pred = probs[np.arange(len(probs)), y_pred_3]
        correct_multi = (y_pred_3 == y_true_3).astype(np.int64)

        ece_multi_overall, bin_edges_m, _, bin_acc_m, _ = compute_ece(
            conf_multi_pred, correct_multi, n_bins=N_BINS_ECE
        )

        multi_overall_path = os.path.join(
            save_dir, f"singlepad_multiclass_reliability_overall_{method_name}.png"
        )
        plot_reliability_bars(
            bin_edges_m,
            bin_acc_m,
            title=f"Multiclass reliability overall (top1, {method_name})",
            out_path=multi_overall_path,
        )

        ece_multi_class = {}
        for c, cname in enumerate(CLASS_ORDER):
            conf_c = probs[:, c]
            corr_c = (y_true_3 == c).astype(np.int64)

            ece_c, bin_edges_c, _, bin_acc_c, _ = compute_ece(
                conf_c, corr_c, n_bins=N_BINS_ECE
            )
            ece_multi_class[c] = ece_c

            path_c = os.path.join(
                save_dir,
                f"singlepad_multiclass_reliability_true_{cname}_{method_name}.png",
            )
            plot_reliability_bars(
                bin_edges_c,
                bin_acc_c,
                title=f"Multiclass reliability (true={cname}, {method_name})",
                out_path=path_c,
            )

        print(f"\n[{method_name}] Multiclass calibration (true-class prob-wise)")
        print(f"  overall ECE (top1) = {ece_multi_overall:.4f}")
        print(
            "  classwise ECE (prob-wise, by true class): "
            f"ok={ece_multi_class.get(0, float('nan')):.4f}, "
            f"undersolder={ece_multi_class.get(1, float('nan')):.4f}, "
            f"pseudosolder={ece_multi_class.get(2, float('nan')):.4f}"
        )

        (
            ece_vec,
            mce_vec,
            bin_edges_vec,
            bin_counts_vec,
            gap_per_bin,
            acc_means_vec,
            conf_means_vec,
        ) = compute_multiclass_vector_ece(
            probs, y_true_3, n_bins=N_BINS_ECE
        )

        print(f"\n[{method_name}] Multiclass vector calibration (all-class probs)")
        print(f"  vector-ECE = {ece_vec:.4f}")
        print(f"  vector-MCE = {mce_vec:.4f}")

        gap_path = os.path.join(
            save_dir,
            f"singlepad_multiclass_vector_gap_{method_name}.png",
        )
        plot_multiclass_vector_gap(
            bin_edges_vec,
            gap_per_bin,
            title=f"Multiclass vector gap per bin ({method_name})",
            out_path=gap_path,
        )

        for c, cname in enumerate(CLASS_ORDER):
            out_c = os.path.join(
                save_dir,
                f"singlepad_multiclass_vector_reliability_{cname}_{method_name}.png",
            )
            plot_multiclass_vector_classwise(
                bin_edges_vec,
                acc_means_vec,
                conf_means_vec,
                class_idx=c,
                class_name=cname,
                title_prefix="Multiclass vector reliability",
                out_path=out_c,
            )

    if not f1_curves:
        print("[WARN] 没有方法产生 F1 曲线，跳过画图。")
        return

    plt.figure(figsize=(8, 5))
    for method_name, info in f1_curves.items():
        f1s = info["f1"]
        color = info["color"]

        best_idx = int(f1s.argmax())
        best_thr = thresholds_curve[best_idx]
        best_f1 = f1s[best_idx]

        print(f"\n[{method_name}] F1 curve (binary defect vs ok)")
        print(f"  Max F1 = {best_f1:.4f} at threshold = {best_thr:.3f}")

        plt.plot(
            thresholds_curve,
            f1s,
            label=f"{method_name} (max F1={best_f1:.3f} @ {best_thr:.2f})",
            color=color,
        )

    plt.xlabel("Threshold on P(defect)")
    plt.ylabel("F1 score (defect vs ok)")
    plt.title("singlepad F1 curve vs threshold (MC Dropout + Temperature Scaling)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best")

    out_fig = os.path.join(save_dir, "singlepad_f1_curve_defect_vs_ok_mcdropout.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nF1 曲线图已保存到: {out_fig}")


# ==============================
# main：推理 + 温度缩放 + 写回 CSV + 评估
# ==============================
if __name__ == "__main__":
    # 设置随机种子，保证温度搜索可复现
    np.random.seed(42)
    torch.manual_seed(42)
    
    part_name = "singlepad"
    model = _build_model_for_part(part_name)

    base_df = pd.read_csv(CSV_PATH)
    print(f"origin df length: {len(base_df)}")

    dup_id = base_df["id"].duplicated().sum()
    if dup_id > 0:
        raise ValueError(f"id 列出现重复: {dup_id} 行，请先清洗再推理")

    cols = [
        "NONE_CONF_DROPOUT",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT",
        "PSEUDO_SOLDER_CONF_DROPOUT",
        "NONE_CONF_DROPOUT_STD",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT_STD",
        "PSEUDO_SOLDER_CONF_DROPOUT_STD",
    ]

    # 温度缩放后的列名
    t_global_cols = [
        "NONE_CONF_DROPOUT_T_GLOBAL",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT_T_GLOBAL",
        "PSEUDO_SOLDER_CONF_DROPOUT_T_GLOBAL",
    ]
    t_per_class_cols = [
        "NONE_CONF_DROPOUT_T_PER_CLASS",
        "INSUFFICIENT_SOLDER_CONF_DROPOUT_T_PER_CLASS",
        "PSEUDO_SOLDER_CONF_DROPOUT_T_PER_CLASS",
    ]

    dropout_p_list = [0]

    for p in dropout_p_list:
        print("\n" + "=" * 80)
        print(f"开始 MC Dropout 实验, dropout p = {p}")
        print("=" * 80 + "\n")

        set_dropout_p(model, p)

        # 推理（返回logits和真实标签）
        result_df, all_logits, y_true = infer_singlepad_batch(
            csv_path=CSV_PATH,
            root_path=ROOT_DIR,
            model=model,
            part_name=part_name,
            batch_size=64,
            mc_times=30,
        )
        print(f"[p={p}] result df length: {len(result_df)}")

        df = base_df.copy()

        df = df.set_index("id")
        result_df = result_df.set_index("id")

        # 写入原始MC Dropout结果
        df[cols] = result_df[cols]

        # ==============================
        # 温度缩放优化
        # ==============================
        if y_true is not None and len(all_logits) > 0:
            # 1. 全局温度优化
            best_t_global, best_ece_global = grid_search_best_global_temperature(
                all_logits, y_true, val_ratio=VAL_RATIO
            )
            # 应用全局温度缩放
            probs_global = temperature_scaling(all_logits, best_t_global)
            
            # 2. 按类温度优化
            best_t_per_class, best_ece_per_class = grid_search_best_per_class_temperatures(
                all_logits, y_true, val_ratio=VAL_RATIO
            )
            # 应用按类温度缩放
            probs_per_class = per_class_temperature_scaling(all_logits, best_t_per_class)
            
            # 将温度缩放后的概率写入DataFrame
            # 先对齐索引
            result_df = result_df.reset_index()
            # 确保顺序一致
            assert len(result_df) == len(probs_global), "样本数量不匹配"
            
            # 添加温度缩放结果到result_df
            result_df[t_global_cols[0]] = probs_global[:, 0]
            result_df[t_global_cols[1]] = probs_global[:, 1]
            result_df[t_global_cols[2]] = probs_global[:, 2]
            
            result_df[t_per_class_cols[0]] = probs_per_class[:, 0]
            result_df[t_per_class_cols[1]] = probs_per_class[:, 1]
            result_df[t_per_class_cols[2]] = probs_per_class[:, 2]
            
            # 重新设置索引并写入主df
            result_df = result_df.set_index("id")
            df[t_global_cols] = result_df[t_global_cols]
            df[t_per_class_cols] = result_df[t_per_class_cols]
            
            # 保存温度优化结果
            temp_results = {
                "dropout_p": p,
                "best_global_temperature": best_t_global,
                "best_global_ece": best_ece_global,
                "best_per_class_temperatures": best_t_per_class,
                "best_per_class_ece": best_ece_per_class
            }
            temp_result_path = os.path.join(SAVE_DIR, f"temperature_results_p{p:.2f}.csv")
            pd.DataFrame([temp_results]).to_csv(temp_result_path, index=False, encoding="utf-8-sig")
            print(f"温度优化结果已保存到: {temp_result_path}")

        # 保存包含温度缩放的CSV
        tag = f"p{p:.2f}".replace(".", "_")
        out_csv = os.path.join(
            SAVE_DIR,
            f"checked_samples_mcdropout_{tag}.csv"
        )
        df.reset_index().to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] MC Dropout 结果 (p={p}) 已写到: {out_csv}")

        # 评估（包含温度缩放结果）
        eval_dir = os.path.join(SAVE_DIR, tag)
        os.makedirs(eval_dir, exist_ok=True)

        eval_singlepad_mcdropout(
            csv_path=out_csv,
            save_dir=eval_dir,
            part_name=part_name,
            temp_scaled_cols={
                "global": t_global_cols,
                "per_class": t_per_class_cols
            },
        )

    print("\n==== 所有 dropout 概率实验已完成 ====\n")
    print("每个 p 的结果：")
    print("  CSV 在:", SAVE_DIR)
    print("  评估图在:", os.path.join(SAVE_DIR, "p0_10 / p0_20 / p0_30 / ..."))