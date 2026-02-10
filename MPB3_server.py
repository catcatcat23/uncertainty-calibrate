# cnn_server.py
import os
import glob
import torch
import numpy as np
from scipy.special import softmax
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm

from utils.utilities import TransformImage
from models.MPB3 import MPB3net


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
                f"Key 不匹配: ckpt {key} ({value.shape}) vs model {model_key} ({model_state_dict[model_key].shape})"
            )

        index += 1

    model.load_state_dict(new_state_dict, strict=True)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 这个目录你已经有了
CKPT_DIR = "/home/cat/workspace/vlm/scripts/models/checkpoints"

N_CLASS = 3                     


# 顺序和 defect_code.keys() 一致：
# defect_code = {'ok': 0, 'undersolder': 4, 'pseudosolder': 6}
CLASS_ORDER = ["ok", "undersolder", "pseudosolder"]


def _find_ckpt(pattern: str) -> str:
    """
    在 CKPT_DIR 下用通配符找 ckpt 文件。
    pattern 例如: 'singlepad*rs6464*dual2*.pth.tar'
    """
    full_pattern = os.path.join(CKPT_DIR, pattern)
    files = glob.glob(full_pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配 ckpt: {full_pattern}")
    if len(files) > 1:
        print(f"[WARN] 匹配到多个 ckpt，只使用第一个：{files[0]}")
    return files[0]

PART_CONFIG = {
    "singlepad": {
        # 会匹配 /home/cat/workspace/vlm/scripts/models/checkpoints/singlepadfcdropoutmobilenetv3largers6464...dual2last.pth.tar
        "ckpt_pattern": "singlepadfcdropoutmobilenetv3largers6464*dual2*.pth.tar",
        "backbone": "fcdropoutmobilenetv3large",
        "n_units": [256, 256],   # nb256nm256
        "n_class": N_CLASS,
        "img_h": 64,             # rs6464 -> 64 x 64
        "img_w": 64,
    },
    "singlepinpad": {
        # 会匹配 /home/cat/workspace/vlm/scripts/models/checkpoints/singlepinpadmobilenetv3smallrs12832...dual2top2.pth.tar
        "ckpt_pattern": "singlepinpadmobilenetv3smallrs12832*dual2*.pth.tar",
        "backbone": "mobilenetv3small",
        "n_units": [256, 256],   # 同样是 nb256nm256
        "n_class": N_CLASS,
        # 注意顺序：TransformImage(rs_img_size_h, rs_img_size_w)
        # rs12832 -> W=128, H=32
        "img_h": 32,
        "img_w": 128,
    },
}

# 已加载的模型缓存：避免每次调用都重新 load
MODEL_CACHE: Dict[str, MPB3net] = {}


def _build_model_for_part(part_name: str) -> MPB3net:
    if part_name not in PART_CONFIG:
        raise ValueError(f"未知 part_name: {part_name}，请在 PART_CONFIG 里加配置。")

    cfg = PART_CONFIG[part_name]
    ckpt_path = _find_ckpt(cfg["ckpt_pattern"])

    print(f"=> 为 {part_name} 加载 CNN 模型: {ckpt_path}")

    model = MPB3net(
        backbone=cfg["backbone"],
        pretrained=False,
        n_class=cfg["n_class"],
        n_units=cfg["n_units"],
        output_form="dual2",   # 如果是 CL 模型，这里要改成 dual2CL / dual2CLC 等
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # 如果你另存的是纯 state_dict
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[WARN] strict=True 加载失败，尝试按 value 对齐: {e}")
        load_model_by_values(state_dict, model)

    model.to(DEVICE)
    model.eval()
    print(f"=> {part_name} 模型已加载到 {DEVICE}")
    return model


def _get_model_and_size(part_name: str) -> Tuple[MPB3net, int, int]:
    # 默认当成 singlepad
    if part_name not in PART_CONFIG:
        print(f"[WARN] meta.part_name={part_name} 未在配置中，使用 singlepad 配置。")
        part_name = "singlepad"

    if part_name not in MODEL_CACHE:
        MODEL_CACHE[part_name] = _build_model_for_part(part_name)

    cfg = PART_CONFIG[part_name]
    model = MODEL_CACHE[part_name]
    return model, cfg["img_h"], cfg["img_w"]


def run_cnn_on_pair(
    ref_image_path: str,
    insp_image_path: str,
    meta: Optional[Dict] = None,
) -> Tuple[float, float, float]:
    """
    输入：ref / insp 绝对路径 + meta（至少含 part_name）
    输出：三个置信度:
        NONE_CONF, INSUFFICIENT_SOLDER_CONF, PSEUDO_SOLDER_CONF
    对应：
        ok, undersolder, pseudosolder
    """
    part_name = None
    if meta is not None:
        part_name = meta.get("part_name", None)
    if part_name is None:
        part_name = "singlepad"

    model, img_h, img_w = _get_model_and_size(part_name)

    # 1) 预处理
    ref_img = TransformImage(
        img_path=ref_image_path,
        rs_img_size_h=img_h,
        rs_img_size_w=img_w,
    ).transform()

    insp_img = TransformImage(
        img_path=insp_image_path,
        rs_img_size_h=img_h,
        rs_img_size_w=img_w,
    ).transform()

    ref_tensor = torch.FloatTensor(ref_img).to(DEVICE)
    insp_tensor = torch.FloatTensor(insp_img).to(DEVICE)

    # 2) 前向
    with torch.no_grad():
        output_bos, output_bom = model(ref_tensor, insp_tensor)

    # 3) 对 bom 做 softmax，得到 3 维概率
    bom_np = softmax(output_bom.detach().cpu().numpy(), axis=1)[0]  # shape: [3]

    # index: 0 -> 'ok'; 1 -> 'undersolder'; 2 -> 'pseudosolder'
    p_ok = float(bom_np[0])
    p_under = float(bom_np[1])
    p_pseudo = float(bom_np[2])

    # 映射到前端三个 slider
    NONE_CONF = p_ok
    INSUFFICIENT_SOLDER_CONF = p_under
    PSEUDO_SOLDER_CONF = p_pseudo

    return NONE_CONF, INSUFFICIENT_SOLDER_CONF, PSEUDO_SOLDER_CONF
