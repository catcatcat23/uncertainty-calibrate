# llm_server/run_llm_on_pair.py

from typing import Tuple, Dict, Optional, List
import os
import json
import base64

import chromadb
from chromadb.config import Settings
from openai import OpenAI  # 走 OpenAI 兼容网关（qwen / glm 网关等）时直接用这个

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


# ========= 0. Chroma & Embedding 配置 =========

# 和建库时保持一致的 Chroma 路径 & collection 名
CHROMA_DIR = os.getenv(
    "RAG_CHROMA_DIR",
    "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310",
)
RAG_COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "pairs_singlepad_singlepinpad_251112",
)

_rag_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False),
)
_rag_collection = _rag_client.get_collection(RAG_COLLECTION_NAME)

# 图片根目录（和建库 / Gradio 前端保持一致）
IMG_ROOT = os.getenv(
    "PAIR_IMG_ROOT",
    "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310",
)



# 强制走离线模式，禁止再连 huggingface
os.environ["HF_HUB_OFFLINE"] = "1"

# 本地模型的绝对路径（你的 snapshot 目录）
LOCAL_CLIP_DIR = os.getenv(
    "PAIR_EMBED_MODEL",  # 如果以后想改路径，可以配环境变量覆盖
    "/home/cat/workspace/vlm/models/hf_models/models--sentence-transformers--clip-ViT-B-32/snapshots/327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
)

print(f"[LLM] Loading SentenceTransformer model from local dir: {LOCAL_CLIP_DIR}")
_clip_model = SentenceTransformer(LOCAL_CLIP_DIR)


def _load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _image_to_data_url(path: str) -> str:
    """
    把本地图片转成 data URL：
      data:image/jpeg;base64,xxxxxxxx
    兼容大多数 OpenAI 兼容的多模态接口（包括 glm-4v 系列）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    return f"data:{mime};base64,{b64}"


def _embed_pair(ref_path: str, insp_path: str) -> np.ndarray:
    """
    和建库脚本里的 embed_pair 完全一致：

        feats = model.encode([ref_img, insp_img], normalize_embeddings=True)
        feat0, feat1 = feats[0], feats[1]
        diff = feat1 - feat0
        pair_feat = concat([feat1, feat0, diff])  # (3D,)

    返回值:
        pair_feat: shape = (3D,) 的 numpy 向量
    """
    ref_img = _load_image(ref_path)
    insp_img = _load_image(insp_path)

    feats = _clip_model.encode(
        [ref_img, insp_img],
        convert_to_numpy=True,
        normalize_embeddings=True,  # ⭐ 和建库时保持一致
    )
    feat0, feat1 = feats[0], feats[1]  # (D,)

    diff = feat1 - feat0
    pair_feat = np.concatenate([feat1, feat0, diff], axis=-1)  # (3D,)

    return pair_feat


# ========= 1. RAG: 用 CLIP-ViT embedding 做向量检索 =========

def _fetch_fewshot_examples(
    ref_abs: str,
    insp_abs: str,
    current_meta: Dict,
    top_k: int = 3,
) -> List[Dict]:
    """
    使用当前样本的 CLIP-ViT pair embedding，在 Chroma 里做向量检索，
    取最相似的若干条已标注样本作为 few-shot 示例。

    策略：
      - query_embeddings = embed_pair(ref_abs, insp_abs)
      - where: checked=True，尽量约束在同 part_name 内
    """
    part_name = current_meta.get("part_name", None)

    # 1. 当前样本的 pair embedding（和建库逻辑一模一样）
    query_emb = _embed_pair(ref_abs, insp_abs)  # (3D,)
    query_emb_list = query_emb.astype(float).tolist()

    # 2. 设置 where 过滤条件
    where: Dict = {"checked": True}
    if part_name is not None:
        where = {"$and": [where, {"part_name": part_name}]}

    # 3. 真·向量检索
    res = _rag_collection.query(
        query_embeddings=[query_emb_list],
        where=where,
        n_results=top_k,
        include=["metadatas"],
    )

    metadatas_list = res.get("metadatas", [])
    if not metadatas_list:
        return []

    # 这里只有一个 query，所以取 [0]
    metas = metadatas_list[0] if len(metadatas_list) > 0 else []

    # Chroma 已经按相似度排序过了，直接返回前 top_k 即可
    return metas[:top_k]


def _format_example(meta: Dict) -> str:
    """
    把一条历史样本的 metadata 格式化成自然语言 few-shot 说明（文字 + JSON）。

    要点：
      - 明确这是“已人工审核”的黄金标注；
      - 显示 insp_defect_label；
      - 显示三类置信度的 JSON，方便模型模仿分布形状。
    """
    part_name = meta.get("part_name", "")
    label = meta.get("insp_defect_label", "")

    none_conf = meta.get("NONE_CONF", None)
    ins_conf = meta.get("INSUFFICIENT_SOLDER_CONF", None)
    pse_conf = meta.get("PSEUDO_SOLDER_CONF", None)

    def _safe_float(x):
        if x is None:
            return "null"
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "null"

    return (
        f"- part_name: {part_name}\n"
        f"- insp_defect_label: {label}\n"
        "  对该样本的【已确认置信度标注】为：\n"
        "{\n"
        f'  "NONE_CONF": {_safe_float(none_conf)},\n'
        f'  "INSUFFICIENT_SOLDER_CONF": {_safe_float(ins_conf)},\n'
        f'  "PSEUDO_SOLDER_CONF": {_safe_float(pse_conf)}\n'
        "}\n"
    )


def _build_global_header(has_fewshots: bool) -> str:
    """
    不依赖具体样本，只负责说明角色、任务和 few-shot 使用方式。
    """
    header = (
        "你是一个专门给 PCB 焊点缺陷打【类别置信度】的助手。\n"
        "系统已经根据检测/分类模型或人工审核，给出了当前样本的缺陷类别，"
        "你**不能修改类别，只负责输出三类的置信度**。\n\n"
        "三类定义如下（类别名称固定，不能更改）：\n"
        "- NONE               ：无缺陷\n"
        "- INSUFFICIENT_SOLDER：焊锡不足\n"
        "- PSEUDO_SOLDER      ：假焊/虚焊\n\n"
        "你的输出是这三类的置信度（0~1 之间的实数），并且会在后处理阶段进行归一化，"
        "通常会期望三者的和接近 1。\n"
        "请保证当前已给定类别对应的置信度通常是三者中最高，"
        "另外两类的置信度用于表达你对“是否存在其他可能”的不确定性。\n\n"
    )

    if has_fewshots:
        header += (
            "接下来我会依次给出若干条【已人工审核通过的历史样本】作为示例。\n"
            "每个示例会按照下面的顺序出现：\n"
            "  1) 一段文字说明（包含 part_name、insp_defect_label 和已确认的置信度 JSON）；\n"
            "  2) 一张 ref 图像；\n"
            "  3) 一张 insp 图像。\n\n"
            "请你学习、模仿这些示例中置信度分布的风格，尤其是：\n"
            "- 已知类别通常具有最高置信度；\n"
            "- 当你认为其他类别有一定可能时，可以给出 0.1~0.3 左右的非零置信度；\n"
            "- 当你几乎排除某个类别时，可以给 0 或非常接近 0 的置信度。\n\n"
        )
    else:
        header += (
            "这次没有历史样本示例，请你只根据给定的图像和缺陷类别信息来打分。\n\n"
        )

    return header


def _build_current_sample_block(current_meta: Dict) -> str:
    """
    描述当前样本的结构化信息 + 最终输出要求。
    这段文字会紧挨着当前样本的 ref/insp 图像出现。
    """
    part_name = current_meta.get("part_name", "")
    raw_label = str(current_meta.get("insp_defect_label", "NONE"))
    csv_name = current_meta.get("csv_name", "")
    csv_index = current_meta.get("csv_index", "")

    text = (
        "【当前样本的信息】\n"
        f"- part_name: {part_name}\n"
        f"- insp_defect_label: {raw_label}   ← 这是已经确定的类别，你不要改动\n"
        f"- csv_name: {csv_name}\n"
        f"- csv_index: {csv_index}\n\n"
        "【最终输出要求】\n"
        "1. 你只需要对“当前样本”输出一次结果。\n"
        "2. 只输出一个 JSON 对象，不要任何解释文字，不要代码块标记（例如 ```json 之类）。\n"
        "3. JSON 必须包含且仅包含以下三个字段，字段名完全一致：\n"
        '   - \"NONE_CONF\"\n'
        '   - \"INSUFFICIENT_SOLDER_CONF\"\n'
        '   - \"PSEUDO_SOLDER_CONF\"\n'
        "4. 每个字段的取值是 0~1 之间的实数（可以是小数），建议使三者之和接近 1。\n\n"
        "输出示例（仅示意，不代表本题答案）：\n"
        "{\n"
        "  \"NONE_CONF\": 0.12,\n"
        "  \"INSUFFICIENT_SOLDER_CONF\": 0.80,\n"
        "  \"PSEUDO_SOLDER_CONF\": 0.08\n"
        "}\n"
        "请按照上述格式，对【当前样本】输出最终的 JSON。\n"
    )

    return text


# ========= 2. 主函数：调用大模型，返回三类置信度 =========

def run_llm_on_pair(
    ref_abs: str,
    insp_abs: str,
    meta: Dict,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    使用大模型对 (ref_abs, insp_abs) 这对图片做推理。

    流程：
      1) 用 CLIP-ViT-B-32 + [insp, ref, insp-ref] 计算当前 pair 的 embedding；
      2) 在同一个 Chroma collection 中做向量检索（where: checked=True, 同 part_name 优先）；
      3) few-shot：把检索到的历史样本“文字 summary + ref/insp 图片”也喂给模型；
      4) 再给出“当前样本”的 ref/insp 图片，只让模型为当前样本打三类置信度；
      5) 解析 JSON，返回 (NONE_CONF, INSUFFICIENT_SOLDER_CONF, PSEUDO_SOLDER_CONF)。
    """
    # 1) API key / 模型名
    api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
    model_name = model_name or os.getenv("LLM_MODEL_NAME", "glm-4v-flash")
        # 3.5 初始化 OpenAI 兼容客户端
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
    )


    if not api_key:
        raise RuntimeError("LLM_API_KEY 未配置，既没有前端传入，也没有环境变量。")

    # 2) 通过向量检索拿 few-shot 示例
    fewshots = _fetch_fewshot_examples(ref_abs, insp_abs, meta, top_k=3)

    # 3) 准备全局头 + 当前样本说明
    header_text = _build_global_header(has_fewshots=bool(fewshots))
    current_block_text = _build_current_sample_block(meta)

    # 4) 构造多模态 messages：先全局文字，再“示例文字+图”，最后“当前样本文字+图”
    content: List[Dict] = []

    # 4.1 全局规则说明（只文字）
    content.append({"type": "text", "text": header_text})

    # 4.2 few-shot 示例：一段文字 + ref 图 + insp 图
    for i, m in enumerate(fewshots, 1):
        ref_rel_fs = m.get("ref_image", "")
        insp_rel_fs = m.get("insp_image", "")
        if not ref_rel_fs or not insp_rel_fs:
            continue

        ref_abs_fs = os.path.join(IMG_ROOT, ref_rel_fs)
        insp_abs_fs = os.path.join(IMG_ROOT, insp_rel_fs)

        # 如果图片路径有问题，跳过该示例
        if (not os.path.exists(ref_abs_fs)) or (not os.path.exists(insp_abs_fs)):
            continue

        # 文本：该示例的说明 + 已确认置信度 JSON
        example_text = f"示例 {i}（已确认标注）：\n" + _format_example(m)
        content.append({"type": "text", "text": example_text})

        # 图片：ref / insp
        ref_data_url_fs = _image_to_data_url(ref_abs_fs)
        insp_data_url_fs = _image_to_data_url(insp_abs_fs)

        content.append(
            {
                "type": "image_url",
                "image_url": {"url": ref_data_url_fs},
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": insp_data_url_fs},
            }
        )

    # 4.3 当前样本：先说明 + 输出要求，再贴 ref / insp
    ref_data_url = _image_to_data_url(ref_abs)
    insp_data_url = _image_to_data_url(insp_abs)

    content.append(
        {
            "type": "text",
            "text": current_block_text,
        }
    )
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": ref_data_url},
        }
    )
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": insp_data_url},
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=256,
        temperature=0.0,
    )

    # 5) 解析 JSON
    reply_content = resp.choices[0].message.content

    if isinstance(reply_content, list):
        # 兼容 [{"type":"text","text":"..."}] 这种结构
        text_parts = []
        for c in reply_content:
            if isinstance(c, dict) and c.get("type") == "text":
                text_parts.append(c.get("text", ""))
            elif hasattr(c, "text"):
                text_parts.append(c.text)
        raw_text = "\n".join(text_parts)
    else:
        raw_text = str(reply_content)

    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        json_str = raw_text[start:end]
        data = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"LLM 返回的内容无法解析为 JSON：{raw_text}") from e

    none_conf = float(data.get("NONE_CONF", 0.0))
    ins_conf = float(data.get("INSUFFICIENT_SOLDER_CONF", 0.0))
    pse_conf = float(data.get("PSEUDO_SOLDER_CONF", 0.0))

    # 6) 简单归一化，防止越界或全 0
    vals = [none_conf, ins_conf, pse_conf]
    vals = [max(0.0, min(1.0, float(v))) for v in vals]
    s = sum(vals)

    if s <= 0:
        # 极端情况：模型输出全 0 或异常，给一个接近均匀的分布
        vals = [0.34, 0.33, 0.33]
    else:
        vals = [v / s for v in vals]

    none_conf, ins_conf, pse_conf = vals

    return none_conf, ins_conf, pse_conf
