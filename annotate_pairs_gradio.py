import os
from typing import Optional, List, Dict

import gradio as gr
from PIL import Image
import chromadb
from chromadb.config import Settings

from llm_server import run_llm_on_pair  # 要返回 3 个 float
from MPB3_server import run_cnn_on_pair  # 要返回 3 个 float

# ========= 1. 路径配置 =========

# Chroma 向量库根目录
CHROMA_DIR = "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310"

# collection 名
COLLECTION_NAME = "pairs_singlepad_singlepinpad_251112"  # 如果不一样，改这里

# 图像根目录：ref_image / insp_image 的相对路径会拼在这里后面
IMG_ROOT = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"

# ======================================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False),
)
collection = client.get_collection(COLLECTION_NAME)

# 当前样本
current_id: Optional[str] = None
current_meta: Optional[dict] = None

# 当前筛选结果（按 csv_name, csv_index 排序）
filtered_items: List[Dict] = []
current_idx: int = -1

# 筛选下拉选项
FILTER_CSV_OPTIONS: List[str] = ["全部"]
FILTER_LABEL_OPTIONS: List[str] = ["全部"]
FILTER_CHECKED_OPTIONS: List[str] = ["全部", "只看未标", "只看已标"]

# 全库统计
GLOBAL_TOTAL: int = 0          # 全部样本数
GLOBAL_CHECKED: int = 0        # checked == True 的数量
GLOBAL_UNCHECKED: int = 0      # checked == False 的数量

# 记录当前使用的筛选条件（save_and_next 里兜底用）
current_filter_csv: str = "全部"
current_filter_label: str = "全部"
current_filter_checked: str = "只看未标"

# ========= 2. 工具函数 =========

def _load_image_safe(path: str, min_height: int = 256) -> Optional[Image.Image]:
    if not os.path.exists(path):
        print(f"[WARN] image not found: {path}")
        return None

    img = Image.open(path).convert("RGB")
    w, h = img.size

    # 如果图太矮，就按比例放大到 min_height
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.NEAREST)

    return img


def _norm_conf(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        v = default
    if v < 0:
        v = default
    return max(0.0, min(1.0, v))


def recompute_pseudo(none_conf, ins_conf):
    """
    根据前两个置信度自动计算第三个：
      PSEUDO_CONF = 1 - NONE_CONF - INSUFFICIENT_CONF
    然后四舍五入到 2 位小数。
    """
    n = _norm_conf(none_conf)
    i = _norm_conf(ins_conf)

    n = max(0.0, min(1.0, n))
    i = max(0.0, min(1.0, i))

    pseudo = 1.0 - n - i
    pseudo = max(0.0, min(1.0, pseudo))
    pseudo = round(pseudo, 2)

    return pseudo


def normalize_and_round3(none_p, ins_p, pse_p, ndigits: int = 2):
    """
    统一处理 CNN 输出的三个概率：
      1. clamp 到 [0,1]
      2. 归一化，总和 = 1
      3. 每个值 round 到 ndigits 位小数
      4. 补偿 round 误差到最大的一类上，保证和仍然是 1
    返回 (none_p, ins_p, pse_p)
    """
    # 1) 先转 float 并 clamp
    probs = [max(0.0, min(1.0, float(x))) for x in (none_p, ins_p, pse_p)]

    s = sum(probs)
    if s <= 0:
        # 极端情况，全是 0，默认给 [1,0,0]
        probs = [1.0, 0.0, 0.0]
        s = 1.0

    # 2) 归一化
    probs = [p / s for p in probs]

    # 3) 分别 round
    rounded = [round(p, ndigits) for p in probs]

    # 4) 修正因 round 引入的误差
    total_rounded = round(sum(rounded), ndigits)
    diff = round(1.0 - total_rounded, ndigits)

    if abs(diff) > 0:
        # 把误差补到当前最大的一类上
        k = max(range(3), key=lambda i: rounded[i])
        rounded[k] = round(rounded[k] + diff, ndigits)

    # 再 clamp 一次
    rounded = [max(0.0, min(1.0, r)) for r in rounded]

    return tuple(rounded)


def build_where_filter(csv_name: str, checked_filter: str, insp_label: str) -> dict:
    """
    构造 chroma 所需的 where：
      - 支持按 csv_name、insp_defect_label、checked 状态筛选
      - 多个条件时：{"$and": [ {...}, {...}, ... ]}
      - 只有一个条件时：直接返回那个 dict
    """
    clauses: List[dict] = []

    # checked 状态筛选
    if checked_filter == "只看未标":
        clauses.append({"checked": False})
    elif checked_filter == "只看已标":
        clauses.append({"checked": True})
    else:
        # "全部"：不加 checked 条件
        pass

    if csv_name and csv_name != "全部":
        clauses.append({"csv_name": csv_name})
    if insp_label and insp_label != "全部":
        clauses.append({"insp_defect_label": insp_label})

    if not clauses:
        # 没有任何条件，就返回空 dict，表示全库
        return {}

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def init_filter_options_and_stats():
    """
    一次性扫全库：
      - 收集 csv_name / insp_defect_label 的所有取值，用于下拉框
      - 统计全库样本数 + checked / unchecked 数量
    """
    global FILTER_CSV_OPTIONS, FILTER_LABEL_OPTIONS
    global GLOBAL_TOTAL, GLOBAL_CHECKED, GLOBAL_UNCHECKED

    csv_set, label_set = set(), set()
    total = 0
    checked_cnt = 0

    offset = 0
    batch_size = 1000

    while True:
        res = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
        ids = res["ids"]
        if not ids:
            break
        metas = res["metadatas"]
        for m in metas:
            total += 1
            if bool(m.get("checked", False)):
                checked_cnt += 1

            if m.get("csv_name") is not None:
                csv_set.add(m["csv_name"])
            if m.get("insp_defect_label") is not None:
                label_set.add(str(m["insp_defect_label"]))
        offset += len(ids)

    GLOBAL_TOTAL = total
    GLOBAL_CHECKED = checked_cnt
    GLOBAL_UNCHECKED = max(0, total - checked_cnt)

    FILTER_CSV_OPTIONS = ["全部"] + sorted(csv_set)
    FILTER_LABEL_OPTIONS = ["全部"] + sorted(label_set)


def refresh_filtered_items(csv_name: str, checked_filter: str, insp_label: str):
    """
    根据筛选条件生成列表：
      - 支持 csv_name / insp_defect_label / checked 状态
      - 结果按 (csv_name, csv_index) 排序
    """
    global filtered_items, current_idx

    where = build_where_filter(csv_name, checked_filter, insp_label)

    items: List[Dict] = []
    offset = 0
    batch_size = 1000

    while True:
        res = collection.get(
            where=where,
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )
        ids = res["ids"]
        if not ids:
            break
        metas = res["metadatas"]
        for i in range(len(ids)):
            items.append({"id": ids[i], "meta": metas[i]})
        offset += len(ids)

    def sort_key(item):
        m = item["meta"]
        csv_n = m.get("csv_name") or ""
        try:
            idx = int(m.get("csv_index", 1e9))
        except Exception:
            idx = int(1e9)
        return (csv_n, idx)

    items.sort(key=sort_key)

    filtered_items = items
    current_idx = 0 if items else -1


def make_outputs(pair_id: Optional[str], meta: Optional[dict], status_prefix: str = ""):
    """
    根据当前样本构造 Gradio 输出：
      - info_html（表格）
      - 两张图
      - 三个 conf
      - checked（UI 默认 True）
      - 状态字符串
      - id 下拉框的 choices + 当前选中值
    """
    global GLOBAL_TOTAL, GLOBAL_CHECKED, GLOBAL_UNCHECKED, filtered_items, current_idx

    # 全库统计字符串
    if GLOBAL_TOTAL > 0:
        ratio = GLOBAL_CHECKED / GLOBAL_TOTAL
        global_part = f"全库进度：已标 {GLOBAL_CHECKED}/{GLOBAL_TOTAL} ({ratio:.1%})，未标 {GLOBAL_UNCHECKED}"
    else:
        global_part = "全库进度：暂无样本"

    # 当前筛选进度
    if filtered_items and current_idx >= 0:
        filter_part = f"当前筛选进度：{current_idx + 1}/{len(filtered_items)}"
    else:
        filter_part = "当前筛选进度：0/0（无匹配样本）"

    pieces = []
    if status_prefix.strip():
        pieces.append(status_prefix.strip())
    pieces.append(filter_part)
    pieces.append(global_part)
    status = "。 ".join(pieces)

    # id 列表
    id_choices = [item["id"] for item in filtered_items] if filtered_items else []

    # 没有样本的情况
    if pair_id is None or meta is None:
        info_html = "<b>当前筛选条件下没有可展示的样本 🎉</b>"

        id_update = gr.update(
            choices=id_choices,
            value=None,
        )

        return (
            info_html,
            None,
            None,
            0.0,
            0.0,
            0.0,
            True,   # 无样本时 UI 也默认勾选
            status,
            id_update,
        )

    # 有样本：正常展示
    ref_rel = meta.get("ref_image", "")
    insp_rel = meta.get("insp_image", "")
    ref_abs = os.path.join(IMG_ROOT, ref_rel)
    insp_abs = os.path.join(IMG_ROOT, insp_rel)

    none_conf = _norm_conf(meta.get("NONE_CONF", -1.0))
    ins_conf = _norm_conf(meta.get("INSUFFICIENT_SOLDER_CONF", -1.0))

    raw_pse = meta.get("PSEUDO_SOLDER_CONF", None)
    if raw_pse is None:
        pse_conf = recompute_pseudo(none_conf, ins_conf)
    else:
        try:
            pse_conf = float(raw_pse)
        except Exception:
            pse_conf = recompute_pseudo(none_conf, ins_conf)
        pse_conf = max(0.0, min(1.0, pse_conf))

    # DB 里的值：仅用于统计/筛选，不直接驱动 UI 选中状态
    checked_in_db = bool(meta.get("checked", False))
    # 前端 UI 默认勾选，避免你每次都点
    checked_for_ui = True

    info_html = f"""
    <table style="border-collapse: collapse; width: 100%; font-size: 14px;">
      <tr>
        <th style="text-align:left; padding:4px; border-bottom:1px solid #ddd;">字段</th>
        <th style="text-align:left; padding:4px; border-bottom:1px solid #ddd;">值</th>
      </tr>
      <tr><td style="padding:4px;">part_name</td><td style="padding:4px;">{meta.get('part_name')}</td></tr>
      <tr><td style="padding:4px;">split</td><td style="padding:4px;">{meta.get('split')}</td></tr>
      <tr><td style="padding:4px;">csv_name</td><td style="padding:4px;">{meta.get('csv_name')}</td></tr>
      <tr><td style="padding:4px;">csv_index</td><td style="padding:4px;">{meta.get('csv_index')}</td></tr>
      <tr><td style="padding:4px;">insp_defect_label</td><td style="padding:4px;">{meta.get('insp_defect_label')}</td></tr>
      <tr><td style="padding:4px;">ref_image</td><td style="padding:4px;">{ref_rel}</td></tr>
      <tr><td style="padding:4px;">insp_image</td><td style="padding:4px;">{insp_rel}</td></tr>
      <tr><td style="padding:4px;">id</td><td style="padding:4px; word-break:break-all;">{pair_id}</td></tr>
      <tr><td style="padding:4px;">checked_in_db</td><td style="padding:4px;">{checked_in_db}</td></tr>
    </table>
    """

    # 当前 id 下拉框选中的 value
    selected_id = pair_id if pair_id in id_choices else (
        id_choices[current_idx] if 0 <= current_idx < len(id_choices) else None
    )

    id_update = gr.update(
        choices=id_choices,
        value=selected_id,
    )

    return (
        info_html,
        _load_image_safe(ref_abs, min_height=144),
        _load_image_safe(insp_abs, min_height=144),
        none_conf,
        ins_conf,
        pse_conf,
        checked_for_ui,
        status,
        id_update,
    )


# ========= 3. Gradio 回调：应用筛选 =========

def apply_filter(csv_name, insp_label, checked_filter):
    """点击“应用筛选”或页面加载时调用：重建 filtered_items，并从第一条开始标注。"""
    global current_id, current_meta, current_filter_csv, current_filter_label, current_filter_checked

    current_filter_csv = csv_name
    current_filter_label = insp_label
    current_filter_checked = checked_filter

    refresh_filtered_items(csv_name, checked_filter, insp_label)

    if current_idx == -1:
        current_id, current_meta = None, None
        prefix = f"已应用筛选（csv={csv_name}, label={insp_label}, checked={checked_filter}）"
        return make_outputs(None, None, status_prefix=prefix)

    item = filtered_items[current_idx]
    current_id, current_meta = item["id"], item["meta"]
    prefix = f"已应用筛选（csv={csv_name}, label={insp_label}, checked={checked_filter}）"
    return make_outputs(current_id, current_meta, status_prefix=prefix)


# ========= 4. 保存并下一条 =========

def save_and_next(none_conf, ins_conf, pse_conf, checked):
    """
    保存前先检查：
      - 三个置信度各自 round 到 2 位小数
      - 若总和 != 1.00，则报错，不写 DB，不跳下一条
    """
    global current_id, current_meta, current_idx
    global GLOBAL_CHECKED, GLOBAL_UNCHECKED
    global current_filter_csv, current_filter_label, current_filter_checked

    # 如果当前没有样本或还没筛选，默认用当前筛选条件（初始为 全部 + 只看未标）
    if current_id is None or current_meta is None or not filtered_items:
        refresh_filtered_items(current_filter_csv, current_filter_checked, current_filter_label)
        if current_idx == -1:
            return make_outputs(None, None, status_prefix="没有样本。")
        item = filtered_items[current_idx]
        current_id, current_meta = item["id"], item["meta"]

    # 1) 先把三个值统一成两位小数（用于检查）
    n2 = round(_norm_conf(none_conf), 2)
    i2 = round(_norm_conf(ins_conf), 2)
    p2 = round(_norm_conf(pse_conf), 2)
    total = round(n2 + i2 + p2, 2)

    if total != 1.0:
        # 不写 DB、不动 current_idx，只在前端提示错误
        tmp_meta = dict(current_meta)
        tmp_meta.update(
            {
                "NONE_CONF": n2,
                "INSUFFICIENT_SOLDER_CONF": i2,
                "PSEUDO_SOLDER_CONF": p2,
                # checked 这里仍保持 DB 原值
            }
        )
        msg = f"保存失败：三类置信度之和为 {total:.2f}，需要等于 1.00，请调整后再保存。"
        return make_outputs(current_id, tmp_meta, status_prefix=msg)

    # 通过检查，再保留四位小数写 DB（值本身只有两位有效）
    n4 = round(n2, 4)
    i4 = round(i2, 4)
    p4 = round(p2, 4)

    # 更新 metadata（覆盖原有置信度）
    old_checked = bool(current_meta.get("checked", False))
    new_checked = bool(checked)

    new_meta = dict(current_meta)
    new_meta.update(
        {
            "NONE_CONF": n4,
            "INSUFFICIENT_SOLDER_CONF": i4,
            "PSEUDO_SOLDER_CONF": p4,
            "checked": new_checked,
        }
    )

    # 写回 chroma
    collection.update(
        ids=[current_id],
        metadatas=[new_meta],
    )

    # 更新全库统计
    if old_checked != new_checked:
        if new_checked:
            GLOBAL_CHECKED += 1
            GLOBAL_UNCHECKED = max(0, GLOBAL_UNCHECKED - 1)
        else:
            GLOBAL_CHECKED = max(0, GLOBAL_CHECKED - 1)
            GLOBAL_UNCHECKED += 1

    # 同步到本地缓存
    if 0 <= current_idx < len(filtered_items):
        filtered_items[current_idx]["meta"] = new_meta
    current_meta = new_meta

    # 下一个样本
    current_idx += 1
    if current_idx >= len(filtered_items):
        current_id, current_meta = None, None
        return make_outputs(None, None, status_prefix="当前筛选条件下已经全部标完。")

    item = filtered_items[current_idx]
    current_id, current_meta = item["id"], item["meta"]
    return make_outputs(current_id, current_meta, status_prefix="已保存当前样本。")


# ========= 4.4 用 CNN 模型自动打分当前样本（只改前端，不写回 DB） =========
def cnn_annotate_current():
    """
    用 CNN 模型自动打分当前样本：
      - 调 run_cnn_on_pair(...)，返回 3 个置信度
      - 用 normalize_and_round3 归一化 + 保留两位小数，且三者之和 = 1
      - 只更新前端 slider，不写回 Chroma，不改变 checked
    """
    global current_id, current_meta

    if current_id is None or current_meta is None:
        return make_outputs(None, None, status_prefix="当前没有样本可供推理，请先应用筛选。")

    # 构造绝对路径
    ref_rel = current_meta.get("ref_image", "")
    insp_rel = current_meta.get("insp_image", "")
    ref_abs = os.path.join(IMG_ROOT, ref_rel)
    insp_abs = os.path.join(IMG_ROOT, insp_rel)

    # 调 CNN 模型 —— 要求返回 (none_conf, ins_conf, pseudo_conf)
    cnn_none, cnn_ins, cnn_pse = run_cnn_on_pair(
        ref_abs,
        insp_abs,
        current_meta,   # 里面的 part_name 用来区分 singlepad / singlepinpad
    )

    # 统一处理：归一化 + 两位小数 + 和为 1
    cnn_none, cnn_ins, cnn_pse = normalize_and_round3(cnn_none, cnn_ins, cnn_pse, ndigits=2)

    # 构造一个“只更新 conf、不改 checked”的临时 meta
    tmp_meta = dict(current_meta)
    tmp_meta.update(
        {
            "NONE_CONF": cnn_none,
            "INSUFFICIENT_SOLDER_CONF": cnn_ins,
            "PSEUDO_SOLDER_CONF": cnn_pse,
            "checked": bool(current_meta.get("checked", False)),
        }
    )

    prefix = f"已用 CNN 模型自动打分当前样本（id={current_id[:8]}...），请人工检查后再点击“保存并看下一条”。"

    return make_outputs(current_id, tmp_meta, status_prefix=prefix)


# ========= 4.5 用大模型自动打分当前样本（只改前端，不写回 DB） =========
def llm_annotate_current(api_key, model_name):
    """
    用大模型自动打分当前样本：
      - 调 run_llm_on_pair(...)，要求返回 3 个置信度
      - 这里仍然用 _norm_conf + 重算 pseudo（你如果想也可以换 normalize_and_round3）
    """
    global current_id, current_meta

    if current_id is None or current_meta is None:
        return make_outputs(None, None, status_prefix="当前没有样本可供推理，请先应用筛选。")

    ref_rel = current_meta.get("ref_image", "")
    insp_rel = current_meta.get("insp_image", "")
    ref_abs = os.path.join(IMG_ROOT, ref_rel)
    insp_abs = os.path.join(IMG_ROOT, insp_rel)

    llm_none, llm_ins, llm_pse = run_llm_on_pair(
        ref_abs,
        insp_abs,
        current_meta,
        api_key=api_key,
        model_name=model_name,
    )

    llm_none = _norm_conf(llm_none)
    llm_ins = _norm_conf(llm_ins)
    # 第三类按“1 - 前两类”重算，保证三者和约等于 1
    llm_pse = recompute_pseudo(llm_none, llm_ins)

    tmp_meta = dict(current_meta)
    tmp_meta.update(
        {
            "NONE_CONF": round(llm_none, 2),
            "INSUFFICIENT_SOLDER_CONF": round(llm_ins, 2),
            "PSEUDO_SOLDER_CONF": round(llm_pse, 2),
            "checked": bool(current_meta.get("checked", False)),
        }
    )

    prefix = f"已调用大模型自动打分当前样本（id={current_id[:8]}...），请人工检查后再点击“保存并看下一条”。"

    return make_outputs(current_id, tmp_meta, status_prefix=prefix)


# ========= 4.6 按 id 跳转到指定样本 =========
def jump_to_id(selected_id: Optional[str]):
    global current_id, current_meta, current_idx

    if not selected_id:
        return make_outputs(current_id, current_meta, status_prefix="未选择 id。")

    if not filtered_items:
        return make_outputs(current_id, current_meta, status_prefix="当前没有筛选结果，无法跳转。")

    for idx, item in enumerate(filtered_items):
        if item["id"] == selected_id:
            current_idx = idx
            current_id = item["id"]
            current_meta = item["meta"]
            return make_outputs(current_id, current_meta, status_prefix=f"已跳转到选中的 id（第 {idx + 1} 条）。")

    return make_outputs(current_id, current_meta, status_prefix="未在当前筛选结果中找到该 id。")


# ========= 5. 初始化筛选选项 & 全库统计 =========

init_filter_options_and_stats()

# ========= 6. Gradio 界面 =========

with gr.Blocks() as demo:
    gr.Markdown("## PCB 缺陷 pair 标注")

    # —— 大模型相关配置（可选） ——
    with gr.Accordion("大模型设置（可选）", open=False):
        api_key_comp = gr.Textbox(
            label="API Key",
            type="password",
            placeholder="如果留空，则使用后台默认的环境变量 LLM_API_KEY",
        )
        model_name_comp = gr.Textbox(
            label="模型名称",
            value="glm-4v-flash",
            placeholder="例如：qwen2-vl-72b、glm-4v-9b 等",
        )

    # —— 筛选区 ——
    with gr.Row():
        csv_filter_comp = gr.Dropdown(
            label="筛选 csv_name",
            choices=FILTER_CSV_OPTIONS,
            value="全部",
        )
        label_filter_comp = gr.Dropdown(
            label="筛选 insp_defect_label",
            choices=FILTER_LABEL_OPTIONS,
            value="全部",
        )

    checked_filter_comp = gr.Dropdown(
        label="checked 状态",
        choices=FILTER_CHECKED_OPTIONS,
        value="只看未标",
    )

    apply_btn = gr.Button("应用筛选")

    # —— 样本信息 + 图片 ——
    info_box = gr.HTML(label="样本信息")

    with gr.Row():
        ref_img_comp = gr.Image(
            label="ref_image",
            type="pil",
            interactive=False,
            height=256,
        )
        insp_img_comp = gr.Image(
            label="insp_image",
            type="pil",
            interactive=False,
            height=256,
        )

    # —— id 选择器，用于跳转 ——
    id_selector_comp = gr.Dropdown(
        label="当前筛选下的样本 id（选择即可跳转）",
        choices=[],
        value=None,
        interactive=True,
    )

    # —— 三个置信度滑块（第三个自动计算） ——
    with gr.Row():
        none_conf_comp = gr.Slider(
            label="NONE_CONF",
            minimum=0.0,
            maximum=1.0,
            step=0.01,   # 步长 0.01
            value=0.0,
        )
        ins_conf_comp = gr.Slider(
            label="INSUFFICIENT_SOLDER_CONF",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
        )
        pse_conf_comp = gr.Slider(
            label="PSEUDO_SOLDER_CONF（自动 = 1 - 前两类）",
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.0,
            interactive=False,  # 只读，由前两类自动算，也会被模型填充
        )

    checked_comp = gr.Checkbox(
        label="checked (这条已完成标注)",
        value=True,  # 初始也设为 True
    )

    status_comp = gr.Markdown("状态：尚未加载")

    # 按钮区
    with gr.Row():
        cnn_btn = gr.Button("用本地模型自动打分当前样本")
        llm_btn = gr.Button("用大模型自动打分当前样本")
        next_btn = gr.Button("保存并看下一条")

    # 联动：前两个变化时自动算第三个
    for comp in [none_conf_comp, ins_conf_comp]:
        comp.change(
            fn=recompute_pseudo,
            inputs=[none_conf_comp, ins_conf_comp],
            outputs=[pse_conf_comp],
        )

    # 页面初次加载：用当前下拉框默认值作为筛选条件
    demo.load(
        fn=apply_filter,
        inputs=[csv_filter_comp, label_filter_comp, checked_filter_comp],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

    # 点击“应用筛选”
    apply_btn.click(
        fn=apply_filter,
        inputs=[csv_filter_comp, label_filter_comp, checked_filter_comp],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

    # id 下拉框改变 → 跳转到对应样本
    id_selector_comp.change(
        fn=jump_to_id,
        inputs=[id_selector_comp],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

    # 点击“用大模型自动打分当前样本”
    llm_btn.click(
        fn=llm_annotate_current,
        inputs=[api_key_comp, model_name_comp],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

    # 点击“用 CNN 模型自动打分当前样本”
    cnn_btn.click(
        fn=cnn_annotate_current,
        inputs=[],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

    # 点击“保存并看下一条”
    next_btn.click(
        fn=save_and_next,
        inputs=[
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
        ],
        outputs=[
            info_box,
            ref_img_comp,
            insp_img_comp,
            none_conf_comp,
            ins_conf_comp,
            pse_conf_comp,
            checked_comp,
            status_comp,
            id_selector_comp,
        ],
    )

demo.launch(server_name="0.0.0.0", server_port=7862)
