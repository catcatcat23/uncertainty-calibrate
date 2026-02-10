import os
import numpy as np
import pandas as pd
from PIL import Image

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ========= 1. 基本路径和 CSV 配置（根据你实际情况改） =========

# 图像根目录（用来把 ref_image / insp_image 的相对路径拼成绝对路径）
IMG_ROOT = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310"

# 四个 CSV 的路径 + tag 信息
# ⚠️ 如果文件名不完全一样，这里改成你真实的名字
CSV_CONFIGS = [
    {
        "path": os.path.join(
            IMG_ROOT,
            "aug_train_pair_labels_singlepinpad_250310_final_rgb_DA758_black_uuid_dropoldpairs_period2.csv",
        ),
        "part_name": "singlepinpad",
        "split": "train",
    },
    {
        "path": os.path.join(
            IMG_ROOT,
            "aug_test_pair_labels_singlepinpad_250310_final_rgb_DA758_black_uuid_dropoldpairs_period2.csv",
        ),
        "part_name": "singlepinpad",
        "split": "test",
    },
    {
        "path": os.path.join(
            IMG_ROOT,
            "aug_train_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs_period2.csv",
        ),
        "part_name": "singlepad",
        "split": "train",
    },
    {
        "path": os.path.join(
            IMG_ROOT,
            "aug_test_pair_labels_singlepad_250310_final_rgb_DA758_black_uuid_dropoldpairs_period2.csv",
        ),
        "part_name": "singlepad",
        "split": "test",
    },
]

# Chroma 持久化目录（库会建在这里）
CHROMA_DIR = "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310"

# 一个 collection 放 4 份 csv，一起管理
COLLECTION_NAME = "pairs_singlepad_singlepinpad_251112"

# =====================================================


def load_image(path: str) -> Image.Image:
    """辅助函数：读取图片并转成 RGB"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def embed_pair(model: SentenceTransformer, ref_path: str, insp_path: str) -> np.ndarray:
    """
    对一对 (ref, insp) 图像做 embedding：
      feat0 = ref 图向量
      feat1 = insp 图向量

    最终向量 = [feat1, feat0, feat1 - feat0]
    用于后面比较两张图的相似/差异。
    """
    ref_img = load_image(ref_path)
    insp_img = load_image(insp_path)

    feats = model.encode(
        [ref_img, insp_img],
        convert_to_numpy=True,
        normalize_embeddings=True,  # 每个向量单位长度
    )
    feat0, feat1 = feats[0], feats[1]  # (D,)

    diff = feat1 - feat0
    pair_feat = np.concatenate([feat1, feat0, diff], axis=-1)  # (3D,)

    return pair_feat


def main():
    # ========= 2. 初始化 Chroma =========
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(allow_reset=True),
    )

    # 如果想完全重建 collection（从零开始），先删掉旧的
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # 我们自己算 embedding
    )

    # ========= 3. 加载 CLIP 模型 =========
    print("[INFO] Loading CLIP model (clip-ViT-B-32) ...")
    model = SentenceTransformer("clip-ViT-B-32")

    # ========= 4. 遍历 4 个 CSV =========
    for cfg_idx, cfg in enumerate(CSV_CONFIGS):
        csv_path = cfg["path"]
        part_name = cfg["part_name"]
        split = cfg["split"]

        print(f"\n[INFO] ({cfg_idx + 1}/{len(CSV_CONFIGS)}) Processing CSV: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"[WARN] CSV not found, skip: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # 有些 csv 可能把 index 存成 "Unnamed: 0"，这里统一成 "index"
        if "index" not in df.columns:
            if "Unnamed: 0" in df.columns:
                df = df.rename(columns={"Unnamed: 0": "index"})
            else:
                raise ValueError(f"CSV {csv_path} 缺少 index 列")

        # 检查必要列
        required_cols = ["index", "ref_image", "insp_image", "insp_defect_label"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV {csv_path} 缺少必要列: {col}")

        csv_name = os.path.basename(csv_path)

        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []

        # 用 enumerate 的 row_idx 只用于保证 id 唯一，不写回 csv
        for row_idx, row in df.iterrows():
            csv_index = int(row["index"])
            ref_rel = row["ref_image"]
            insp_rel = row["insp_image"]
            insp_defect_label = row["insp_defect_label"]

            ref_abs = os.path.join(IMG_ROOT, ref_rel)
            insp_abs = os.path.join(IMG_ROOT, insp_rel)

            # 生成向量
            emb = embed_pair(model, ref_abs, insp_abs)
            emb_list = emb.astype(float).tolist()

            # 唯一 id：part + split + csv 文件名 + 行号
            pair_id = f"{part_name}::{split}::{csv_name}::{row_idx}"

            # group_key: 同一 part 下，同一对图像的逻辑 key
            group_key = f"{part_name}::{ref_rel}::{insp_rel}"

            metadata = {
                # 来源信息
                "part_name": part_name,
                "split": split,
                "csv_name": csv_name,
                "csv_index": csv_index,      # ⭐ 回写只靠这个就够了

                # 逻辑分组 key（相同 pair 的样本共享这个）
                "group_key": group_key,

                # 相对路径
                "ref_image": ref_rel,
                "insp_image": insp_rel,

                # 原始标签
                "insp_defect_label": str(insp_defect_label),

                # 预留标注字段（初始为空）
                "NONE_CONF": -1.0,
                "INSUFFICIENT_SOLDER_CONF": -1.0,
                "PSEUDO_SOLDER_CONF": -1.0,
                "checked": False,
            }

            batch_ids.append(pair_id)
            batch_embeddings.append(emb_list)
            batch_metadatas.append(metadata)

            # 每 64 条写一次，防止一次性太大
            if len(batch_ids) >= 64:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                print(
                    f"  [INFO] Added batch of {len(batch_ids)} rows "
                    f"({part_name}, {split})"
                )
                batch_ids, batch_embeddings, batch_metadatas = [], [], []

        # 把最后还没写进去的补上
        if batch_ids:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )
            print(
                f"  [INFO] Added final batch of {len(batch_ids)} rows for {csv_path}"
            )

    print("\n[INFO] All CSVs processed. Chroma collection built successfully.")


if __name__ == "__main__":
    main()
