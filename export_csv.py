import os
import csv
from typing import List, Dict

import chromadb
from chromadb.config import Settings

# ===== 跟标注界面保持一致的配置 =====
CHROMA_DIR = "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310"
COLLECTION_NAME = "pairs_singlepad_singlepinpad_251112"

# 导出文件名（在当前目录下生成）
OUT_CSV = "checked_samples.csv"


def main():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(allow_reset=False),
    )
    collection = client.get_collection(COLLECTION_NAME)

    # 只导出 checked == True 的样本
    where = {"checked": True}

    rows: List[Dict] = []
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

        for _id, m in zip(ids, metas):
            row = {
                "id": _id,
                "csv_name": m.get("csv_name"),
                "csv_index": m.get("csv_index"),
                "part_name": m.get("part_name"),
                "split": m.get("split"),
                "insp_defect_label": m.get("insp_defect_label"),

                "NONE_CONF": m.get("NONE_CONF"),
                "INSUFFICIENT_SOLDER_CONF": m.get("INSUFFICIENT_SOLDER_CONF"),
                "PSEUDO_SOLDER_CONF": m.get("PSEUDO_SOLDER_CONF"),

                "checked": m.get("checked"),
                "ref_image": m.get("ref_image"),
                "insp_image": m.get("insp_image"),
            }
            rows.append(row)

        offset += len(ids)
        print(f"[INFO] fetched {offset} rows so far...")

    # 如果一条都没有，直接提示
    if not rows:
        print("No checked samples found.")
        return

    fieldnames = [
        "id",
        "csv_name",
        "csv_index",
        "part_name",
        "split",
        "insp_defect_label",
        "NONE_CONF",
        "INSUFFICIENT_SOLDER_CONF",
        "PSEUDO_SOLDER_CONF",
        "checked",
        "ref_image",
        "insp_image",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Exported {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
