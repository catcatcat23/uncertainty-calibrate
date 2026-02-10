import pandas as pd

CSV_PATH = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310/checked_samples.csv"

df = pd.read_csv(CSV_PATH)
df = df[df["part_name"] == "singlepad"].copy().reset_index(drop=True)  # ✅ 只保留 singlepad

print(f'数据量{len(df)}')
print("列名：")
print(list(df.columns))

# 检查 key 是否唯一
if {"ref_image", "insp_image", "id"}.issubset(df.columns):
    dup_num = df.duplicated(subset=["ref_image", "insp_image"]).sum()
    print(f"\n(ref_image, insp_image) 重复行数: {dup_num}")
else:
    print("\n没有找到 ref_image / insp_image / id 这几列")

# import pandas as pd

# CSV_PATH = "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310/checked_samples.csv"
# df = pd.read_csv(CSV_PATH)

# # 按 (ref_image, insp_image) 去重，只保留第一次出现的那一行
# df_clean = df.drop_duplicates(subset=["ref_image", "insp_image"], keep="first")

# print("去重前行数:", len(df))
# print("去重后行数:", len(df_clean))

# df_clean.to_csv(
#     "/home/cat/workspace/defect_data/defect_DA758_black_uuid_250310/send2terminal/250310/checked_samples_clean.csv",
#     index=False,
#     encoding="utf-8-sig",
# )
# print("已保存为 checked_samples_clean.csv")
