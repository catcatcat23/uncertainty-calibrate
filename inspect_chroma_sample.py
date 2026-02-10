import numpy as np
import chromadb
from chromadb.config import Settings

CHROMA_DIR = "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310"
COLLECTION_NAME = "pairs_singlepad_singlepinpad_251112"

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False),
)
collection = client.get_collection(COLLECTION_NAME)
where = { 'checked': True }
# 取前 3 条看看
res = collection.get(
    where=where,
    include=[ "metadatas", "embeddings"],

)
print(len(res["ids"]), "records found with", where)

print("=== Example records ===")
for i in range(3):
    _id = res["ids"][i]
    meta = res["metadatas"][i]
    emb = res["embeddings"][i]

    print(f"\n--- record {i} ---")
    print("id:", _id)
    print("metadata keys:", list(meta.keys()))
    print("metadata:", meta)
    # 看看向量长度
    arr = np.array(emb)
    print("embedding shape:", arr.shape)
