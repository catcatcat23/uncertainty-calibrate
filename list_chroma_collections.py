import chromadb
from chromadb.config import Settings

CHROMA_DIR = "/home/cat/workspace/vlm/chroma_db/defect_DA758_black_uuid_250310"

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False),
)

cols = client.list_collections()
print("=== collections in this DB ===")
for c in cols:
    print(" -", c.name)
