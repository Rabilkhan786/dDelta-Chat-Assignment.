from pathlib import Path

from src.ingest.pdf_native import NativePDFAdapter
from src.delta.align import Aligner
from src.delta.engine import DeltaEngine
from src.chat.index import DocumentIndexer
from src.config.settings import settings

adapter = NativePDFAdapter()

left = adapter.parse(
    Path(settings.paths.revision_a)
)

right = adapter.parse(
    Path(settings.paths.revision_b)
)

aligner = Aligner()
alignment = aligner.align(left, right)

engine = DeltaEngine()
deltas = engine.compare(alignment)

print(f"Total Deltas: {len(deltas)}")

for delta in deltas:
    print("-" * 80)
    print(delta)

indexer = DocumentIndexer()

documents = indexer.create_delta_documents(deltas)

print(f"\nTotal LangChain Documents: {len(documents)}")

for doc in documents:
    print("=" * 80)
    print(doc.page_content)
    print(doc.metadata)
    
print("=" * 80)
print("MATCHES")
print("=" * 80)

for i, match in enumerate(alignment.matches[:20], start=1):
    print(f"\nMatch {i}")
    print("-" * 80)
    print("LEFT :", repr(match.left.text))
    print("RIGHT:", repr(match.right.text))
    print("Score:", match.score)
    
print("\nLEFT UNMATCHED")
print("=" * 80)

for e in alignment.unmatched_left[:20]:
    print(repr(e.text))

print("\nRIGHT UNMATCHED")
print("=" * 80)

for e in alignment.unmatched_right[:20]:
    print(repr(e.text))