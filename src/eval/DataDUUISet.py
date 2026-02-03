# File fokusses on creating a DUUI dataset for testing the ability of the RAGBot especially for testing the Query-Results

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import json
import re


Difficulty = Literal["easy", "medium", "hard"]
Category = str


@dataclass(frozen=True)
class DUUIDataset:
    """
    One evaluation item for a DUUI-focused RAG benchmark.

    - question: user query
    - gold_answer: reference answer (short, factual, checkable)
    - required_context: high-level labels/keywords that should be needed
    - relevant_chunks: identifiers you use to point to ground-truth sources
      (e.g., file paths, doc anchors, chunk_ids)
    - negative_chunks: optional "distractor" chunks that should NOT be used
    """
    id: str
    category: Category
    difficulty: Difficulty
    keywords: List[str]
    question: str
    gold_answer: str
    required_context: List[str] = field(default_factory=list)
    relevant_chunks: List[str] = field(default_factory=list)
    negative_chunks: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def validate(self) -> None:
        """Raise ValueError if this sample is malformed."""
        if not re.match(r"^[a-zA-Z0-9_\-]+$", self.id):
            raise ValueError(f"Invalid id '{self.id}'. Use [a-zA-Z0-9_-].")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Invalid difficulty '{self.difficulty}'.")
        if not self.question.strip():
            raise ValueError("question must be non-empty.")
        if not self.gold_answer.strip():
            raise ValueError("gold_answer must be non-empty.")
        # optional but helpful: ensure unique lists
        if len(set(self.required_context)) != len(self.required_context):
            raise ValueError(f"required_context contains duplicates in sample {self.id}.")
        if len(set(self.relevant_chunks)) != len(self.relevant_chunks):
            raise ValueError(f"relevant_chunks contains duplicates in sample {self.id}.")
        if len(set(self.negative_chunks)) != len(self.negative_chunks):
            raise ValueError(f"negative_chunks contains duplicates in sample {self.id}.")


@dataclass
class DUUIEvalDataset:
    """
    A dataset container with helpers for:
    - add/validate
    - save/load JSONL or JSON
    - simple filtering

    JSONL format: one JSON object per line (recommended for versioning).
    """
    name: str = "duui_rag_eval"
    samples: List[DUUIDataset] = field(default_factory=list)

    def add(self, sample: DUUIDataset, validate: bool = True) -> None:
        if validate:
            sample.validate()
        if any(s.id == sample.id for s in self.samples):
            raise ValueError(f"Duplicate sample id '{sample.id}'.")
        self.samples.append(sample)

    def validate(self) -> None:
        seen = set()
        for s in self.samples:
            s.validate()
            if s.id in seen:
                raise ValueError(f"Duplicate sample id '{s.id}' in dataset.")
            seen.add(s.id)

    def filter(
        self,
        *,
        category: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> "DUUIEvalDataset":
        ids_set = set(ids) if ids else None
        out = DUUIEvalDataset(name=f"{self.name}_filtered")
        for s in self.samples:
            if category is not None and s.category != category:
                continue
            if difficulty is not None and s.difficulty != difficulty:
                continue
            if ids_set is not None and s.id not in ids_set:
                continue
            out.samples.append(s)
        return out

    # ---------- Serialization ----------

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    @staticmethod
    def from_jsonl(path: str | Path, *, name: Optional[str] = None) -> "DUUIEvalDataset":
        path = Path(path)
        ds = DUUIEvalDataset(name=name or path.stem)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sample = DUUIDataset(**obj)
                ds.add(sample, validate=True)
        return ds

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "samples": [asdict(s) for s in self.samples],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "DUUIEvalDataset":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        ds = DUUIEvalDataset(name=payload.get("name", path.stem))
        for obj in payload.get("samples", []):
            ds.add(DUUIDataset(**obj), validate=True)
        return ds


# ---------------- Example usage ----------------
if __name__ == "__main__":
    ds = DUUIEvalDataset(name="duui_eval_v1")

    ds.add(
        DUUIDataset(
            id="duui-001",
            category="pipeline_debugging",
            difficulty="easy",
            question="Warum findet DUUI meine Komponente zur Laufzeit nicht?",
            gold_answer="Die Komponente ist nicht (oder falsch) in der Pipeline-Konfiguration registriert; "
                        "prüfe ComponentRegistration/PipelineYAML und den exakten Namen/Tag.",
            required_context=["ComponentRegistration", "PipelineYAML"],
            relevant_chunks=["docs/pipeline.md#registration", "examples/pipeline.yaml"],
            negative_chunks=["docs/typesystem.md"],
            notes="Negativ: Antwort soll nicht behaupten, dass Docker kaputt ist, wenn Kontext das nicht sagt.",
        )
    )

    ds.validate()
    ds.to_jsonl("duui_eval_v1.jsonl")
    loaded = DUUIEvalDataset.from_jsonl("duui_eval_v1.jsonl")
    print(f"Loaded {len(loaded.samples)} samples from JSONL.")
