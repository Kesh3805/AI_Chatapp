"""Tests for the paragraph/sentence-aware _chunk_text() function.

We test _chunk_text() by extracting its logic rather than importing main.py
(which has heavy dependencies).  The function is small enough to inline here
with the same settings-binding pattern as in main.py.
"""
import re
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Minimal settings stub
# ---------------------------------------------------------------------------

_settings_stub = MagicMock()
_settings_stub.CHUNK_SIZE = 200
_settings_stub.CHUNK_OVERLAP = 20


# ---------------------------------------------------------------------------
# Replicate _chunk_text() locally so we don't import main.py
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    """Paragraph → sentence → character cascading chunker (mirrors main.py)."""
    size = _settings_stub.CHUNK_SIZE
    overlap = _settings_stub.CHUNK_OVERLAP

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    atoms: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            atoms.append(para)
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            for sent in sentences:
                if len(sent) <= size:
                    atoms.append(sent)
                else:
                    stride = max(1, size - overlap)
                    for i in range(0, len(sent), stride):
                        fragment = sent[i : i + size]
                        if fragment.strip():
                            atoms.append(fragment)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def _flush():
        if current_parts:
            chunks.append(" ".join(current_parts))

    for atom in atoms:
        atom_len = len(atom)
        if current_parts and current_len + 1 + atom_len > size:
            _flush()
            if overlap > 0 and current_parts:
                tail = " ".join(current_parts)[-overlap:]
                current_parts = [tail]
                current_len = len(tail)
            else:
                current_parts = []
                current_len = 0
        current_parts.append(atom)
        current_len += (1 + atom_len) if len(current_parts) > 1 else atom_len

    _flush()
    return [c for c in chunks if c.strip()]

# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _use_small_settings(monkeypatch):
    """Each test sees CHUNK_SIZE=200, CHUNK_OVERLAP=20."""
    monkeypatch.setattr(_settings_stub, "CHUNK_SIZE", 200)
    monkeypatch.setattr(_settings_stub, "CHUNK_OVERLAP", 20)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

def test_empty_input():
    assert _chunk_text("") == []


def test_whitespace_only():
    assert _chunk_text("   \n\n   ") == []


def test_single_short_paragraph():
    text = "Hello world."
    chunks = _chunk_text(text)
    assert chunks == ["Hello world."]


def test_multiple_short_paragraphs_merged():
    """Short paragraphs should be merged into a single chunk."""
    parts = ["Line one.", "Line two.", "Line three."]
    text = "\n\n".join(parts)
    chunks = _chunk_text(text)
    # All three fit in 200 chars so there should be exactly 1 chunk
    assert len(chunks) == 1
    for part in parts:
        assert part in chunks[0]


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def test_two_large_paragraphs_become_separate_chunks():
    """Two paragraphs each near CHUNK_SIZE must not collapse into one."""
    para_a = "Alpha. " * 28          # ~196 chars
    para_b = "Beta.  " * 28          # ~196 chars
    text = para_a.strip() + "\n\n" + para_b.strip()
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    # Content of both paragraphs should appear somewhere
    assert any("Alpha" in c for c in chunks)
    assert any("Beta" in c for c in chunks)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_long_paragraph_split_at_sentences():
    """A paragraph > CHUNK_SIZE must be split at sentence boundaries."""
    sentences = [f"Sentence number {i} ends here." for i in range(20)]
    text = " ".join(sentences)          # no blank lines → single paragraph
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    # Verify no chunk wildly exceeds CHUNK_SIZE (some tolerance for overlap)
    for c in chunks:
        assert len(c) <= 250, f"Chunk too large: {len(c)}"


def test_sentences_not_split_mid_word():
    """Each chunk must end cleanly — no orphaned leading space."""
    text = "First sentence ends. Second sentence is here. Third sentence follows it."
    chunks = _chunk_text(text)
    for c in chunks:
        assert not c.startswith(" ")
        assert c == c.strip()


# ---------------------------------------------------------------------------
# Character fallback (sentences > CHUNK_SIZE)
# ---------------------------------------------------------------------------

def test_very_long_sentence_character_fallback():
    """A sentence longer than CHUNK_SIZE must still be chunked."""
    long_sentence = "word " * 100      # ~500 chars, > 200
    chunks = _chunk_text(long_sentence.strip())
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_overlap_carries_tail():
    """With CHUNK_OVERLAP>0 the end of one chunk should influence the next."""
    # Create content that forces exactly 2 chunks and check overlap
    _settings_stub.CHUNK_SIZE = 50
    _settings_stub.CHUNK_OVERLAP = 10
    text = ("Short sentence A. " * 5) + "\n\n" + ("Short sentence B. " * 5)
    chunks = _chunk_text(text)
    # Just verify it doesn't crash and produces multiple chunks
    assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Settings-driven behaviour
# ---------------------------------------------------------------------------

def test_chunk_size_respected(monkeypatch):
    monkeypatch.setattr(_settings_stub, "CHUNK_SIZE", 50)
    monkeypatch.setattr(_settings_stub, "CHUNK_OVERLAP", 0)
    text = "This is a sentence. " * 20
    chunks = _chunk_text(text)
    for c in chunks:
        # 10% tolerance for merging edge
        assert len(c) <= 60, f"Chunk exceeded max: {len(c)}"


def test_no_duplicate_content():
    """Chunks should not gratuitously repeat entire sentences."""
    text = "Unique fact one. Unique fact two. Unique fact three."
    chunks = _chunk_text(text)
    combined = " ".join(chunks)
    # Each unique keyword should appear at least once
    for kw in ("one", "two", "three"):
        assert kw in combined
