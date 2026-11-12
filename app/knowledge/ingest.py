from pathlib import Path


SUPPORTED_SUFFIXES = {".md", ".txt"}
CHUNK_CHARS = 1000


def load_documents(source_dir):
    root = Path(source_dir)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in SUPPORTED_SUFFIXES:
            yield path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")


def split_paragraph(paragraph):
    return [paragraph[start:start + CHUNK_CHARS] for start in range(0, len(paragraph), CHUNK_CHARS)]


def chunk_text(text):
    pieces = []
    current = ""
    for paragraph in (block.strip() for block in text.split("\n\n")):
        if not paragraph:
            continue
        for part in split_paragraph(paragraph):
            if current and len(current) + len(part) + 2 > CHUNK_CHARS:
                pieces.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def build_chunks(source_dir):
    chunks = []
    for source, text in load_documents(source_dir):
        for index, piece in enumerate(chunk_text(text)):
            chunks.append({"chunk_id": f"{source}#{index}", "source": source, "text": piece})
    return chunks
