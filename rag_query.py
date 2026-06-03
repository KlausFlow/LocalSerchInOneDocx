"""CLI-скрипт для MVP RAG по одному txt-документу.

На текущем этапе реализованы разбор аргументов, проверки, загрузка
txt-документа, нормализация текста и чанкинг по словам. Локальный
retrieval и формирование ответа будут добавлены на следующих этапах.
"""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_CHUNK_SIZE = 80
DEFAULT_OVERLAP = 20
DEFAULT_TOP_K = 3
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_ANSWERER = "simple"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="MVP RAG по одному txt-документу.",
    )
    parser.add_argument(
        "--document",
        required=True,
        help="Путь к txt-документу.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Вопрос пользователя.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Размер чанка в словах. По умолчанию: {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Перекрытие чанков. По умолчанию: {DEFAULT_OVERLAP}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Число источников. По умолчанию: {DEFAULT_TOP_K}.",
    )
    parser.add_argument(
        "--retriever",
        choices=["hybrid"],
        default=DEFAULT_RETRIEVER,
        help=f"Тип retrieval. По умолчанию: {DEFAULT_RETRIEVER}.",
    )
    parser.add_argument(
        "--answerer",
        choices=["simple"],
        default=DEFAULT_ANSWERER,
        help=f"Тип answerer. По умолчанию: {DEFAULT_ANSWERER}.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Проверить пользовательские аргументы."""
    document_path = Path(args.document)

    if not document_path.exists():
        raise ValueError("Документ не найден.")
    if not document_path.is_file():
        raise ValueError("Путь к документу должен указывать на файл.")
    if not args.question.strip():
        raise ValueError("Вопрос не должен быть пустым.")
    if args.chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше 0.")
    if args.overlap < 0:
        raise ValueError("Перекрытие чанков не должно быть отрицательным.")
    if args.overlap >= args.chunk_size:
        raise ValueError("Перекрытие чанков должно быть меньше размера чанка.")
    if args.top_k <= 0:
        raise ValueError("Число источников top_k должно быть больше 0.")


def load_document(path: str | Path) -> str:
    """Загрузить поддерживаемый документ и вернуть нормализованный текст."""
    document_path = Path(path)

    if document_path.suffix.lower() == ".txt":
        return load_txt(document_path)

    raise ValueError("Сейчас поддерживаются только txt-документы.")


def load_txt(path: Path) -> str:
    """Прочитать txt-документ в UTF-8 и нормализовать его содержимое."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Не удалось прочитать txt-документ в UTF-8.") from error
    except OSError as error:
        raise ValueError("Не удалось прочитать документ.") from error

    normalized_text = normalize_text(text)
    if not normalized_text:
        raise ValueError("Документ пустой.")

    return normalized_text


def normalize_text(text: str) -> str:
    """Нормализовать пробелы и переносы строк без изменения смысла текста."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    normalized_text = "\n".join(lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)

    return normalized_text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict[str, object]]:
    """Разбить нормализованный текст на чанки по словам."""
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше 0.")
    if overlap < 0:
        raise ValueError("Перекрытие чанков не должно быть отрицательным.")
    if overlap >= chunk_size:
        raise ValueError("Перекрытие чанков должно быть меньше размера чанка.")

    words = text.split()
    if not words:
        raise ValueError("Документ пустой.")

    def make_chunk(chunk_id: int, start: int, end: int) -> dict[str, object]:
        return {
            "id": chunk_id,
            "text": " ".join(words[start:end]),
            "start_word": start + 1,
            "end_word": end,
        }

    if len(words) <= chunk_size:
        return [make_chunk(1, 0, len(words))]

    chunks: list[dict[str, object]] = []
    step = chunk_size - overlap
    start = 0

    while start + chunk_size <= len(words):
        chunks.append(make_chunk(len(chunks) + 1, start, start + chunk_size))
        start += step

    last_full_end = int(chunks[-1]["end_word"])
    tail_size = len(words) - last_full_end

    if tail_size == 0:
        return chunks

    tail_threshold = chunk_size * 0.4
    if tail_size <= tail_threshold:
        last_start = int(chunks[-1]["start_word"]) - 1
        chunks[-1]["text"] = " ".join(words[last_start:])
        chunks[-1]["end_word"] = len(words)
        return chunks

    tail_start = last_full_end
    chunk_start = max(0, tail_start - overlap)
    chunks.append(make_chunk(len(chunks) + 1, chunk_start, len(words)))

    return chunks


def make_preview(text: str, limit: int = 200) -> str:
    """Сделать однострочный preview текста с ограничением длины."""
    preview = " ".join(text.split())
    if len(preview) > limit:
        return preview[:limit].rstrip() + "..."
    return preview


def print_pipeline_stub_summary(
    args: argparse.Namespace,
    text: str,
    chunks: list[dict[str, object]],
) -> None:
    """Вывести единый summary для текущего состояния pipeline."""
    print("CLI, загрузка документа и чанкинг работают.")
    print("Текущая конфигурация:")
    print(f"document: {args.document}")
    print(f"question: {args.question}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"overlap: {args.overlap}")
    print(f"top_k: {args.top_k}")
    print(f"retriever: {args.retriever}")
    print(f"answerer: {args.answerer}")

    print("Статистика документа:")
    print(f"characters: {len(text)}")
    print(f"words: {len(text.split())}")
    print(f"chunks: {len(chunks)}")

    print("Preview первых чанков:")
    for chunk in chunks[:3]:
        chunk_id = chunk["id"]
        start_word = chunk["start_word"]
        end_word = chunk["end_word"]
        preview = make_preview(str(chunk["text"]), limit=200)
        print(f"chunk {chunk_id}: {start_word}-{end_word} | {preview}")

    print("Retrieval и answerer будут реализованы на следующих этапах.")


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = parse_args(argv)

    try:
        validate_args(args)
        text = load_document(args.document)
        chunks = split_into_chunks(text, args.chunk_size, args.overlap)
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    print_pipeline_stub_summary(args, text, chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
