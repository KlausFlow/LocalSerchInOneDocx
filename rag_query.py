"""CLI-скрипт для MVP RAG по одному txt-документу.

На текущем этапе реализованы только разбор и проверка аргументов.
Загрузка документа, локальный retrieval и формирование ответа будут
добавлены на следующих этапах.
"""

import argparse
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


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = parse_args(argv)

    try:
        validate_args(args)
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    print("CLI и проверки аргументов работают.")
    print("Текущая конфигурация:")
    print(f"document: {args.document}")
    print(f"question: {args.question}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"overlap: {args.overlap}")
    print(f"top_k: {args.top_k}")
    print(f"retriever: {args.retriever}")
    print(f"answerer: {args.answerer}")
    print(
        "Загрузка документа, retrieval и answerer будут реализованы "
        "на следующих этапах."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
