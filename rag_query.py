"""CLI-скрипт для MVP RAG по одному txt-документу.

На текущем этапе реализованы разбор аргументов, проверки, загрузка
txt-документа, нормализация текста, чанкинг по словам и локальный
retrieval. Формирование ответа будет добавлено на следующем этапе.
"""

import argparse
from collections import Counter
import math
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CHUNK_SIZE = 80
DEFAULT_OVERLAP = 20
DEFAULT_TOP_K = 3
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_ANSWERER = "simple"
MIN_RETRIEVAL_SCORE = 0.45
NO_ANSWER_MESSAGE = "В документе нет достаточной информации для ответа."
DEFAULT_MAX_ANSWER_SENTENCES = 3


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


def tokenize_words(text: str) -> list[str]:
    """Вернуть lowercase-токены из русских/английских букв и цифр."""
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text.lower())


def bm25_scores(
    question: str,
    chunks: list[dict[str, object]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Посчитать BM25 score для каждого чанка."""
    if not chunks:
        return []

    query_terms = tokenize_words(question)
    if not query_terms:
        return [0.0] * len(chunks)

    chunk_tokens = [tokenize_words(str(chunk["text"])) for chunk in chunks]
    chunk_lengths = [len(tokens) for tokens in chunk_tokens]
    avgdl = sum(chunk_lengths) / len(chunk_lengths)
    if avgdl == 0:
        return [0.0] * len(chunks)

    term_frequencies = [Counter(tokens) for tokens in chunk_tokens]
    query_vocabulary = set(query_terms)
    document_frequency = {
        term: sum(1 for tokens in chunk_tokens if term in tokens)
        for term in query_vocabulary
    }

    n_chunks = len(chunks)
    scores: list[float] = []
    for frequencies, chunk_length in zip(term_frequencies, chunk_lengths):
        score = 0.0
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            df = document_frequency[term]
            idf = math.log(1 + (n_chunks - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * chunk_length / avgdl)
            score += idf * (tf * (k1 + 1)) / denominator
        scores.append(score)

    return scores


def tfidf_char_scores(
    question: str,
    chunks: list[dict[str, object]],
) -> list[float]:
    """Посчитать TF-IDF char n-gram similarity для каждого чанка."""
    if not chunks:
        return []
    if not question.strip():
        return [0.0] * len(chunks)

    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    if not any(text.strip() for text in chunk_texts):
        return [0.0] * len(chunks)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    try:
        matrix = vectorizer.fit_transform([question, *chunk_texts])
    except ValueError:
        return [0.0] * len(chunks)

    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    return [float(score) for score in similarities]


def normalize_scores(scores: list[float]) -> list[float]:
    """Привести scores к диапазону 0..1."""
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def retrieve_top_chunks(
    question: str,
    chunks: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    """Вернуть top-k релевантных чанков по hybrid retrieval."""
    if not chunks or top_k <= 0:
        return []

    raw_bm25 = bm25_scores(question, chunks)
    raw_tfidf = tfidf_char_scores(question, chunks)
    bm25 = normalize_scores(raw_bm25)
    tfidf = normalize_scores(raw_tfidf)
    if len(chunks) == 1:
        bm25 = [1.0 if raw_bm25 and raw_bm25[0] > 0 else 0.0]
        tfidf = [1.0 if raw_tfidf and raw_tfidf[0] > 0 else 0.0]

    final_scores = [
        0.6 * bm25_score + 0.4 * tfidf_score
        for bm25_score, tfidf_score in zip(bm25, tfidf)
    ]

    if not final_scores or max(final_scores) == 0:
        return []

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: final_scores[index],
        reverse=True,
    )

    retrieved_chunks: list[dict[str, object]] = []
    for index in ranked_indices:
        if final_scores[index] < MIN_RETRIEVAL_SCORE:
            continue

        chunk = chunks[index]
        retrieved_chunks.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "score": float(final_scores[index]),
                "bm25_score": float(bm25[index]),
                "tfidf_score": float(tfidf[index]),
            }
        )
        if len(retrieved_chunks) == top_k:
            break

    return retrieved_chunks


def split_into_sentences(text: str) -> list[str]:
    """Разбить текст на простые предложения."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", ". ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [" ".join(sentence.split()) for sentence in sentences if sentence.strip()]


def sentence_overlap_score(question: str, sentence: str) -> float:
    """Оценить пересечение токенов вопроса и предложения."""
    question_tokens = set(tokenize_words(question))
    if not question_tokens:
        return 0.0

    sentence_tokens = set(tokenize_words(sentence))
    common_tokens = question_tokens & sentence_tokens
    score = len(common_tokens) / len(question_tokens)

    sentence_lower = sentence.lower()
    for token in question_tokens:
        if len(token) >= 4 and token in sentence_lower:
            score += 0.05

    return score


def is_valid_answer_sentence(sentence: str) -> bool:
    """Проверить, что предложение достаточно полное для ответа."""
    sentence = sentence.strip()
    tokens = tokenize_words(sentence)
    if len(tokens) < 3:
        return False
    if len(sentence) < 20:
        return False
    return True


def token_overlap_ratio(left: str, right: str) -> float:
    """Оценить долю пересечения токенов между двумя предложениями."""
    left_tokens = set(tokenize_words(left))
    right_tokens = set(tokenize_words(right))
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = left_tokens & right_tokens
    return len(intersection) / min(len(left_tokens), len(right_tokens))


def is_duplicate_sentence(sentence: str, selected_sentences: list[str]) -> bool:
    """Проверить, повторяет ли предложение уже выбранный смысловой фрагмент."""
    return any(
        token_overlap_ratio(sentence, selected) >= 0.7
        for selected in selected_sentences
    )


def select_unique_valid_sentences(
    sentences: list[str],
    max_sentences: int,
) -> list[str]:
    """Выбрать валидные предложения без коротких обрубков и смысловых дублей."""
    selected_sentences: list[str] = []
    for sentence in sentences:
        normalized_sentence = " ".join(sentence.split())
        if not is_valid_answer_sentence(normalized_sentence):
            continue
        if is_duplicate_sentence(normalized_sentence, selected_sentences):
            continue

        selected_sentences.append(normalized_sentence)
        if len(selected_sentences) == max_sentences:
            break

    return selected_sentences


def generate_simple_answer(
    question: str,
    retrieved_chunks: list[dict[str, object]],
    max_sentences: int = DEFAULT_MAX_ANSWER_SENTENCES,
) -> str:
    """Собрать краткий extractive-ответ из найденных чанков."""
    if not retrieved_chunks:
        return NO_ANSWER_MESSAGE

    seen_sentences: set[str] = set()
    scored_sentences: list[tuple[float, int, int, str]] = []
    fallback_candidates: list[str] = []

    for chunk_index, chunk in enumerate(retrieved_chunks):
        sentences = split_into_sentences(str(chunk["text"]))
        if chunk_index == 0:
            fallback_candidates = sentences

        for sentence_index, sentence in enumerate(sentences):
            normalized_sentence = " ".join(sentence.split())
            if not is_valid_answer_sentence(normalized_sentence):
                continue

            sentence_key = normalized_sentence.lower()
            if sentence_key in seen_sentences:
                continue

            seen_sentences.add(sentence_key)
            score = sentence_overlap_score(question, normalized_sentence)
            if score > 0:
                scored_sentences.append(
                    (score, chunk_index, sentence_index, normalized_sentence)
                )

    if scored_sentences:
        scored_sentences.sort(key=lambda item: (-item[0], item[1], item[2]))
        ordered_candidates = [
            sentence for _, _, _, sentence in scored_sentences
        ]
        selected_sentences = select_unique_valid_sentences(
            ordered_candidates,
            max_sentences,
        )
    else:
        selected_sentences = select_unique_valid_sentences(
            fallback_candidates,
            min(2, max_sentences),
        )

    if not selected_sentences:
        return NO_ANSWER_MESSAGE

    answer = " ".join(selected_sentences).strip()
    answer_sentences = split_into_sentences(answer)
    selected_answer_sentences = select_unique_valid_sentences(
        answer_sentences,
        max_sentences,
    )

    answer = " ".join(selected_answer_sentences).strip()
    if not answer:
        return NO_ANSWER_MESSAGE

    return answer


def build_prompt(question: str, retrieved_chunks: list[dict[str, object]]) -> str:
    """Собрать prompt для будущей LLM только из найденных фрагментов."""
    lines = [
        "Отвечай только по найденным фрагментам документа.",
        (
            "Если информации недостаточно, напиши: "
            f"{NO_ANSWER_MESSAGE}"
        ),
        "",
        f"Вопрос: {question}",
        "",
        "Фрагменты:",
    ]

    if not retrieved_chunks:
        lines.append("Релевантные фрагменты не найдены.")
    else:
        for chunk in retrieved_chunks:
            lines.append(f"chunk {chunk['id']}: {chunk['text']}")

    return "\n".join(lines)


def make_preview(text: str, limit: int = 200) -> str:
    """Сделать однострочный preview текста с ограничением длины."""
    preview = " ".join(text.split())
    if len(preview) > limit:
        return preview[:limit].rstrip() + "..."
    return preview


def print_result(
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, object]],
) -> None:
    """Вывести финальный результат MVP."""
    print("Вопрос:")
    print(question)
    print()
    print("Ответ:")
    print(answer)
    print()
    print("Источники:")
    if not retrieved_chunks:
        print("Релевантные фрагменты не найдены.")
        return

    for source_number, chunk in enumerate(retrieved_chunks, start=1):
        preview = make_preview(str(chunk["text"]), limit=300)
        print(
            f"{source_number}) "
            f"chunk {chunk['id']}, "
            f"score={float(chunk['score']):.3f}, "
            f"words={chunk['start_word']}-{chunk['end_word']}"
        )
        print(preview)
        print()


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = parse_args(argv)

    try:
        validate_args(args)
        text = load_document(args.document)
        chunks = split_into_chunks(text, args.chunk_size, args.overlap)
        retrieved_chunks = retrieve_top_chunks(args.question, chunks, args.top_k)
        answer = generate_simple_answer(args.question, retrieved_chunks)
    except ValueError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2

    print_result(args.question, answer, retrieved_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
