"""CLI-скрипт для локального RAG по одному txt-документу.

Скрипт разбирает аргументы командной строки, загружает txt-документ,
нормализует текст, разбивает его на чанки, выполняет локальный retrieval,
формирует простой extractive-ответ и выводит найденные источники.
"""

import argparse
from collections import Counter
from difflib import SequenceMatcher
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
MIN_ANSWER_SENTENCE_SCORE = 0.25
MIN_QUERY_TERM_COVERAGE = 0.5
WHY_ANSWER_CUES = ("потому", "так как", "поскольку", "из-за", "из за", "это объясняется")
SYNONYM_GROUPS = (
    {
        "выкинуть", "выкинули", "выкинул", "выкинула", "выкинуло", "выкинут",
        "выбрасывать", "выбросить", "выбросили", "выбросил", "выбросила",
        "выброшен", "выброшена", "выброшено", "выброшенный", "выброшенную",
        "вышвырнуть", "вышвырнули", "вышвырнул", "вышвырнула",
    },
)
TERM_SYNONYMS = {
    term: group
    for group in SYNONYM_GROUPS
    for term in group
}
RUSSIAN_STOPWORDS = {
    "кто", "что", "какой", "какая", "какие", "какое",
    "как", "почему", "зачем", "где", "когда",
    "это", "этот", "эта", "эти", "такой", "такая", "такие",
    "в", "во", "на", "по", "к", "ко", "с", "со", "из", "от", "до",
    "для", "чего", "чем",
    "и", "а", "но", "или", "же", "ли",
    "был", "была", "были", "будет", "есть",
    "нужен", "нужна", "нужно", "нужны",
    "году", "года", "год",
    "вещь", "вещи", "предмет", "предметы",
}


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


def get_meaningful_query_terms(question: str) -> list[str]:
    """Вернуть смысловые термины вопроса без частых служебных слов."""
    tokens = tokenize_words(question)
    meaningful = [
        token
        for token in tokens
        if token not in RUSSIAN_STOPWORDS
        and (len(token) >= 3 or (token.isascii() and len(token) >= 2))
    ]
    return meaningful or tokens



def normalized_question_start(question: str) -> str:
    """Вернуть начало вопроса в виде нормализованной строки токенов."""
    return " ".join(tokenize_words(question))


def is_definition_question(question: str) -> bool:
    """Проверить, похож ли вопрос на запрос определения сущности."""
    normalized = normalized_question_start(question)
    return (
        normalized.startswith("кто такой ")
        or normalized.startswith("кто такая ")
        or normalized.startswith("кто такие ")
        or normalized.startswith("что такое ")
        or normalized.startswith("что называется ")
        or normalized.startswith("что представляет собой ")
    )


def is_why_question(question: str) -> bool:
    """Проверить, начинается ли вопрос с 'почему'."""
    return normalized_question_start(question).startswith("почему ")


def is_object_question(question: str) -> bool:
    """Проверить, похож ли вопрос на запрос конкретного объекта/списка объектов."""
    normalized = normalized_question_start(question)
    if is_definition_question(question):
        return False
    return normalized.startswith("что ") or normalized.startswith("какие ")


def char_ngrams(token: str, n: int = 3) -> set[str]:
    """Вернуть символьные n-граммы токена для грубого сравнения форм слов."""
    if len(token) < n:
        return {token}
    return {token[index:index + n] for index in range(len(token) - n + 1)}


def char_ngram_overlap(left: str, right: str) -> float:
    """Оценить похожесть слов по общим символьным триграммам."""
    left_ngrams = char_ngrams(left)
    right_ngrams = char_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return len(left_ngrams & right_ngrams) / min(len(left_ngrams), len(right_ngrams))


def terms_match(query_term: str, text_token: str) -> bool:
    """Проверить, совпадают ли термины с учетом небольших различий формы."""
    if query_term == text_token:
        return True
    if text_token in TERM_SYNONYMS.get(query_term, set()):
        return True
    if len(query_term) >= 4 and len(text_token) >= 4:
        if query_term in text_token or text_token in query_term:
            return True
    if query_term[0] != text_token[0]:
        return False
    if SequenceMatcher(None, query_term, text_token).ratio() >= 0.82:
        return True
    if len(query_term) >= 5 and len(text_token) >= 5:
        return char_ngram_overlap(query_term, text_token) >= 0.5
    return False


def matched_query_terms(question: str, text: str) -> set[str]:
    """Вернуть смысловые термины вопроса, найденные в тексте."""
    query_terms = set(get_meaningful_query_terms(question))
    text_tokens = tokenize_words(text)
    matched: set[str] = set()
    for term in query_terms:
        if any(terms_match(term, token) for token in text_tokens):
            matched.add(term)
    return matched

def bm25_scores(
    question: str,
    chunks: list[dict[str, object]],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Посчитать BM25 score для каждого чанка."""
    if not chunks:
        return []

    query_terms = get_meaningful_query_terms(question)
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



def definition_position_boost(question: str, chunk: dict[str, object], index: int, total: int) -> float:
    """Универсальный небольшой бонус ранним фрагментам для вопросов-определений."""
    if not is_definition_question(question):
        return 0.0
    coverage = term_coverage_score(question, str(chunk["text"]))
    if coverage == 0.0:
        return 0.0
    if total <= 1:
        return 0.2 * coverage
    early_score = 1 - index / (total - 1)
    return 0.25 * coverage * early_score


def apply_definition_reranking(
    question: str,
    chunks: list[dict[str, object]],
    scores: list[float],
) -> list[float]:
    """Для однотерминных вопросов-определений сильнее учитывать первое появление термина."""
    if not is_definition_question(question):
        return scores
    if len(get_meaningful_query_terms(question)) > 2:
        return scores
    total = len(chunks)
    if total <= 1:
        return scores

    reranked: list[float] = []
    for index, (chunk, score) in enumerate(zip(chunks, scores)):
        coverage = term_coverage_score(question, str(chunk["text"]))
        if coverage == 0.0:
            reranked.append(score)
            continue
        early_score = 1 - index / (total - 1)
        reranked.append(0.05 * score + 0.95 * coverage * early_score)
    return reranked

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
    final_scores = [
        0.35 * score + 0.65 * best_chunk_sentence_score(question, chunk)
        for score, chunk in zip(final_scores, chunks)
    ]
    final_scores = [
        score + definition_position_boost(question, chunk, index, len(chunks))
        for index, (score, chunk) in enumerate(zip(final_scores, chunks))
    ]
    final_scores = apply_definition_reranking(question, chunks, final_scores)

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
        coverage = term_coverage_score(question, str(chunk["text"]))
        if coverage < MIN_QUERY_TERM_COVERAGE:
            continue

        retrieved_chunks.append(
            {
                "id": chunk["id"],
                "text": chunk["text"],
                "start_word": chunk["start_word"],
                "end_word": chunk["end_word"],
                "score": float(final_scores[index]),
                "bm25_score": float(bm25[index]),
                "tfidf_score": float(tfidf[index]),
                "coverage_score": float(coverage),
            }
        )
        if len(retrieved_chunks) == top_k:
            break

    return retrieved_chunks


def split_into_sentences(text: str) -> list[str]:
    """Разбить текст на простые предложения."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", ". ", text)
    # Многоточие тоже считаем границей предложения: в художественных текстах
    # после него часто начинается новая смысловая фраза.
    text = text.replace("…", "…. ")
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    return [" ".join(sentence.split()) for sentence in sentences if sentence.strip()]


def sentence_overlap_score(question: str, sentence: str) -> float:
    """Оценить пересечение смысловых терминов вопроса и предложения."""
    question_terms = set(get_meaningful_query_terms(question))
    if not question_terms:
        return 0.0

    matched_terms = matched_query_terms(question, sentence)
    score = len(matched_terms) / len(question_terms)

    sentence_lower = sentence.lower()
    for term in question_terms:
        if len(term) >= 4 and term in sentence_lower:
            score += 0.05

    return score


def term_coverage_score(question: str, text: str) -> float:
    """Оценить покрытие смысловых терминов вопроса текстом."""
    question_terms = set(get_meaningful_query_terms(question))
    if not question_terms:
        return 0.0
    return len(matched_query_terms(question, text)) / len(question_terms)


def why_answer_cue_score(question: str, sentence: str) -> float:
    """Дать небольшой бонус предложениям с причинным ответом для why-вопросов."""
    if not is_why_question(question):
        return 0.0
    sentence_lower = sentence.lower().strip(" —-–")
    if sentence_lower.startswith(WHY_ANSWER_CUES):
        return 0.35
    if any(cue in sentence_lower for cue in WHY_ANSWER_CUES):
        return 0.2
    return 0.0


def generic_object_answer_penalty(question: str, sentence: str) -> float:
    """Штрафовать общие рассуждения вместо конкретного объекта в object-вопросах."""
    if not is_object_question(question):
        return 0.0
    sentence_lower = sentence.lower()
    generic_markers = (
        "все, что", "всё, что", "все что", "всё что",
        "всякий", "всякая", "всякое", "любой", "любая", "любое",
        "прочие", "прочих", "предметы", "предметов", "вещи", "вещей",
        "разные предметы", "прочие предметы", "свита", "свитой",
    )
    if any(marker in sentence_lower for marker in generic_markers):
        return 0.7
    if "что бы" in sentence_lower and " ни " in sentence_lower:
        return 0.7
    question_tokens = set(tokenize_words(question))
    sentence_tokens = set(tokenize_words(sentence))
    if "бы" in sentence_tokens and "бы" not in question_tokens:
        return 0.35
    return 0.0


def score_sentence(question: str, sentence: str) -> float:
    """Оценить релевантность предложения вопросу."""
    overlap_score = sentence_overlap_score(question, sentence)
    coverage_score = term_coverage_score(question, sentence)
    cue_score = why_answer_cue_score(question, sentence)
    penalty = generic_object_answer_penalty(question, sentence)
    return max(0.0, 0.45 * overlap_score + 0.45 * coverage_score + cue_score - penalty)


def best_chunk_sentence_score(question: str, chunk: dict[str, object]) -> float:
    """Вернуть лучший sentence-level score внутри чанка."""
    scores = [
        score_sentence(question, sentence)
        for sentence in split_into_sentences(str(chunk["text"]))
        if is_valid_answer_sentence(sentence)
    ]
    if not scores:
        return 0.0
    return max(scores)


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

    if is_definition_question(question):
        max_sentences = min(max_sentences, 1)

    unique_sentences: list[str] = []
    scored_sentences: list[tuple[float, int, int, str]] = []

    for chunk_index, chunk in enumerate(retrieved_chunks):
        sentences = split_into_sentences(str(chunk["text"]))

        for sentence_index, sentence in enumerate(sentences):
            normalized_sentence = " ".join(sentence.split())
            if not is_valid_answer_sentence(normalized_sentence):
                continue

            if is_duplicate_sentence(normalized_sentence, unique_sentences):
                continue

            unique_sentences.append(normalized_sentence)
            score = score_sentence(question, normalized_sentence)
            if score >= MIN_ANSWER_SENTENCE_SCORE:
                scored_sentences.append(
                    (score, chunk_index, sentence_index, normalized_sentence)
                )

    if not scored_sentences:
        return NO_ANSWER_MESSAGE

    if is_why_question(question):
        cue_sentences = [
            item for item in scored_sentences
            if why_answer_cue_score(question, item[3]) > 0
        ]
        if cue_sentences:
            scored_sentences = cue_sentences

    best_score = max(score for score, _, _, _ in scored_sentences)
    dynamic_threshold = max(MIN_ANSWER_SENTENCE_SCORE, best_score * 0.8)
    scored_sentences = [
        item for item in scored_sentences
        if item[0] >= dynamic_threshold
    ]
    if not scored_sentences:
        return NO_ANSWER_MESSAGE

    scored_sentences.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = scored_sentences[:max_sentences]
    selected.sort(key=lambda item: (item[1], item[2]))
    selected_sentences = [sentence for _, _, _, sentence in selected]

    selected_sentences = [
        sentence
        for sentence in selected_sentences
        if is_valid_answer_sentence(sentence)
    ]
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
