#!/usr/bin/env python3
"""
Локальный RAG-прототип по одному документу.

Что делает скрипт:
1) читает один документ;
2) режет его на фрагменты;
3) строит локальные векторные представления фрагментов и вопроса;
4) ищет релевантные фрагменты;
5) формирует extractive-ответ только по найденным фрагментам.

Внешние API, GigaChat/OpenAI и тяжелые RAG-фреймворки не используются.
Документ целиком в answerer не передается: ответ строится только по top-k найденным фрагментам.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ANSWER_INSTRUCTION = """Ты отвечаешь на вопрос только по найденным фрагментам документа.
Не используй внешние знания и не добавляй факты, которых нет в контексте.
Если в найденных фрагментах нет ответа, напиши: "В найденных фрагментах нет ответа."""

PROMPT_TEMPLATE = """{instruction}

Найденные фрагменты:
{context}

Вопрос:
{question}

Ответ:"""

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", flags=re.UNICODE)
CAP_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:[- ][А-ЯЁ][а-яё]{2,})?\b")

MONTHS_RE = (
    r"января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря"
)
DATE_RE = re.compile(rf"\b\d{{1,2}}\s+(?:{MONTHS_RE})\b", re.IGNORECASE)
TIME_RE = re.compile(
    r"\b\d{1,2}\s+час(?:а|ов)?(?:\s+\d{1,2}\s+минут(?:а|ы)?)?"
    r"(?:\s+\d{1,2}\s+секунд(?:а|ы)?)?(?:\s+(?:утра|вечера|дня|ночи))?\b",
    re.IGNORECASE,
)
NUMERIC_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

RUSSIAN_STOPWORDS = {
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть", "в", "вам",
    "вас", "весь", "во", "вот", "все", "всего", "всех", "вы", "где", "да", "даже",
    "для", "до", "его", "ее", "если", "есть", "еще", "же", "за", "здесь", "и", "из",
    "или", "им", "их", "к", "как", "какая", "какие", "какой", "каком", "кем", "ко",
    "когда", "кого", "который", "которая", "которые", "которым", "кто", "ли", "либо",
    "мне", "может", "мы", "на", "над", "надо", "наш", "не", "него", "нее", "нет", "ни",
    "них", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно", "от", "по", "под",
    "при", "про", "с", "со", "так", "также", "такой", "такая", "такое", "такие", "там",
    "те", "тем", "то", "того", "тоже", "той", "только", "том", "тот", "тут", "ты", "у",
    "уже", "чего", "чей", "чем", "через", "что", "чтобы", "это", "этого", "этой", "этом",
    "этот", "эту", "я", "лишь", "очень", "сам", "сама", "сами", "свое", "свою", "свои",
}

NAME_CUES = {
    "имя", "имена", "звать", "звали", "зовут", "назвать", "назвали", "называть", "называли",
    "кличка", "клички", "прозвали", "именовали",
}
DEFINITION_CUES = {
    "это", "являться", "являлся", "являлась", "являлись", "быть", "был", "была", "были",
    "председатель", "секретарь", "майор", "капитан", "командир", "профессор", "доктор",
    "инженер", "изобретатель", "ученый", "автор", "герой", "персонаж", "руководитель",
    "директор", "основатель", "член", "начальник", "путешественник", "исследователь",
}
EVENT_CUES = {
    "случиться", "произойти", "стать", "оказаться", "попасть", "погибнуть", "умереть",
    "издохнуть", "пропасть", "исчезнуть", "найти", "увидеть", "обнаружить",
}
WHY_CUES = {
    "потому", "поскольку", "так", "причина", "из-за", "вследствие", "обусловлено", "вызывать",
    "вызвало", "дело", "благодаря", "следствие", "поэтому", "оттого",
}
TIME_CUES = {
    "когда", "время", "дата", "час", "минута", "секунда", "утро", "день", "вечер", "ночь",
    "полночь", "произвести", "произведенный", "выпустить", "выпущенный", "пустить",
}
OBJECT_NAME_CUES = {
    "называться", "называлось", "назывался", "называли", "название", "имя", "тип", "вид",
    "именоваться", "будет", "была", "был",
}
WHO_CUES = {"кто", "человек", "люди", "участник", "персона", "находиться", "участвовать", "вместе"}

ROLE_WORDS = {
    "председатель", "секретарь", "майор", "капитан", "профессор", "доктор", "инженер",
    "ученый", "автор", "исследователь", "путешественник", "член", "командир", "директор",
}
BAD_NAME_WORDS = {
    "Глава", "Книга", "Часть", "После", "Затем", "Но", "Да", "Нет", "Вот", "Это", "Если",
    "В", "Во", "На", "И", "А", "Он", "Она", "Они", "Мы", "Вы", "Ты", "Ну", "Однако",
    "Луна", "Земля", "Солнце", "Америка", "Франция", "Балтимор", "Мексиканского", "Некий", "Предложение", "ГЛАВА",
}

SYNONYMS = {
    "произойти": {"случиться", "совершиться", "стать", "случай"},
    "случиться": {"произойти", "совершиться", "стать", "случай"},
    "причина": {"фактор", "основание", "объяснение", "источник"},
    "фактор": {"причина", "условие", "обстоятельство", "механизм"},
    "влияние": {"воздействие", "эффект", "роль"},
    "воздействие": {"влияние", "эффект", "роль"},
    "увеличить": {"повысить", "расширить", "рост", "увеличение"},
    "повысить": {"увеличить", "поднять", "рост", "повышение"},
    "уменьшить": {"снизить", "сократить", "уменьшение", "снижение"},
    "снизить": {"уменьшить", "сократить", "снижение"},
    "найти": {"обнаружить", "отыскать", "увидеть", "найденный"},
    "обнаружить": {"найти", "отыскать", "увидеть"},
    "название": {"наименование", "имя", "тип", "вид"},
    "имя": {"название", "наименование", "звать", "назвать"},
    "цель": {"задача", "назначение", "смысл"},
    "задача": {"цель", "назначение", "проблема"},
    "метод": {"способ", "подход", "прием"},
    "способ": {"метод", "подход", "прием"},
    "результат": {"итог", "следствие", "вывод"},
    "использовать": {"применять", "воспользоваться", "использование"},
    "начало": {"старт", "начаться", "первый"},
    "конец": {"завершение", "окончание", "последний"},
    "время": {"дата", "момент", "период", "час", "минута", "секунда"},
    "дата": {"время", "момент", "период", "год"},
    "запуск": {"пуск", "начало"},
    "отклониться": {"уклониться", "отклонение", "уклонение", "измениться"},
    "уклониться": {"отклониться", "отклонение", "уклонение", "измениться"},
}



@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    text: str
    token_start: int
    token_end: int


@dataclass(frozen=True)
class QueryInfo:
    raw: str
    intent: str
    lemmas: list[str]
    content_terms: list[str]
    focus_terms: list[str]
    expanded_tokens: list[str]
    raw_lower: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    bm25_score: float
    word_tfidf_score: float
    char_tfidf_score: float
    lsa_score: float
    coverage_score: float
    phrase_score: float


@dataclass(frozen=True)
class AnswerResult:
    text: str
    used_chunk_ids: set[int]


class MorphNormalizer:
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}
        self._morph = None
        try:
            import pymorphy3  # type: ignore
            self._morph = pymorphy3.MorphAnalyzer()
        except Exception:
            self._morph = None

    def normalize(self, token: str) -> str:
        token = token.lower().replace("ё", "е")
        if token in self._cache:
            return self._cache[token]
        if self._morph is not None:
            try:
                parsed = self._morph.parse(token)
                lemma = parsed[0].normal_form if parsed else token
            except Exception:
                lemma = simple_stem(token)
        else:
            lemma = simple_stem(token)
        lemma = lemma.lower().replace("ё", "е")
        self._cache[token] = lemma
        return lemma

    def normalize_name_word(self, word: str) -> str:
        clean = re.sub(r"[^A-Za-zА-Яа-яЁё-]", "", word)
        if not clean:
            return word
        if self._morph is not None:
            try:
                parsed = self._morph.parse(clean)
                # Для имён и фамилий нормальная форма обычно хорошо возвращает именительный падеж.
                for p in parsed[:5]:
                    tag = str(p.tag)
                    if "Name" in tag or "Surn" in tag or "Patr" in tag:
                        return p.normal_form.title().replace("-", "-")
                for p in parsed[:5]:
                    tag = str(p.tag)
                    if "VERB" not in tag and "INFN" not in tag:
                        return p.normal_form.title()
                return fallback_normalize_name_word(clean)
            except Exception:
                pass
        return fallback_normalize_name_word(clean)


def fallback_normalize_name_word(word: str) -> str:
    word = word[:1].upper() + word[1:].lower()
    low = word.lower()
    if len(word) > 5 and low.endswith("ой"):
        return word[:-2] + "а"
    if len(word) > 5 and (low.endswith("ом") or low.endswith("ем")):
        return word[:-2]
    if len(word) > 6 and low.endswith("ена"):
        return word[:-1]
    if len(word) > 4 and low.endswith("оля"):
        return word[:-1] + "ь"
    return word


MORPH = MorphNormalizer()


def simple_stem(token: str) -> str:
    endings = (
        "иями", "ями", "ами", "его", "ого", "ему", "ому", "ими", "ыми", "ими", "ую", "юю",
        "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой", "ам", "ям", "ах", "ях",
        "ом", "ем", "ою", "ею", "ов", "ев", "ей", "ия", "ья", "а", "я", "ы", "и", "у",
        "ю", "е", "о", "ь",
    )
    for ending in endings:
        if len(token) > len(ending) + 3 and token.endswith(ending):
            return token[: -len(ending)]
    return token


def normalize_set(words: Iterable[str]) -> set[str]:
    return {MORPH.normalize(w) for w in words}


def read_document(path: Path, encoding: Optional[str] = None) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not path.is_file():
        raise ValueError(f"Путь не является файлом: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".log"}:
        return read_text_file(path, encoding)
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return read_text_file(path, encoding)


def read_text_file(path: Path, encoding: Optional[str]) -> str:
    encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "cp1251", "windows-1251"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        if enc is None:
            continue
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Не удалось прочитать текстовый файл: {last_error}")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Для PDF установите зависимости из requirements.txt") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Для DOCX установите зависимости из requirements.txt") from exc
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?…:;])", r"\1", text)
    return text.strip()


def tokenize_raw(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def tokenize_search(text: str) -> list[str]:
    lemmas: list[str] = []
    for token in tokenize_raw(text):
        if len(token) <= 1 or token in RUSSIAN_STOPWORDS:
            continue
        lemma = MORPH.normalize(token)
        if len(lemma) > 1 and lemma not in RUSSIAN_STOPWORDS:
            lemmas.append(lemma)
    return lemmas


def split_sentences(text: str) -> list[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    # Не идеально, но для художественного и учебного текста достаточно: делим по знакам конца фразы.
    parts = re.split(r"(?<=[.!?…])\s+(?=[—«\"'А-ЯA-Z0-9])", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        words = text.split()
        return [" ".join(words[i : i + 35]) for i in range(0, len(words), 35)]
    return parts


def split_long_sentence(sentence: str, max_words: int) -> list[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def make_chunks(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk-size должен быть положительным")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть >= 0 и меньше chunk-size")

    sentences: list[str] = []
    for sent in split_sentences(text):
        sentences.extend(split_long_sentence(sent, max_words=max(chunk_size, 80)))
    if not sentences:
        return []

    sent_lens = [max(1, len(tokenize_search(sent))) for sent in sentences]
    raw: list[tuple[int, int, list[str]]] = []

    start = 0
    token_start = 0
    while start < len(sentences):
        current: list[str] = []
        current_len = 0
        i = start
        while i < len(sentences):
            sent_len = sent_lens[i]
            if current and current_len + sent_len > chunk_size and current_len >= chunk_size * 0.55:
                break
            current.append(sentences[i])
            current_len += sent_len
            i += 1

        token_end = token_start + current_len
        raw.append((token_start, token_end, current))
        if i >= len(sentences):
            break

        back = 0
        next_start = i
        while next_start > start and back < overlap:
            next_start -= 1
            back += sent_lens[next_start]
        start = max(next_start, start + 1)
        token_start = max(0, token_end - back)

    if len(raw) >= 2:
        last_size = raw[-1][1] - raw[-1][0]
        if last_size <= chunk_size * 0.40:
            prev = raw[-2]
            last = raw[-1]
            raw[-2] = (prev[0], last[1], prev[2] + last[2])
            raw.pop()

    return [Chunk(i, normalize_spaces(" ".join(sents)), s, e) for i, (s, e, sents) in enumerate(raw, start=1)]


def unique_ordered(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def detect_intent(question: str) -> str:
    q = question.lower().replace("ё", "е")
    if re.search(r"\b(как\s+звали|как\s+зовут|имена?|кличк[аиуой]?)\b", q):
        return "name"
    if re.search(r"\b(кто|что)\s+(такой|такая|такое|такие)\b", q):
        return "definition"
    if re.search(r"\b(когда|во\s+сколько|в\s+какое\s+время|время)\b", q):
        return "time"
    if re.search(r"\b(почему|зачем|из-за\s+чего|по\s+какой\s+причине)\b", q):
        return "why"
    if re.search(r"\b(как\s+называл\w*|как\s+называет\w*|название|как\s+назвали)\b", q):
        return "object_name"
    if re.search(r"\bкто\b", q):
        return "who"
    if re.search(r"\b(что\s+случилось|что\s+произошло|что\s+стало|что\s+произошло\s+с)\b", q):
        return "event"
    return "generic"


def analyze_query(question: str) -> QueryInfo:
    raw_lower = question.lower().replace("ё", "е")
    intent = detect_intent(question)
    lemmas = unique_ordered(tokenize_search(question))

    cue_sets = {
        "name": NAME_CUES,
        "definition": DEFINITION_CUES,
        "time": TIME_CUES,
        "why": WHY_CUES,
        "object_name": OBJECT_NAME_CUES,
        "who": WHO_CUES,
        "event": EVENT_CUES,
    }
    cue_lemmas = normalize_set(cue_sets.get(intent, set()))
    content_terms = [lemma for lemma in lemmas if lemma not in cue_lemmas]
    focus_terms = extract_focus_terms(question, intent, content_terms)

    expanded: list[str] = []
    expanded.extend(lemmas)
    expanded.extend(content_terms * 2)
    expanded.extend(focus_terms * 4)
    expanded.extend(normalize_set(cue_sets.get(intent, set())))

    for term in list(content_terms) + list(focus_terms):
        expanded.extend(expand_synonyms(term))
    
    if intent == "time":
        expanded.extend(normalize_set({"выпущенный", "выпустить", "пустить", "выстрел", "дата", "час", "минута", "секунда"}))
    if intent == "object_name":
        expanded.extend(normalize_set({"название", "наименование", "тип", "вид", "называться", "будет"}))
    if intent == "who":
        expanded.extend(normalize_set({"пассажир", "путешественник", "исследователь", "внутри", "вместе", "находиться"}))
    if intent == "why":
        expanded.extend(normalize_set({"потому", "причина", "дело", "вследствие", "из-за", "обусловлено"}))

    expanded = unique_ordered(expanded)
    return QueryInfo(question, intent, lemmas, content_terms, focus_terms, expanded, raw_lower)


def expand_synonyms(term: str) -> list[str]:
    out: list[str] = []
    for key, values in SYNONYMS.items():
        key_norm = MORPH.normalize(key)
        if fuzzy_match(term, key_norm):
            out.extend(MORPH.normalize(v) for v in values)
    return unique_ordered(out)


def extract_focus_terms(question: str, intent: str, content_terms: list[str]) -> list[str]:
    q = question.lower().replace("ё", "е")
    if intent == "definition":
        match = re.search(r"\b(?:кто|что)\s+(?:такой|такая|такое|такие)\s+(.+?)[?!.]*$", q)
        if match:
            terms = tokenize_search(match.group(1))
            if terms:
                return unique_ordered(terms)
    if intent in {"event", "name"}:
        # Вопросы вида "что случилось с X", "как звали X".
        match = re.search(r"\b(?:с|со|о|об|про)\s+([a-zа-я0-9ё-]+)", q)
        if match:
            terms = tokenize_search(match.group(1))
            if terms:
                return unique_ordered(terms)
    if intent == "who":
        # Берем объект после предлога в вопросах вида "кто ... в/на/с чем".
        matches = re.findall(r"\b(?:в|во|внутри|на|из|изнутри|с|со)\s+([a-zа-я0-9ё-]+)", q)
        if matches:
            terms: list[str] = []
            for m in matches:
                terms.extend(tokenize_search(m))
            if terms:
                return unique_ordered(terms)
    if intent == "object_name":
        # Для вопросов о названии полезнее держать объект, а не само слово "называлось".
        terms = [t for t in content_terms if t not in normalize_set(OBJECT_NAME_CUES)]
        if terms:
            return unique_ordered(terms[:3])
    return unique_ordered(content_terms)


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs_tokens = docs_tokens
        self.k1 = k1
        self.b = b
        self.lengths = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avg_len = float(np.mean(self.lengths)) if len(self.lengths) else 0.0
        self.freqs = [Counter(tokens) for tokens in docs_tokens]
        df: dict[str, int] = defaultdict(int)
        for tokens in docs_tokens:
            for token in set(tokens):
                df[token] += 1
        n = len(docs_tokens)
        self.idf = {token: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()}

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.docs_tokens), dtype=np.float32)
        if not query_tokens or self.avg_len <= 0:
            return scores
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for i, freqs in enumerate(self.freqs):
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * float(self.lengths[i]) / self.avg_len)
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


class LocalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_tokens = [tokenize_search(c.text) for c in chunks]
        self.lemma_texts = [" ".join(toks) for toks in self.chunk_tokens]
        self.raw_texts = [c.text.lower().replace("ё", "е") for c in chunks]
        self.bm25 = BM25Index(self.chunk_tokens)

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"
        )
        self.char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)

        self.word_matrix_raw = self.word_vectorizer.fit_transform(self.lemma_texts)
        self.char_matrix_raw = self.char_vectorizer.fit_transform(self.raw_texts)
        self.word_matrix = normalize(self.word_matrix_raw, norm="l2", copy=False)
        self.char_matrix = normalize(self.char_matrix_raw, norm="l2", copy=False)

        self.svd: Optional[TruncatedSVD] = None
        self.lsa_matrix: Optional[np.ndarray] = None
        self._init_lsa()

    def _init_lsa(self) -> None:
        n_docs, n_features = self.word_matrix_raw.shape
        n_components = min(80, n_docs - 1, n_features - 1)
        if n_components < 2:
            return
        try:
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            matrix = self.svd.fit_transform(self.word_matrix_raw)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.lsa_matrix = matrix / norms
        except Exception:
            self.svd = None
            self.lsa_matrix = None

    def search(self, query: QueryInfo, *, top_k: int, mode: str) -> list[SearchResult]:
        q_tokens = query.expanded_tokens
        bm25 = self.bm25.score(q_tokens)

        q_word = self.word_vectorizer.transform([" ".join(q_tokens)])
        q_word = normalize(q_word, norm="l2", copy=False)
        word_scores = (self.word_matrix @ q_word.T).toarray().ravel().astype(np.float32)

        q_char = self.char_vectorizer.transform([query.raw_lower])
        q_char = normalize(q_char, norm="l2", copy=False)
        char_scores = (self.char_matrix @ q_char.T).toarray().ravel().astype(np.float32)

        lsa_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self.svd is not None and self.lsa_matrix is not None:
            try:
                q_lsa = self.svd.transform(q_word)
                q_norm = np.linalg.norm(q_lsa)
                if q_norm > 0:
                    q_lsa = q_lsa / q_norm
                    lsa_scores = (self.lsa_matrix @ q_lsa.T).ravel().astype(np.float32)
                    lsa_scores = np.maximum(lsa_scores, 0)
            except Exception:
                pass

        coverage = np.array([coverage_score(tokens, query) for tokens in self.chunk_tokens], dtype=np.float32)
        phrase = np.array([phrase_score(chunk.text, tokens, query) for chunk, tokens in zip(self.chunks, self.chunk_tokens)], dtype=np.float32)
        gate = np.array([intent_gate(chunk.text, tokens, query) for chunk, tokens in zip(self.chunks, self.chunk_tokens)], dtype=np.float32)

        if mode == "bm25":
            combined = normalize_scores(bm25) * gate
        else:
            combined = (
                0.32 * normalize_scores(bm25)
                + 0.25 * normalize_scores(word_scores)
                + 0.16 * normalize_scores(char_scores)
                + 0.12 * normalize_scores(lsa_scores)
                + 0.10 * normalize_scores(coverage)
                + 0.05 * normalize_scores(phrase)
            ) * gate

        top_k = min(max(1, top_k), len(self.chunks))
        indices = np.argsort(combined)[::-1][:top_k]
        return [
            SearchResult(
                chunk=self.chunks[int(i)],
                score=float(combined[int(i)]),
                bm25_score=float(bm25[int(i)]),
                word_tfidf_score=float(word_scores[int(i)]),
                char_tfidf_score=float(char_scores[int(i)]),
                lsa_score=float(lsa_scores[int(i)]),
                coverage_score=float(coverage[int(i)]),
                phrase_score=float(phrase[int(i)]),
            )
            for i in indices
        ]


def normalize_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    max_val = float(np.max(arr))
    if max_val <= 1e-12:
        return np.zeros_like(arr)
    return arr / max_val


def compact_repeats(token: str) -> str:
    if not token:
        return token
    out = [token[0]]
    for ch in token[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def fuzzy_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)):
        return True
    if len(a) >= 5 and len(b) >= 5:
        return compact_repeats(a) == compact_repeats(b) or simple_stem(a) == simple_stem(b)
    return False


def fuzzy_contains(tokens: Sequence[str], term: str) -> bool:
    token_set = tokens if isinstance(tokens, set) else set(tokens)
    if term in token_set:
        return True
    if len(term) < 4:
        return False
    compact_term = compact_repeats(term)
    for tok in token_set:
        if len(tok) < 4:
            continue
        if tok.startswith(term) or term.startswith(tok):
            return True
        if len(term) >= 5 and compact_repeats(tok) == compact_term:
            return True
    return False


def count_hits(tokens: Sequence[str], terms: Sequence[str]) -> int:
    token_set = tokens if isinstance(tokens, set) else set(tokens)
    return sum(1 for term in terms if fuzzy_contains(token_set, term))


def coverage_score(chunk_tokens: list[str], query: QueryInfo) -> float:
    if not query.content_terms:
        return 0.0
    content_hits = count_hits(chunk_tokens, query.content_terms)
    focus_hits = count_hits(chunk_tokens, query.focus_terms)
    return (content_hits / max(1, len(query.content_terms))) + 0.45 * (focus_hits / max(1, len(query.focus_terms)))


def phrase_score(text: str, tokens: list[str], query: QueryInfo) -> float:
    sents = split_sentences(text)
    best = 0.0
    for sent in sents:
        stoks = tokenize_search(sent)
        if not stoks:
            continue
        focus = count_hits(stoks, query.focus_terms) / max(1, len(query.focus_terms))
        content = count_hits(stoks, query.content_terms) / max(1, len(query.content_terms))
        cue = 0.0
        if query.intent == "name":
            cue = 1.0 if count_hits(stoks, list(normalize_set(NAME_CUES))) else 0.0
        elif query.intent == "definition":
            cue = 1.0 if count_hits(stoks, list(normalize_set(DEFINITION_CUES))) else 0.0
        elif query.intent == "time":
            cue = 1.0 if has_temporal_expression(sent) else 0.0
        elif query.intent == "why":
            cue = 1.0 if has_causal_marker(sent) else 0.0
        elif query.intent == "object_name":
            cue = 1.0 if count_hits(stoks, list(normalize_set(OBJECT_NAME_CUES))) else 0.0
        elif query.intent == "who":
            cue = 1.0 if contains_person_like_name(sent) else 0.0
            if has_inside_phrase(sent, query.focus_terms):
                cue += 0.7
        elif query.intent == "event":
            cue = 1.0 if count_hits(stoks, list(normalize_set(EVENT_CUES))) else 0.0
        best = max(best, 0.45 * focus + 0.35 * content + 0.20 * cue)
    return best


def intent_gate(text: str, tokens: list[str], query: QueryInfo) -> float:
    content_hits = count_hits(tokens, query.content_terms)
    focus_hits = count_hits(tokens, query.focus_terms)

    # Универсальное правило: если в вопросе несколько содержательных слов, одинокое совпадение не должно побеждать.
    if len(query.content_terms) >= 2:
        if content_hits == 0:
            base = 0.18
        elif content_hits == 1:
            base = 0.48
        else:
            base = 1.0
    else:
        base = 1.0 if content_hits > 0 or not query.content_terms else 0.55

    if query.focus_terms and focus_hits == 0:
        base *= 0.38

    if query.intent == "name":
        has_name_cue = count_hits(tokens, list(normalize_set(NAME_CUES))) > 0
        return base * (1.25 if has_name_cue and focus_hits else 0.45 if has_name_cue else 0.65)
    if query.intent == "time":
        return base * (1.20 if has_temporal_expression(text) else 0.35)
    if query.intent == "why":
        return base * (1.15 if has_causal_marker(text) else 0.65)
    if query.intent == "object_name":
        has_object_cue = count_hits(tokens, list(normalize_set(OBJECT_NAME_CUES))) > 0
        return base * (1.20 if has_object_cue else 0.55)
    if query.intent == "who":
        bonus = 1.0
        if contains_person_like_name(text):
            bonus *= 1.12
        if query.focus_terms and has_inside_phrase(text, query.focus_terms):
            bonus *= 1.35
        return base * bonus
    return base


def has_temporal_expression(text: str) -> bool:
    return bool(DATE_RE.search(text) or TIME_RE.search(text) or NUMERIC_TIME_RE.search(text))


def has_causal_marker(text: str) -> bool:
    low = text.lower().replace("ё", "е")
    markers = ["потому что", "так как", "дело в том", "из-за", "вследствие", "поэтому", "оттого", "причин", "вызвал", "вызвало", "обусловлено", "благодаря"]
    return any(m in low for m in markers)


def has_inside_phrase(text: str, focus_terms: Sequence[str]) -> bool:
    if not focus_terms:
        return False
    low_tokens = tokenize_raw(text)
    lemmas = [MORPH.normalize(t) for t in low_tokens]
    preps = {"в", "во", "внутри", "на", "из"}
    focus_expanded = set(focus_terms)
    for term in focus_terms:
        focus_expanded.update(expand_synonyms(term))
    for i, lemma in enumerate(lemmas):
        if lemma in focus_expanded:
            left = set(low_tokens[max(0, i - 3) : i]) | set(lemmas[max(0, i - 3) : i])
            if left & preps:
                return True
    return False


def contains_person_like_name(text: str) -> bool:
    for name in CAP_RE.findall(text):
        if clean_name(name) and clean_name(name) not in BAD_NAME_WORDS:
            return True
    return False


def expand_with_neighbors(results: list[SearchResult], chunks: list[Chunk], *, radius: int, max_results: int) -> list[SearchResult]:
    if radius <= 0:
        return results[:max_results]
    by_id: dict[int, SearchResult] = {r.chunk.chunk_id: r for r in results}
    for res in results:
        base_idx = res.chunk.chunk_id - 1
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            idx = base_idx + offset
            if 0 <= idx < len(chunks):
                # Сосед нужен как контекст, но не должен выглядеть как равноценный hit.
                decayed = max(0.0, res.score * (0.40 ** abs(offset)))
                chunk = chunks[idx]
                current = by_id.get(chunk.chunk_id)
                if current is None or decayed > current.score:
                    by_id[chunk.chunk_id] = SearchResult(chunk, decayed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return sorted(by_id.values(), key=lambda r: r.score, reverse=True)[:max_results]


class ExtractiveAnswerer:
    def __init__(self, *, min_score: float = 0.04) -> None:
        self.min_score = min_score

    def answer(self, query: QueryInfo, results: list[SearchResult], *, max_sentences: int) -> AnswerResult:
        if not results or results[0].score < self.min_score:
            return AnswerResult("В найденных фрагментах нет ответа.", set())

        for handler in (
            self.answer_name,
            self.answer_who,
            self.answer_time,
            self.answer_object_name,
            self.answer_definition,
            self.answer_why,
        ):
            out = handler(query, results)
            if out is not None:
                return out

        return self.answer_generic(query, results, max_sentences=max_sentences)

    def answer_name(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "name":
            return None
        for res in results:
            for window in sentence_windows(res.chunk.text, radius=1):
                wtoks = tokenize_search(window)
                if query.focus_terms and not count_hits(wtoks, query.focus_terms):
                    continue
                names = extract_names_after_focus(window, query.focus_terms)
                if names:
                    return AnswerResult("Имена: " + join_names(names) + ".", {res.chunk.chunk_id})
        return None

    def answer_who(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "who":
            return None
        candidates: list[tuple[float, list[str], int, str]] = []
        for res in results:
            for window in sentence_windows(res.chunk.text, radius=1):
                wtoks = tokenize_search(window)
                if query.focus_terms and not count_hits(wtoks, query.focus_terms):
                    continue
                names = extract_person_names(window)
                if not names:
                    continue
                score = res.score
                if has_inside_phrase(window, query.focus_terms):
                    score += 0.35
                score += 0.05 * min(len(names), 4)
                candidates.append((score, names, res.chunk.chunk_id, window))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        names = candidates[0][1][:6]
        return AnswerResult("В найденном фрагменте указаны: " + join_names(names) + ".", {candidates[0][2]})

    def answer_time(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "time":
            return None
        best: Optional[tuple[float, str, int]] = None
        for res in results:
            for sent in split_sentences(res.chunk.text):
                if not has_temporal_expression(sent):
                    continue
                stoks = tokenize_search(sent)
                focus_hits = count_hits(stoks, query.focus_terms + query.content_terms)
                low_sent = sent.lower().replace("ё", "е")
                has_date = bool(DATE_RE.search(sent))
                has_time = bool(TIME_RE.search(sent) or NUMERIC_TIME_RE.search(sent))
                full_datetime = has_date and has_time
                relation = 0.0
                for m in list(DATE_RE.finditer(sent)) + list(TIME_RE.finditer(sent)) + list(NUMERIC_TIME_RE.finditer(sent)):
                    before = low_sent[max(0, m.start() - 80):m.start()]
                    if any(root in before for root in ("выпущ", "запущ", "пущ", "начат", "состоя", "произвед")):
                        relation += 0.55
                    if any(root in before for root in ("усмотр", "наблюд", "сообщ", "телеграм")):
                        relation -= 0.65
                score = 0.20 * res.score + 0.22 * focus_hits + (1.10 if full_datetime else 0.55 if has_date else 0.15 if has_time else 0.0) + relation
                if best is None or score > best[0]:
                    best = (score, sent, res.chunk.chunk_id)
        if best is None:
            return None
        return AnswerResult(clean_answer_sentence(best[1]), {best[2]})

    def answer_object_name(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "object_name":
            return None
        for res in results:
            for sent in split_sentences(res.chunk.text):
                stoks = tokenize_search(sent)
                if query.focus_terms and not count_hits(stoks, query.focus_terms):
                    continue
                extracted = extract_object_name_phrase(sent, query.focus_terms)
                if extracted:
                    return AnswerResult(extracted, {res.chunk.chunk_id})
        return None

    def answer_definition(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "definition":
            return None
        for res in results:
            for window in sentence_windows(res.chunk.text, radius=1):
                wtoks = tokenize_search(window)
                if query.focus_terms and not count_hits(wtoks, query.focus_terms):
                    continue
                if count_hits(wtoks, list(normalize_set(DEFINITION_CUES))) or contains_person_like_name(window):
                    return AnswerResult(clean_answer_sentence(window), {res.chunk.chunk_id})
        return None

    def answer_why(self, query: QueryInfo, results: list[SearchResult]) -> Optional[AnswerResult]:
        if query.intent != "why":
            return None
        best: Optional[tuple[float, str, int]] = None
        for res in results:
            for window in sentence_windows(res.chunk.text, radius=1):
                if not has_causal_marker(window):
                    continue
                wtoks = tokenize_search(window)
                score = res.score + 0.15 * count_hits(wtoks, query.focus_terms + query.content_terms)
                if best is None or score > best[0]:
                    best = (score, window, res.chunk.chunk_id)
        if best is None:
            return None
        return AnswerResult(clean_answer_sentence(best[1]), {best[2]})

    def answer_generic(self, query: QueryInfo, results: list[SearchResult], *, max_sentences: int) -> AnswerResult:
        candidates = collect_candidates(results)
        if not candidates:
            return AnswerResult("В найденных фрагментах нет ответа.", set())

        sent_texts = [c[0] for c in candidates]
        sent_scores = sentence_scores(sent_texts, query)
        chunk_scores = normalize_scores(np.array([c[2] for c in candidates], dtype=np.float32))
        final = 0.72 * sent_scores + 0.28 * chunk_scores

        selected: list[tuple[int, str, int]] = []
        used_chunks: set[int] = set()
        used_texts: list[str] = []
        for idx in np.argsort(final)[::-1]:
            idx = int(idx)
            if final[idx] <= 0.02:
                continue
            sent, chunk_id, _ = candidates[idx]
            if is_bad_answer_sentence(sent):
                continue
            if query.focus_terms and query.intent in {"event", "generic"}:
                if not count_hits(tokenize_search(sent), query.focus_terms):
                    continue
            if is_near_duplicate(sent, used_texts):
                continue
            selected.append((idx, clean_answer_sentence(sent), chunk_id))
            used_chunks.add(chunk_id)
            used_texts.append(sent)
            if len(selected) >= max_sentences:
                break

        if not selected:
            return AnswerResult("В найденных фрагментах нет ответа.", set())
        selected.sort(key=lambda item: item[0])
        return AnswerResult(normalize_spaces(" ".join(s for _, s, _ in selected)), used_chunks)


def sentence_windows(text: str, *, radius: int = 1) -> list[str]:
    sents = split_sentences(text)
    windows: list[str] = []
    for i in range(len(sents)):
        left = max(0, i - radius)
        right = min(len(sents), i + radius + 1)
        window = normalize_spaces(" ".join(sents[left:right]))
        if len(window) > 900:
            window = normalize_spaces(sents[i])
        windows.append(window)
    return windows


def collect_candidates(results: list[SearchResult]) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    seen: set[str] = set()
    for res in results:
        for sent in split_sentences(res.chunk.text):
            clean = clean_answer_sentence(sent)
            if len(clean) < 20 or is_bad_answer_sentence(clean):
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((clean, res.chunk.chunk_id, res.score))
    return out


def sentence_scores(sentences: list[str], query: QueryInfo) -> np.ndarray:
    if not sentences:
        return np.array([], dtype=np.float32)
    sent_tokens = [tokenize_search(s) for s in sentences]
    lemma_texts = [" ".join(t) for t in sent_tokens]
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(lemma_texts)
    q_vec = vectorizer.transform([" ".join(query.expanded_tokens)])
    tfidf = (matrix @ q_vec.T).toarray().ravel().astype(np.float32)
    coverage = np.array([coverage_score(t, query) for t in sent_tokens], dtype=np.float32)
    phrase = np.array([phrase_score(s, t, query) for s, t in zip(sentences, sent_tokens)], dtype=np.float32)
    question_penalty = np.array([0.55 if s.strip().endswith("?") else 1.0 for s in sentences], dtype=np.float32)
    return (0.50 * normalize_scores(tfidf) + 0.30 * normalize_scores(coverage) + 0.20 * normalize_scores(phrase)) * question_penalty


def extract_names_after_focus(text: str, focus_terms: Sequence[str]) -> list[str]:
    sents = split_sentences(text)
    for sent in sents:
        stoks = tokenize_search(sent)
        if focus_terms and not count_hits(stoks, focus_terms):
            continue
        low = sent.lower().replace("ё", "е")
        # Берем часть после слова-фокуса, чтобы не хватать имена из предыдущей части предложения.
        start = 0
        if focus_terms:
            for match in WORD_RE.finditer(low):
                lemma = MORPH.normalize(match.group(0))
                if any(fuzzy_match(lemma, f) for f in focus_terms):
                    start = match.end()
                    break
        tail = sent[start : start + 260]
        dash_pos = min([p for p in [tail.find("—"), tail.find(":"), tail.find("-")] if p >= 0] or [-1])
        if dash_pos >= 0:
            tail = tail[dash_pos + 1 :]
        names = extract_person_names(tail, allow_common_nouns=True)
        if names:
            return names[:4]
    return []


def extract_person_names(text: str, *, allow_common_nouns: bool = False) -> list[str]:
    names: list[str] = []
    for raw in CAP_RE.findall(text):
        cleaned = clean_name(raw)
        if not cleaned or cleaned in BAD_NAME_WORDS:
            continue
        if is_probably_not_name(cleaned, allow_common_nouns=allow_common_nouns):
            continue
        norm = normalize_name_phrase(cleaned)
        if norm and norm not in names and norm not in BAD_NAME_WORDS:
            names.append(norm)
        if len(names) >= 8:
            break
    return names


def clean_name(name: str) -> str:
    name = normalize_spaces(name.strip("—-.,:;!?…()[]{}«»\"'"))
    if not name or name in BAD_NAME_WORDS:
        return ""
    return name


def is_probably_not_name(name: str, *, allow_common_nouns: bool) -> bool:
    words = name.split()
    if not words:
        return True
    if words[0] in BAD_NAME_WORDS:
        return True
    if allow_common_nouns:
        return False
    # В обычном режиме отсекаем заглавные слова, которые часто являются началом предложения, а не именем.
    lemma_words = [MORPH.normalize(w) for w in words]
    role_lemmas = normalize_set(ROLE_WORDS | {"путешественник", "исследователь", "герой", "спасатель", "человек"})
    if any(lw in role_lemmas for lw in lemma_words):
        return True
    if len(words) == 1 and words[0] in {"Это", "Если", "После", "Затем", "Впрочем", "Теперь", "Следовательно"}:
        return True
    return False


def normalize_name_phrase(name: str) -> str:
    parts = []
    for word in name.split():
        if word.lower() in {"и", "с", "со"}:
            continue
        parts.append(MORPH.normalize_name_word(word))
    return " ".join(parts)


def join_names(names: Sequence[str]) -> str:
    names = [n for n in unique_ordered(names) if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " и " + names[-1]


def extract_object_name_phrase(sentence: str, focus_terms: Sequence[str]) -> Optional[str]:
    sent = normalize_spaces(sentence)
    low = sent.lower().replace("ё", "е")
    # X будет Y / X была Y / X называлась Y.
    patterns = [
        r"(?P<object>\w*оруди\w*|\w*пушк\w*|\w*оружи\w*)\s+(?:буд\w+|был\w*)\s+(?P<name>[^,.!?;:]{3,90})",
        r"(?P<object>\w+)\s+называл\w+\s+(?P<name>[^,.!?;:]{3,90})",
        r"(?:тип(?:а)?|вид(?:а)?)\s+(?P<name>[^,.!?;:]{3,90})",
    ]
    for pattern in patterns:
        match = re.search(pattern, low, flags=re.IGNORECASE)
        if match:
            name = match.group("name")
            name = normalize_spaces(name.strip(" —-"))
            if len(name.split()) > 12:
                name = " ".join(name.split()[:12])
            return "В найденном фрагменте указано: " + name + "."
    # Если шаблон не сработал, но предложение явно содержит объект и название, возвращаем предложение.
    if focus_terms and count_hits(tokenize_search(sent), focus_terms):
        if re.search(r"пушк|оруди|оружи|называл|тип", low):
            return clean_answer_sentence(sent)
    return None


def clean_answer_sentence(text: str) -> str:
    text = normalize_spaces(text)
    # Убираем заголовки, которые часто прилипают к ответу при чанкинге книг.
    text = re.sub(r"\bГЛАВА\s+[А-ЯЁA-Z\s\-]+\.?\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def is_bad_answer_sentence(text: str) -> bool:
    clean = normalize_spaces(text)
    if not clean:
        return True
    if re.fullmatch(r"[—\-\sА-ЯЁA-Z0-9.]+", clean) and "ГЛАВА" in clean.upper():
        return True
    if clean.count("?") >= 2 and len(clean) < 180:
        return True
    return False


def is_near_duplicate(text: str, selected: list[str], threshold: float = 0.80) -> bool:
    tokens = set(tokenize_search(text))
    if not tokens:
        return False
    for other in selected:
        other_tokens = set(tokenize_search(other))
        if not other_tokens:
            continue
        jaccard = len(tokens & other_tokens) / len(tokens | other_tokens)
        if jaccard >= threshold:
            return True
    return False


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context = []
    for i, res in enumerate(results, start=1):
        context.append(f"Фрагмент {i} (chunk_id={res.chunk.chunk_id}, score={res.score:.3f}):\n{res.chunk.text}")
    return PROMPT_TEMPLATE.format(instruction=ANSWER_INSTRUCTION, context="\n\n".join(context), question=question)


def preview(text: str, limit: int = 560) -> str:
    text = normalize_spaces(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def print_results(
    *,
    question: str,
    answer: AnswerResult,
    results: list[SearchResult],
    top_k: int,
    show_prompt: bool,
    debug_scores: bool,
) -> None:
    print("Вопрос:")
    print(question)
    print()
    print("Ответ:")
    print(answer.text)
    print()
    print("Источники:")

    # Использованный answerer chunk показываем первым, затем остальные по score.
    shown = sorted(results, key=lambda r: (r.chunk.chunk_id not in answer.used_chunk_ids, -r.score))[:top_k]
    if not shown:
        print("Нет найденных фрагментов.")
    for i, res in enumerate(shown, start=1):
        if debug_scores:
            score_text = (
                f"score={res.score:.3f}, bm25={res.bm25_score:.3f}, word_tfidf={res.word_tfidf_score:.3f}, "
                f"char_tfidf={res.char_tfidf_score:.3f}, lsa={res.lsa_score:.3f}, "
                f"coverage={res.coverage_score:.3f}, phrase={res.phrase_score:.3f}"
            )
        else:
            score_text = f"score={res.score:.3f}"
        print(f"{i}) chunk {res.chunk.chunk_id}, {score_text}, tokens={res.chunk.token_start}-{res.chunk.token_end}")
        print(preview(res.chunk.text))
        print()

    if show_prompt:
        print("Промпт/инструкция, по которой должен формироваться ответ:")
        print(build_prompt(question, shown))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Локальный RAG по одному документу без внешних API и скачивания LLM.")
    parser.add_argument("--document", required=True, type=Path, help="Путь к документу: txt/md/pdf/docx")
    parser.add_argument("--question", required=True, help="Вопрос по документу")
    parser.add_argument("--chunk-size", type=int, default=220, help="Размер чанка в словах. По умолчанию: 220")
    parser.add_argument("--overlap", type=int, default=60, help="Перекрытие чанков в словах. По умолчанию: 60")
    parser.add_argument("--top-k", type=int, default=3, help="Сколько источников вывести. По умолчанию: 3")
    parser.add_argument("--candidate-k", type=int, default=16, help="Сколько кандидатов брать для answerer. По умолчанию: 16")
    parser.add_argument("--neighbor-radius", type=int, default=0, help="Сколько соседних чанков добавить вокруг найденных. По умолчанию: 0")
    parser.add_argument("--max-answer-sentences", type=int, default=2, help="Максимум предложений в fallback-ответе. По умолчанию: 2")
    parser.add_argument(
        "--retriever",
        choices=["local", "lexical", "bm25"],
        default="local",
        help="local/lexical = BM25 + TF-IDF + LSA + правила. bm25 = только BM25. По умолчанию: local",
    )
    parser.add_argument("--min-score", type=float, default=0.04, help="Минимальный score для ответа. По умолчанию: 0.04")
    parser.add_argument("--encoding", default=None, help="Кодировка txt. Если не указана, пробуются utf-8/cp1251")
    parser.add_argument("--show-prompt", action="store_true", help="Показать инструкцию отвечать только по найденным фрагментам")
    parser.add_argument("--debug-scores", action="store_true", help="Показать компоненты score")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        raw_text = read_document(args.document, args.encoding)
        text = normalize_text(raw_text)
        chunks = make_chunks(text, chunk_size=args.chunk_size, overlap=args.overlap)
        if not chunks:
            print("Документ пустой или не содержит текста.", file=sys.stderr)
            return 1

        query = analyze_query(args.question)
        retriever = LocalRetriever(chunks)
        mode = "local" if args.retriever == "lexical" else args.retriever
        initial = retriever.search(query, top_k=max(args.top_k, args.candidate_k), mode=mode)
        results = expand_with_neighbors(initial, chunks, radius=args.neighbor_radius, max_results=max(args.top_k, args.candidate_k))

        answerer = ExtractiveAnswerer(min_score=args.min_score)
        answer = answerer.answer(query, results, max_sentences=args.max_answer_sentences)

        print_results(
            question=args.question,
            answer=answer,
            results=results,
            top_k=args.top_k,
            show_prompt=args.show_prompt,
            debug_scores=args.debug_scores,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
