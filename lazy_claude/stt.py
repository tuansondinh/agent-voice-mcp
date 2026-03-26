"""stt.py — Speech-to-text via whisper.cpp (pywhispercpp).

All output (logging, progress) goes to stderr.  stdout is reserved for the
MCP protocol channel and must stay clean.

Public API
----------
load_model(model_name: str = "base.en") -> Model
    Download (first run) and load a whisper.cpp model.  Returns a
    pywhispercpp.model.Model instance.

transcribe(audio: np.ndarray, model: Model | None = None) -> TranscribeResult
    Transcribe a 16 kHz mono float32 numpy array.  Returns a TranscribeResult
    with the recognised text (stripped of artefacts) and the no_speech_prob
    averaged across segments.  For empty/silent input, text is "" and
    no_speech_prob is 1.0.

TranscribeResult
    Dataclass: text (str) + no_speech_prob (float).

_strip_artifacts(text: str) -> str
    Remove common whisper hallucination tokens and extraneous whitespace.
    Exposed for testing.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pywhispercpp.model import Model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "small.en"
_DEFAULT_MODEL_MULTILINGUAL = "small"  # multilingual Whisper model for non-English STT


# ---------------------------------------------------------------------------
# TranscribeResult — structured return type for transcribe()
# ---------------------------------------------------------------------------


@dataclass
class TranscribeResult:
    """Result returned by :func:`transcribe`.

    Attributes
    ----------
    text:
        Recognised text, stripped of artefacts.  Empty string for silent or
        empty input.
    no_speech_prob:
        Whisper's no-speech probability averaged across all segments.
        Ranges from 0.0 (definitely speech) to 1.0 (definitely silence/noise).
        For early-exit cases (empty audio, too short) this is set to 1.0.
        Phase 2 uses this to discard frames that Whisper itself classifies as
        non-speech, complementing VAD pre-filtering.
    """

    text: str
    no_speech_prob: float

# Minimum number of samples required to attempt transcription.
# Whisper needs at least ~0.1 s of audio to be meaningful.
_MIN_SAMPLES = 1_600  # 0.1 s at 16 kHz

# Known whisper hallucination patterns (case-insensitive).
# These appear when the model receives silence or very short audio.
_ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\[BLANK_AUDIO\]", re.IGNORECASE),
    re.compile(r"\[MUSIC\]", re.IGNORECASE),
    re.compile(r"\[NOISE\]", re.IGNORECASE),
    re.compile(r"\[SILENCE\]", re.IGNORECASE),
    re.compile(r"\( silence \)", re.IGNORECASE),
    # YouTube / podcast hallucinations
    re.compile(r"Thank you for watching\.?", re.IGNORECASE),
    re.compile(r"Subscribe to our channel\.?", re.IGNORECASE),
    re.compile(r"Please subscribe\.?", re.IGNORECASE),
    re.compile(r"Like and subscribe\.?", re.IGNORECASE),
]

# If a single word repeats more than this many consecutive times, treat it as
# a hallucination loop and collapse / drop it.
_MAX_WORD_REPEAT = 4


# ---------------------------------------------------------------------------
# stderr logging helper
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Write a log line to stderr (never stdout)."""
    print(f"[lazy-claude stt] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Artifact stripping
# ---------------------------------------------------------------------------


def _strip_artifacts(text: str) -> str:
    """Remove known whisper hallucination tokens and clean up whitespace.

    Parameters
    ----------
    text:
        Raw transcription string from whisper.

    Returns
    -------
    str
        Cleaned text, possibly empty.
    """
    # Apply each hallucination pattern
    for pattern in _ARTIFACT_PATTERNS:
        text = pattern.sub("", text)

    # Collapse repeated words: "hello hello hello hello hello" → drop entirely
    # Match any word repeated more than _MAX_WORD_REPEAT times consecutively.
    def _collapse_repeats(m: re.Match[str]) -> str:
        word = m.group(1)
        # Keep a single occurrence only if it looks like real speech context
        # (i.e. surrounded by other text); in isolation it's noise — drop it.
        return ""

    repeat_re = re.compile(
        r"\b(\w+)(?:\s+\1){" + str(_MAX_WORD_REPEAT) + r",}\b",
        re.IGNORECASE,
    )
    text = repeat_re.sub(_collapse_repeats, text)

    # Strip leading / trailing whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Model loader (download on first use)
# ---------------------------------------------------------------------------


def model_name_for_language(language: str) -> str:
    """Return the appropriate Whisper model name for *language*.

    English uses the English-only ``small.en`` model (faster, more accurate).
    Other languages require the multilingual ``small`` model.

    Parameters
    ----------
    language:
        BCP-47 language code, e.g. ``'en'`` or ``'de'``.

    Returns
    -------
    str
        Whisper model name to pass to :func:`load_model`.
    """
    if language.lower().startswith('en'):
        return _DEFAULT_MODEL
    return _DEFAULT_MODEL_MULTILINGUAL


def load_model(model_name: str = _DEFAULT_MODEL) -> "Model":
    """Return a ready-to-use pywhispercpp Model instance.

    Downloads the GGML model on the first call and caches it in the
    pywhispercpp default models directory (platform user-data dir).

    Parameters
    ----------
    model_name:
        One of the pywhispercpp AVAILABLE_MODELS, e.g. ``"base.en"``,
        ``"small"``, ``"medium.en"``.  Default is ``"small.en"``.

    Returns
    -------
    Model
        A loaded pywhispercpp.model.Model instance, ready for transcription.
    """
    # Import lazily so the module can be imported without loading whisper.
    from pywhispercpp.model import Model  # noqa: PLC0415

    _log(f"Loading whisper model '{model_name}' …")

    # redirect_whispercpp_logs_to=sys.stderr keeps all C++ output off stdout.
    model = Model(
        model_name,
        redirect_whispercpp_logs_to=sys.stderr,
        print_progress=False,
        print_realtime=False,
        print_timestamps=False,
        print_special=False,
    )
    _log(f"Model '{model_name}' ready.")
    return model


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def transcribe(
    audio: np.ndarray,
    *,
    model: "Model | None" = None,
    model_name: str = _DEFAULT_MODEL,
    language: str = 'en',
) -> TranscribeResult:
    """Transcribe a 16 kHz mono float32 audio array.

    Parameters
    ----------
    audio:
        1-D float32 numpy array sampled at 16 kHz.
    model:
        Pre-loaded Model instance.  If not provided, one is loaded via
        ``load_model(model_name)``.
    model_name:
        Model name to use when ``model`` is not provided.  Ignored when
        *model* is given.  Defaults to ``small.en`` (English-only model).
        Pass ``model_name=model_name_for_language(lang)`` for other languages.
    language:
        BCP-47 language hint passed to Whisper, e.g. ``'de'`` for German.
        ``'en'`` (default) uses English-only fast path.  Set to ``'auto'`` for
        automatic detection (slower).

    Returns
    -------
    TranscribeResult
        ``.text`` is the recognised text, stripped of artefacts.  Empty
        string for empty or effectively silent input.
        ``.no_speech_prob`` is the average no-speech probability across all
        segments (1.0 for early-exit / empty cases).
    """
    # Guard: nothing to transcribe
    if audio is None or len(audio) == 0:
        return TranscribeResult(text="", no_speech_prob=1.0)

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < _MIN_SAMPLES:
        _log(f"Audio too short ({len(audio)} samples < {_MIN_SAMPLES}), returning empty.")
        return TranscribeResult(text="", no_speech_prob=1.0)

    if model is None:
        model = load_model(model_name)

    _log(f"Transcribing {len(audio) / 16_000:.2f}s of audio …")

    # Build whisper.cpp params — pass language hint when not English so the
    # model skips its auto-detection step and decodes directly in the target language.
    transcribe_params: dict = dict(
        print_progress=False,
        print_realtime=False,
        print_timestamps=False,
        print_special=False,
    )
    lang_lower = language.lower()
    if lang_lower not in ('en', 'english', ''):
        transcribe_params['language'] = lang_lower

    try:
        segments = model.transcribe(
            audio,
            **transcribe_params,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR during transcription: {exc}")
        return TranscribeResult(text="", no_speech_prob=1.0)

    # Join all segment texts and compute average no_speech_prob
    texts: list[str] = []
    no_speech_probs: list[float] = []
    for seg in segments:
        if seg.text:
            texts.append(seg.text)
        # pywhispercpp segments expose no_speech_prob; fall back to 0.0 if absent.
        nsp = getattr(seg, "no_speech_prob", 0.0)
        no_speech_probs.append(float(nsp))

    raw = " ".join(texts)
    avg_no_speech_prob = (
        sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
    )

    text = _strip_artifacts(raw)
    _log(f"Transcription: {text!r} (no_speech_prob={avg_no_speech_prob:.3f})")
    return TranscribeResult(text=text, no_speech_prob=avg_no_speech_prob)
