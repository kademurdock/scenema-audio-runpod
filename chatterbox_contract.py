"""Chatterbox job contract: input checks, sound tags, text splitting. No GPU, no model.

One worker serves three Resemble AI Chatterbox models, picked by "model":

turbo         Chatterbox-Turbo (350M, English, one-step decoder). Reads the
              sound tags in SOUND_TAGS inline. Only temperature is a knob.
original      Chatterbox 0.5B (English) with exaggeration and cfg_weight.
multilingual  Chatterbox Multilingual V3 (0.5B, 23 languages, language_id).

Mode names match the VoxCPM2 worker: "clone" speaks in the voice of a clip at
reference_url; "design" uses the model's own built-in voice (Chatterbox cannot
make a voice from a description, so a description is refused, not ignored).
"""
import re
import secrets

MODELS = ("turbo", "original", "multilingual")
DEFAULT_MODEL = "turbo"
MODES = ("design", "clone")
MAX_TEXT_CHARS = 5000
# Upstream's demos cap one generate() call at 300 characters; each call stops
# at 1000 speech tokens (40 s). Dense scripts get shorter parts.
CHUNK_CHARS = 300
DENSE_CHUNK_CHARS = 150
DENSE_LANGUAGES = ("zh", "ja", "ko")
SEED_MAX = 2**31 - 1

# Chatterbox-Turbo's added tokenizer tokens (ResembleAI/chatterbox-turbo
# added_tokens.json). The first nine are the event tags upstream's Turbo demo
# offers; the other ten are style tokens the tokenizer also carries but
# upstream does not document.
EVENT_TAGS = ("[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
              "[sniff]", "[gasp]", "[chuckle]", "[laugh]")
STYLE_TAGS = ("[angry]", "[fear]", "[surprised]", "[whispering]", "[advertisement]",
              "[dramatic]", "[narration]", "[crying]", "[happy]", "[sarcastic]")
SOUND_TAGS = EVENT_TAGS + STYLE_TAGS
# Other engines' spellings, so a line written for them still works.
TAG_ALIASES = {
    "laughs": "[laugh]", "laughing": "[laugh]", "laughter": "[laugh]", "chuckles": "[chuckle]",
    "giggle": "[chuckle]", "giggles": "[chuckle]", "sighs": "[sigh]", "sighing": "[sigh]",
    "coughs": "[cough]", "coughing": "[cough]", "gasps": "[gasp]", "groans": "[groan]",
    "sniffs": "[sniff]", "sniffle": "[sniff]", "throat clear": "[clear throat]",
    "clears throat": "[clear throat]", "clearing throat": "[clear throat]", "shh": "[shush]",
    "whisper": "[whispering]", "whispers": "[whispering]", "cries": "[crying]", "crying": "[crying]",
    "sarcasm": "[sarcastic]", "surprise": "[surprised]", "scared": "[fear]",
}
_TAGS_BY_NAME = {tag[1:-1]: tag for tag in SOUND_TAGS}
_TAG_RE = re.compile(r"\[([^\[\]]{1,40})\]")
_KEEP = ""  # holds the space inside [clear throat] while text is split

# Chatterbox Multilingual's language_id codes (chatterbox.mtl_tts.SUPPORTED_LANGUAGES).
LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

# Knob ranges and defaults from upstream's generate() signatures and demo sliders.
TEMPERATURE = {"turbo": (0.05, 2.0), "original": (0.05, 5.0), "multilingual": (0.05, 5.0)}
DEFAULT_TEMPERATURE = 0.8
EXAGGERATION = (0.25, 2.0, 0.5)
CFG_WEIGHT = (0.0, 1.0, 0.5)

# Sentence ends: Latin marks need a space or the end after them; CJK marks do not.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)|[。！？]+[”’）」』]*")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+|(?<=[，；：、])")


def _number(inp, name, kind, low, high, default):
    value = inp.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if kind is int and number != value and not (isinstance(value, float) and value.is_integer()):
        raise ValueError(f"{name} must be a whole number.")
    if not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return number


def _tag_name(raw):
    return re.sub(r"\s+", " ", raw).strip().lower()


def sound_tags(text, model):
    """Rewrite known sound tags to Turbo's spelling and return (text, tags used).

    Turbo would read an unknown [word] aloud, so it is refused. The other two
    models have no sound tags: a known tag there is refused too, and any other
    bracketed words are left to be spoken.
    """
    used = []

    def swap(match):
        name = _tag_name(match.group(1))
        tag = _TAGS_BY_NAME.get(name) or TAG_ALIASES.get(name)
        if model != "turbo":
            if tag is not None:
                raise ValueError(f"Sound tags such as {tag} work only with model turbo.")
            return match.group(0)
        if tag is None:
            raise ValueError(f"Unknown sound tag [{match.group(1)}]. Turbo knows: {' '.join(SOUND_TAGS)}.")
        used.append(tag)
        return tag

    return _TAG_RE.sub(swap, text), used


def _hard_split(piece, limit):
    out, line = [], ""
    for word in piece.split():
        while len(word) > limit:  # one unbroken run longer than a chunk
            if line:
                out.append(line)
                line = ""
            out.append(word[:limit])
            word = word[limit:]
        if line and len(line) + 1 + len(word) > limit:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        out.append(line)
    return out


def _sentences(text):
    """Sentences with their end marks and any closing quote or bracket kept."""
    start = 0
    for match in _SENTENCE_END.finditer(text):
        yield text[start:match.end()]
        start = match.end()
    yield text[start:]


def _join(current, unit):
    if not current:
        return unit
    # No space between CJK pieces that had none.
    sep = "" if re.match(r"[　-鿿가-힯＀-￯]", unit) and \
        re.search(r"[　-鿿가-힯＀-￯]$", current) else " "
    return current + sep + unit


def split_text(text, limit=CHUNK_CHARS):
    """Split at sentence ends into chunks of at most `limit` characters, in order.

    Sentences are packed together while they fit. A sentence that is too long
    alone is split at commas and similar marks, then at spaces.
    """
    text = text.replace("[clear throat]", f"[clear{_KEEP}throat]")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    units = []
    for sentence in _sentences(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            units.append(sentence)
            continue
        for clause in _CLAUSE_END.split(sentence):
            clause = clause.strip()
            if clause:
                units.extend([clause] if len(clause) <= limit else _hard_split(clause, limit))
    chunks, current = [], ""
    for unit in units:
        joined = _join(current, unit)
        if current and len(joined) > limit:
            chunks.append(current)
            current = unit
        else:
            current = joined
    if current:
        chunks.append(current)
    return [c.replace(_KEEP, " ") for c in chunks]


def _words(inp, *names):
    for name in names:
        value = inp.get(name)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return name
    return None


def parse(inp):
    """Check a job's input and return the plan the handler runs."""
    if not isinstance(inp, dict):
        raise ValueError("input must be an object.")
    model = inp.get("model", DEFAULT_MODEL)
    if isinstance(model, str):
        model = model.strip().lower()
    if model not in MODELS:
        raise ValueError('model must be "turbo", "original" or "multilingual".')

    reference = inp.get("reference_url", inp.get("reference_voice_url"))
    if reference is not None and not isinstance(reference, str):
        raise ValueError("reference_url must be a link.")
    reference = (reference or "").strip() or None
    mode = inp.get("mode")
    if mode is None:
        mode = "clone" if reference else "design"
    if mode not in MODES:
        raise ValueError('mode must be "design" or "clone".')
    if mode == "clone" and not reference:
        raise ValueError("clone mode needs reference_url (a clip of the voice).")
    if mode == "design" and reference:
        raise ValueError("design mode uses the built-in voice and takes no clip. Use clone mode for a clip.")
    described = _words(inp, "description", "voice_description", "style")
    if described:
        raise ValueError(f"Chatterbox cannot make a voice from words, so {described} is not used. "
                         "Leave it out for the built-in voice, or send a clip in clone mode.")

    text = inp.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is too long (at most {MAX_TEXT_CHARS} characters).")
    text, tags = sound_tags(re.sub(r"\s+", " ", text).strip(), model)
    if model == "turbo" and not re.sub(_TAG_RE, "", text).strip(" .,!?"):
        raise ValueError("text has no words to speak.")

    language = inp.get("language_id", inp.get("language"))
    if language is not None:
        if not isinstance(language, str) or language.strip().lower() not in LANGUAGES:
            raise ValueError(f"language_id must be one of: {', '.join(LANGUAGES)}.")
        language = language.strip().lower()
    if model == "multilingual":
        if language is None:
            raise ValueError(f"model multilingual needs language_id, one of: {', '.join(LANGUAGES)}.")
    elif language not in (None, "en"):
        raise ValueError(f"model {model} speaks English only. Use model multilingual for "
                         f"{LANGUAGES[language]}.")

    low, high = TEMPERATURE[model]
    spec = {
        "model": model,
        "mode": mode,
        "seed": _number(inp, "seed", int, 0, SEED_MAX, None),
        "temperature": _number(inp, "temperature", float, low, high, DEFAULT_TEMPERATURE),
        "language_id": language if model == "multilingual" else "en",
        "sound_tags": tags,
        "ignored": [],
    }
    if spec["seed"] is None:
        spec["seed"] = secrets.randbelow(SEED_MAX) + 1
    if model == "turbo":
        # Turbo's generate() ignores both (upstream logs a warning), so they
        # are reported back as ignored rather than refused.
        spec["exaggeration"] = spec["cfg_weight"] = None
        spec["ignored"] = [name for name in ("exaggeration", "cfg_weight") if inp.get(name) is not None]
        for name in spec["ignored"]:
            _number(inp, name, float, 0.0, 2.0, None)
    else:
        spec["exaggeration"] = _number(inp, "exaggeration", float, *EXAGGERATION)
        spec["cfg_weight"] = _number(inp, "cfg_weight", float, *CFG_WEIGHT)
    if inp.get("prompt_text") is not None:
        spec["ignored"].append("prompt_text")  # Chatterbox needs no transcript of the clip.
    if reference:
        spec["reference_url"] = reference
    limit = DENSE_CHUNK_CHARS if spec["language_id"] in DENSE_LANGUAGES else CHUNK_CHARS
    spec["chunks"] = split_text(text, limit)
    return spec
