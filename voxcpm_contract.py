"""VoxCPM2 job contract: input checks and text splitting. No GPU, no model."""
import re
import secrets

MODES = ("design", "clone")
MAX_TEXT_CHARS = int(5000)
MAX_PREFIX_CHARS = 400
CHUNK_CHARS = 300
SEED_MAX = 2**31 - 1

_SENTENCE_END = re.compile(r"[.!?。！？…]+[\"'”’)\]]*(?=\s|$)")
_CLAUSE_END = re.compile(r"(?<=[,;:—，；：])\s+")


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


def _flag(inp, name, default):
    value = inp.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def clean_prefix(words, name):
    """A description or style becomes the one parenthesised lead VoxCPM2 reads.

    Brackets inside it would end the lead early and be spoken, so they go.
    """
    if words is None:
        return ""
    if not isinstance(words, str):
        raise ValueError(f"{name} must be text.")
    words = re.sub(r"[()（）\[\]]", " ", words)
    words = re.sub(r"\s+", " ", words).strip()
    if len(words) > MAX_PREFIX_CHARS:
        raise ValueError(f"{name} is too long (at most {MAX_PREFIX_CHARS} characters).")
    return words


def with_prefix(prefix, text):
    """VoxCPM2's own examples put the lead right before the text, no space."""
    return f"({prefix}){text}" if prefix else text


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


def split_text(text, limit=CHUNK_CHARS):
    """Split at sentence ends into chunks of at most `limit` characters, in order.

    Sentences are packed together while they fit. A sentence that is too long
    alone is split at commas and similar marks, then at spaces.
    """
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
        if current and len(current) + 1 + len(unit) > limit:
            chunks.append(current)
            current = unit
        else:
            current = f"{current} {unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def parse(inp):
    """Check a job's input and return the plan the handler runs."""
    if not isinstance(inp, dict):
        raise ValueError("input must be an object.")
    mode = inp.get("mode")
    if mode not in MODES:
        raise ValueError('mode must be "design" or "clone".')
    text = inp.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is too long (at most {MAX_TEXT_CHARS} characters).")
    spec = {
        "mode": mode,
        "seed": _number(inp, "seed", int, 0, SEED_MAX, None),
        "cfg_value": _number(inp, "cfg_value", float, 0.5, 5.0, 2.0),
        "inference_timesteps": _number(inp, "inference_timesteps", int, 4, 50, 10),
        "normalize": _flag(inp, "normalize", False),
        "chunks": split_text(text, CHUNK_CHARS),
    }
    if spec["seed"] is None:
        spec["seed"] = secrets.randbelow(SEED_MAX) + 1
    if mode == "design":
        if inp.get("reference_voice_url") or inp.get("prompt_text"):
            raise ValueError("design mode takes a description, not a clip. Use clone mode for a clip.")
        spec["description"] = clean_prefix(inp.get("description", inp.get("voice_description")), "description")
        if not spec["description"]:
            raise ValueError("design mode needs a description of the voice.")
    else:
        url = inp.get("reference_voice_url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("clone mode needs reference_voice_url.")
        spec["reference_voice_url"] = url.strip()
        spec["style"] = clean_prefix(inp.get("style"), "style")
        prompt_text = inp.get("prompt_text")
        if prompt_text is not None:
            if not isinstance(prompt_text, str):
                raise ValueError("prompt_text must be text.")
            prompt_text = re.sub(r"\s+", " ", prompt_text).strip()
            if len(prompt_text) > 1000:
                raise ValueError("prompt_text is too long (at most 1000 characters).")
        spec["prompt_text"] = prompt_text or None
    return spec
