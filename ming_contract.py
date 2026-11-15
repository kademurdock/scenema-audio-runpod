"""Ming-3.0-tts-tiny job contract: input checks and task choice. No GPU, no model.

The job shape matches the VoxCPM2 worker (mode "design" or "clone", text,
optional seed). Ming runs one task per call and rejects mixed controls, so
this module maps a job onto exactly one of its tasks:

design  description only              -> voice_design (style = description)
        description + style           -> freeform (one instruction, no clip)
        voice (a built-in voice name) -> ip
clone   emotion                       -> emotion (speaker embedding of the clip)
        speed / pitch / volume        -> basic (missing ones are "medium")
        style                         -> freeform with the clip as speaker
        prompt_text                   -> seed, multilingual, or paralinguistic
                                         when the text has sound tags
        nothing else                  -> basic at medium speed, pitch, volume
"""
import re
import secrets

MODES = ("design", "clone")
MAX_TEXT_CHARS = 5000
MAX_PREFIX_CHARS = 600
MAX_PROMPT_TEXT_CHARS = 1000
SEED_MAX = 2**31 - 1

# Ming's 14 inline sound tags (ming_tiny.task_protocol.PARALINGUISTIC_TAGS),
# case-sensitive in the model; matched here without case and rewritten.
SOUND_TAGS = (
    "[Breathing]", "[Cough]", "[Sigh]", "[Laughter]", "[Surprise-ah]", "[Surprise-oh]",
    "[Dissatisfaction-hnn]", "[Uhm]", "[Confirmation-en]", "[Question-ei]", "[Question-ah]",
    "[Question-en]", "[Question-oh]", "[Question-huh]",
)
# Common spellings from other engines, so a line written for them still works.
TAG_ALIASES = {
    "laugh": "[Laughter]", "laughs": "[Laughter]", "laughing": "[Laughter]", "chuckle": "[Laughter]",
    "chuckles": "[Laughter]", "giggle": "[Laughter]", "giggles": "[Laughter]",
    "sighs": "[Sigh]", "sighing": "[Sigh]", "coughs": "[Cough]", "coughing": "[Cough]",
    "breath": "[Breathing]", "breathes": "[Breathing]", "breathe": "[Breathing]", "inhale": "[Breathing]",
    "inhales": "[Breathing]", "exhale": "[Breathing]", "exhales": "[Breathing]",
    "um": "[Uhm]", "uh": "[Uhm]", "umm": "[Uhm]", "hmm": "[Uhm]", "huh": "[Question-huh]",
}
_TAGS_BY_NAME = {tag[1:-1].lower(): tag for tag in SOUND_TAGS}
_TAG_RE = re.compile(r"\[([A-Za-z][A-Za-z -]{0,30})\]")

# English words for Ming's fixed control values (task_protocol.py).
SPEEDS = {"slow": "慢速", "medium": "中速", "normal": "中速", "fast": "快速",
          "慢速": "慢速", "中速": "中速", "快速": "快速"}
LEVELS = {"low": "低", "medium": "中", "normal": "中", "high": "高", "低": "低", "中": "中", "高": "高"}
EMOTIONS = {"happy": "高兴", "sad": "悲伤", "angry": "愤怒", "neutral": "中性", "surprised": "惊讶",
            "fearful": "恐惧", "disgusted": "厌恶",
            "高兴": "高兴", "悲伤": "悲伤", "愤怒": "愤怒", "中性": "中性", "惊讶": "惊讶", "恐惧": "恐惧", "厌恶": "厌恶"}
# target_language names Ming accepts; the last five use its multilingual task.
LANGUAGES = {"en": "English", "english": "English", "zh": "Chinese", "chinese": "Chinese",
             "de": "German", "german": "German", "es": "Spanish", "spanish": "Spanish",
             "fr": "French", "french": "French", "ko": "Korean", "korean": "Korean",
             "ja": "Japanese", "japanese": "Japanese"}
VOICES = ("US_Emma", "UK_George", "UK_Charlotte")  # documented English built-ins; others pass through
_CJK = re.compile(r"[一-鿿]")


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


def clean_words(words, name, limit=MAX_PREFIX_CHARS):
    """A description, style or transcript as one clean line."""
    if words is None:
        return ""
    if not isinstance(words, str):
        raise ValueError(f"{name} must be text.")
    words = re.sub(r"\s+", " ", words).strip()
    if name in ("description", "style"):
        # A caller may wrap a lead in brackets the way VoxCPM2 reads it.
        words = re.sub(r"^[(（\[]\s*(.*?)\s*[)）\]]$", r"\1", words)
    if len(words) > limit:
        raise ValueError(f"{name} is too long (at most {limit} characters).")
    return words


def _choice(inp, name, table):
    value = inp.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in table:
        names = sorted({k for k in table if k.isascii()})
        raise ValueError(f"{name} must be one of: {', '.join(names)}.")
    return table[value.strip().lower()]


def sound_tags(text):
    """Rewrite known sound tags to Ming's spelling; refuse unknown ones.

    Returns (text, tags used in order). An unknown [word] would be read aloud.
    """
    used = []

    def swap(match):
        name = match.group(1).strip().lower()
        tag = _TAGS_BY_NAME.get(name) or TAG_ALIASES.get(name)
        if tag is None:
            raise ValueError(f"Unknown sound tag [{match.group(1)}]. Ming knows: {' '.join(SOUND_TAGS)}.")
        used.append(tag)
        return tag

    return _TAG_RE.sub(swap, text), used


def detect_language(text):
    return "Chinese" if _CJK.search(text) else "English"


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
    text, tags = sound_tags(re.sub(r"\s+", " ", text).strip())
    if not re.sub(_TAG_RE, "", text).strip():
        raise ValueError("text has no words to speak.")
    language = _choice(inp, "language", LANGUAGES)
    spec = {
        "mode": mode,
        "text": text,
        "sound_tags": tags,
        "seed": _number(inp, "seed", int, 0, SEED_MAX, None),
        "cfg_scale": _number(inp, "cfg_scale", float, 1.0, 5.0, 2.0),
        "flow_steps": _number(inp, "flow_steps", int, 4, 50, 10),
        "normalize": _flag(inp, "normalize", True),
        "args": {},
    }
    if spec["seed"] is None:
        spec["seed"] = secrets.randbelow(SEED_MAX) + 1
    description = clean_words(inp.get("description", inp.get("voice_description")), "description")
    style = clean_words(inp.get("style"), "style")
    emotion = _choice(inp, "emotion", EMOTIONS)
    speed = _choice(inp, "speed", SPEEDS)
    pitch = _choice(inp, "pitch", LEVELS)
    volume = _choice(inp, "volume", LEVELS)
    controls = emotion or speed or pitch or volume
    args = spec["args"]

    if mode == "design":
        if inp.get("reference_voice_url") or inp.get("prompt_text"):
            raise ValueError("design mode takes a description, not a clip. Use clone mode for a clip.")
        if controls:
            raise ValueError("emotion, speed, pitch and volume need clone mode; describe them in style instead.")
        if tags:
            raise ValueError("Sound tags need clone mode with prompt_text (Ming's paralinguistic task).")
        if language and language not in ("English", "Chinese"):
            raise ValueError("Other languages need clone mode with prompt_text.")
        voice = inp.get("voice")
        if voice is not None:
            if not isinstance(voice, str) or not re.fullmatch(r"[\w一-鿿-]{1,40}", voice.strip()):
                raise ValueError("voice must be a built-in voice name such as US_Emma.")
            if description or style:
                raise ValueError("Use either a built-in voice or a description, not both.")
            spec["task"], args["ip"] = "ip", voice.strip()
        elif not description:
            raise ValueError("design mode needs a description of the voice.")
        elif style:
            spec["task"], args["instruction"] = "freeform", f"{description.rstrip('.。')}. {style}"
        else:
            spec["task"], args["style"] = "voice_design", description
        return spec

    url = inp.get("reference_voice_url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("clone mode needs reference_voice_url.")
    spec["reference_voice_url"] = url.strip()
    prompt_text = clean_words(inp.get("prompt_text"), "prompt_text", MAX_PROMPT_TEXT_CHARS)
    if inp.get("voice") is not None:
        raise ValueError("voice is a design-mode option.")
    if sum(bool(x) for x in (emotion, speed or pitch or volume, style)) > 1:
        raise ValueError("Ming takes one of style, emotion, or speed/pitch/volume per take, not a mix.")
    if tags and (controls or style or not prompt_text):
        raise ValueError("Sound tags need prompt_text (the clip's exact words) and no style or controls.")
    if language and language not in ("English", "Chinese") and (controls or style or not prompt_text):
        raise ValueError(f"{language} needs prompt_text (the clip's exact words) and no style or controls.")
    if emotion:
        spec["task"], args["emotion"] = "emotion", emotion
    elif speed or pitch or volume:
        spec["task"] = "basic"
        args.update(speed=speed or "中速", pitch=pitch or "中", volume=volume or "中")
    elif style:
        spec["task"], args["instruction"] = "freeform", style
    elif prompt_text:
        language = language or detect_language(text)
        spec["task"] = ("paralinguistic" if tags else
                        "seed" if language in ("English", "Chinese") else "multilingual")
        args.update(prompt_text=prompt_text, target_language=language)
    else:
        # Ming's full clone (seed) needs the clip's words. Without them the
        # clip still sets the voice through its speaker embedding.
        spec["task"] = "basic"
        args.update(speed="中速", pitch="中", volume="中")
    spec["uses_prompt_text"] = "prompt_text" in args
    return spec
