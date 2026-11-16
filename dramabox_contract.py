"""DramaBox job contract: input checks. No GPU, no model.

DramaBox reads one prose prompt: a speaker description, dialogue inside
double quotes (spoken), and stage directions outside the quotes (performed,
never spoken). An optional reference clip sets the voice.

design  no reference clip: the prompt's description picks the voice
clone   reference_url: the clip's first reference_seconds set the voice
"""
import re
import secrets

MODES = ("design", "clone")
MAX_PROMPT_CHARS = 4000
MAX_SINGLE_TAKE_S = 45.0   # upstream's per-take cap; longer scenes are split and crossfaded
SEED_MAX = 2**31 - 1

_CURLY = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"'})
_QUOTED = re.compile(r'"([^"]+)"')
_INNER_END = re.compile(r'([.!?]+)\s+(?=\S)')


class PlainError(ValueError):
    """A plain-words problem with the job, safe to show the caller word for word."""


def _number(inp, name, kind, low, high, default):
    value = inp.get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        raise PlainError(f"{name} must be a number.")
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise PlainError(f"{name} must be a number.") from None
    if kind is int and number != value and not (isinstance(value, float) and value.is_integer()):
        raise PlainError(f"{name} must be a whole number.")
    if not low <= number <= high:
        raise PlainError(f"{name} must be between {low} and {high}.")
    return number


def _flag(inp, name, default):
    value = inp.get(name, default)
    if not isinstance(value, bool):
        raise PlainError(f"{name} must be true or false.")
    return value


def clean_prompt(prompt):
    """One line, straight double quotes, balanced, with at least one spoken line.

    Upstream only knows the straight double quote; curly quotes would make the
    model speak the stage directions too.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise PlainError("prompt is required.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise PlainError(f"prompt is too long (at most {MAX_PROMPT_CHARS} characters).")
    prompt = re.sub(r"\s+", " ", prompt.translate(_CURLY)).strip()
    if prompt.count('"') % 2:
        raise PlainError('prompt has an unclosed double quote. Put every spoken line inside "double quotes".')
    if not any(q.strip() for q in _QUOTED.findall(prompt)):
        raise PlainError('prompt needs at least one spoken line inside "double quotes"; '
                         "words outside the quotes are performed, not spoken.")
    return prompt


def split_spoken_sentences(prompt):
    """'"A. B."' becomes '"A." "B."': still spoken, but upstream's quote-aware
    chunker can now cut a long speech between its sentences."""
    return _QUOTED.sub(lambda m: '"' + _INNER_END.sub(r'\1" "', m.group(1).strip()) + '"', prompt)


def parse(inp):
    """Check a job's input and return the plan the handler runs."""
    if not isinstance(inp, dict):
        raise PlainError("input must be an object.")
    prompt = clean_prompt(inp.get("prompt", inp.get("text")))
    url = inp.get("reference_url", inp.get("reference_voice_url"))
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise PlainError("reference_url must be a link to the clip.")
    mode = inp.get("mode")
    if mode is None:
        mode = "clone" if url else "design"
    if mode not in MODES:
        raise PlainError('mode must be "design" or "clone".')
    if mode == "design" and url:
        raise PlainError("design mode takes no reference clip. Use clone mode for a clip.")
    if mode == "clone" and not url:
        raise PlainError("clone mode needs reference_url.")
    rescale = inp.get("rescale_scale")
    if rescale is None or rescale == "auto":
        rescale = "auto"
    else:
        rescale = _number(inp, "rescale_scale", float, 0.0, 1.0, None)
    duration = _number(inp, "duration", float, 0.0, MAX_SINGLE_TAKE_S, 0.0)
    if 0 < duration < 3.0:
        raise PlainError("duration must be 0 (automatic) or between 3 and 45 seconds.")
    spec = {
        "mode": mode,
        "prompt": prompt,
        "seed": _number(inp, "seed", int, 0, SEED_MAX, None),
        "cfg_scale": _number(inp, "cfg_scale", float, 1.0, 10.0, 2.5),
        "stg_scale": _number(inp, "stg_scale", float, 0.0, 5.0, 1.5),
        "rescale_scale": rescale,
        "duration_multiplier": _number(inp, "duration_multiplier", float, 0.8, 2.0, 1.1),
        "duration": duration,
        "reference_seconds": _number(inp, "reference_seconds", float, 3.0, 30.0, 10.0),
        "watermark": _flag(inp, "watermark", True),
    }
    if spec["seed"] is None:
        spec["seed"] = secrets.randbelow(SEED_MAX) + 1
    if mode == "clone":
        spec["reference_url"] = url.strip()
    return spec
