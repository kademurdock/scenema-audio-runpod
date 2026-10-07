"""AuK requests and legacy screenplay compatibility. No GPU imports."""
import math
import json
import re
import xml.etree.ElementTree as ET


def retain_bootstrap_sample(inp, env):
    """An ops-only opt-in for one authenticated bridge owner's next normal job."""
    owner = env.get('AUK_DIAGNOSTIC_USER_ID', '')
    # The bridge supplies out_prefix from its trusted userId, not speech text.
    # A request-side boolean cannot enable retention for another account.
    return (bool(re.fullmatch(r'[a-f0-9]{24}', owner)) and
            inp.get('out_prefix') == owner and inp.get('voice_sample') is True and
            inp.get('auk_task', 'speech') == 'speech' and
            not inp.get('reference_voice_url'))


def edit_windows(total, inp):
    """Frame-aligned, bounded windows, including a selected range's untouched edges."""
    start = float(inp.get('edit_start') or 0)
    end = float(inp['edit_end']) if inp.get('edit_end') is not None else total
    if not all(math.isfinite(x) for x in (total, start, end)) or total <= 0 or start < 0 or start >= total or end <= start or end > total + 0.05:
        raise ValueError('Choose an edit range inside the recording and a positive target length.')
    end = min(total, end)
    target = float(inp['gen_seconds']) if inp.get('gen_seconds') is not None else end - start
    if not math.isfinite(target) or target <= 0:
        raise ValueError('Choose an edit range inside the recording and a positive target length.')
    count = max(1, math.ceil((end - start + target) / 28))
    if count > 360:
        raise ValueError('Select a shorter recording section for this edit.')
    return [{
        'start': start + (end - start) * i / count,
        'end': end if i == count - 1 else start + (end - start) * (i + 1) / count,
        'seconds': target / count,
    } for i in range(count)]


def prepare_edit_instruction(instruction):
    """Put single voice-change requests in AuK's documented timbre template."""
    match = re.fullmatch(
        r'(?:please\s+)?(?:make|change|turn|convert)\s+(?:this|the)\s+'
        r'(?:speaking\s+)?voice\s+(?:sound\s+like|into|to)\s+([^\r\n.!?;]+?)[.!]?',
        instruction, re.IGNORECASE)
    if not match:
        return instruction
    description = match.group(1).strip()
    if not description or re.search(
            r'\b(?:replace|insert|remove|delete|add|then|say|sing|words|lyrics|content|'
            r'raise|lower|adjust|increase|decrease|separate)\b', description, re.IGNORECASE):
        return instruction
    return ('Keep the spoken content unchanged and change the timbre to: ' +
            json.dumps(description, ensure_ascii=False) + '.')


def model_instruction(piece, has_reference=False):
    """Use AuK's speech and edit contracts while preserving the saved request."""
    if "text" not in piece:
        return prepare_edit_instruction(piece["instruction"])
    text = json.dumps(piece["text"], ensure_ascii=False)
    if has_reference:
        return f"Say the following with the same voice: {text}"
    direction = json.dumps(piece["direction"], ensure_ascii=False)
    return (f"Generate speech based on the following description: {direction}. "
            f"The content to speak is: {text}.")


def plan(inp):
    task = inp.get("auk_task", "speech")
    if task not in ("speech", "edit"):
        raise ValueError("Choose speech or edit.")
    seed = inp.get("seed", 42)
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer from 0 to 4294967295.")
    if task == "edit":
        instruction = str(inp.get("instruction") or "").strip()
        if not instruction or not inp.get("reference_voice_url"):
            raise ValueError("Editing needs an imported recording and an instruction.")
        seconds = inp.get("gen_seconds")
        if seconds is not None:
            seconds = float(seconds)
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Target seconds must be positive.")
        return [{"instruction": instruction, "seconds": seconds, "seed": seed}]
    prompt = str(inp.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Write the words to perform first.")
    voice = str(inp.get("voice_description") or "A natural, expressive conversational voice.")
    if "<!" in prompt:
        raise ValueError("Declarations are not supported in screenplay XML.")
    if prompt.startswith("<speak"):
        root = ET.fromstring(prompt)
        voice = root.attrib.get("voice", voice)
        # Preserve acting direction; sound effects belong in the Seed Audio lane.
        blocks = [(voice, root.text or "")]
        direction = voice
        for node in root:
            if node.tag == "action":
                direction = voice + ". " + " ".join(node.itertext())
            elif node.tag != "sound":
                blocks.append((direction, " ".join(node.itertext())))
            blocks.append((direction, node.tail or ""))
    else:
        blocks = [(voice, prompt)]
    pace = float(inp.get("pace", 1))
    if not math.isfinite(pace) or not 0.5 <= pace <= 3:
        raise ValueError("Pace must be between 0.5 and 3.")
    result = []
    for direction, words in blocks:
        # Short acoustic windows prevent memory growth; the project can be long.
        tokens = words.split()
        while tokens:
            end = min(max(8, int(18 * 2.6 / pace)), len(tokens))
            if len(tokens) > end:
                for i in range(end, max(8, end // 2), -1):
                    if re.search(r'[.!?;][\"\u201d\u2019]*$', tokens[i - 1]):
                        end = i
                        break
            text = " ".join(tokens[:end]); tokens = tokens[end:]
            piece = {"direction": direction, "text": text,
                     "seconds": max(1, end * pace / 2.6 + 0.5), "seed": seed}
            piece["instruction"] = model_instruction(piece)
            result.append(piece)
    if not result:
        raise ValueError("The screenplay contains no spoken words.")
    if inp.get("voice_sample") is True and not inp.get("reference_voice_url"):
        result.insert(0, voice_sample_piece(result[0], pace))
    return result


def voice_sample_piece(first, pace=1):
    """A private opening take that only sets the voice (Oct 2 2026).

    Instruct TTS sometimes speaks the voice description aloud: a preview began
    with the description's own words before the script. With voice_sample on,
    the description meets only this throwaway take of the script's opening
    words; the handler keeps its first eight seconds as the reference and
    every piece the listener hears is same-voice speech. Never in the output.
    """
    tokens = first["text"].split()
    end = min(12, len(tokens))
    for i in range(min(16, len(tokens)), 5, -1):
        if re.search(r'[.!?;][\"\u201d\u2019]*$', tokens[i - 1]):
            end = i
            break
    piece = {"direction": first["direction"], "text": " ".join(tokens[:end]),
             "seconds": max(1, end * pace / 2.6 + 0.5), "seed": first["seed"], "sample": True}
    piece["instruction"] = model_instruction(piece)
    return piece
