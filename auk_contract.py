"""AuK requests and legacy screenplay compatibility. No GPU imports."""
import math
import re
import xml.etree.ElementTree as ET


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
            instruction = (f'Based on the following description: "{direction}", '
                           f'generate speech content "{text}".')
            result.append({"instruction": instruction, "text": text,
                           "seconds": max(1, end * pace / 2.6 + 0.5), "seed": seed})
    if not result:
        raise ValueError("The screenplay contains no spoken words.")
    return result
