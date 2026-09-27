"""Word timing for YuE2 covers, run in the isolated /opt/cover environment (torch and torchaudio 2.8).

Usage: align.py <audio.f32> <words.json> <out.json>
       align.py --selftest

<audio.f32> is raw little-endian 32-bit float stereo PCM at 44.1 kHz
(ffmpeg -f f32le -acodec pcm_f32le -ac 2 -ar 44100). <words.json> is {"words": [...]}, each
entry lower-case a-z and apostrophes (lyric_sync.aligner_text); an empty entry has nothing to
align and comes back null. An optional "star_after": [bool per word] lets MMS_FA's star token
(any vocal the lyrics do not name, such as an ad-lib between lines) sit after those words, and
at the very start and end; without it the alignment is exactly as before.

1. torchaudio's Hybrid Demucs (HDEMUCS_HIGH_MUSDB_PLUS) separates the vocal stem, in
   overlapping 10 s chunks (torchaudio's own recipe).
2. torchaudio's MMS forced aligner (MMS_FA: wav2vec 2.0 CTC, 20 ms frames) places the known
   words on that stem. Emissions are computed in 30 s windows with 2 s of context each side.
3. A vocal-energy envelope gives each word the longest quiet run just before it, so a take's
   pauses and held notes can be told apart without trusting CTC word ends.

Both models are baked into the image under $YUE_ALIGN_HOME (a torch hub cache), so nothing is
downloaded at run time. Writes {"words": [{"start", "end", "score", "quiet_before"} | null],
"seconds", "device", "peak_gib", "timing"}, or {"error": ...} with exit status 2."""
import json
import math
import os
import sys
import time

os.environ.setdefault('TORCH_HOME', os.environ.get('YUE_ALIGN_HOME', '/opt/align'))

try:
    import torch  # noqa: E402
    import torchaudio  # noqa: E402
    from torchaudio.transforms import Fade  # noqa: E402
except ImportError:  # the worker's CPU tests import targets() without torch
    torch = torchaudio = Fade = None

MIX_RATE = 44100
ALIGN_RATE = 16000
HOP = 320                        # one MMS_FA frame: 20 ms at 16 kHz
WINDOW = 30 * ALIGN_RATE         # emission window in samples (a whole number of frames)
CONTEXT = 2 * ALIGN_RATE         # context either side of a window, trimmed off
RECEPTIVE = 400                  # samples one wav2vec 2.0 frame sees
QUIET_DB = 25.0                  # this far below the stem's loud level counts as quiet
SEGMENT, OVERLAP = 10.0, 0.1     # separation chunk seconds and overlap share


def pick_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_mix(path):
    with open(path, 'rb') as stream:
        audio = torch.frombuffer(bytearray(stream.read()), dtype=torch.float32)
    return audio[: audio.numel() // 2 * 2].reshape(-1, 2).T.contiguous()


def vocal_stem(mix, device):
    """Mono vocal stem at 44.1 kHz from a [2, samples] mix."""
    model = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model().to(device).eval()
    vocals = list(model.sources).index('vocals')
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    mix = ((mix - mean) / std).to(device)[None]
    length = mix.shape[-1]
    chunk = int(MIX_RATE * SEGMENT * (1 + OVERLAP))
    overlap = int(MIX_RATE * OVERLAP)
    fade = Fade(fade_in_len=0, fade_out_len=overlap, fade_shape='linear')
    out = torch.zeros(1, 2, length, device=device)
    start, end = 0, chunk
    with torch.inference_mode():
        while start < length - overlap:
            piece = fade(model(mix[:, :, start:end]))
            out[:, :, start:end] += piece[:, vocals]
            if start == 0:
                fade.fade_in_len = overlap
                start += chunk - overlap
            else:
                start += chunk
            end += chunk
            if end >= length:
                fade.fade_out_len = 0
    stem = (out[0] * std + mean).mean(0).float().cpu()
    del model, out, mix
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return stem


def emissions(stem, device, star=False):
    """[frames, labels] CTC log-probabilities of the 16 kHz stem, and the label dictionary.
    With star, the last label is MMS_FA's star token at log-probability 0 on every frame (as
    torchaudio's own recipe appends it), after the letters' log-softmax."""
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=star).to(device).eval()
    rows, n = [], stem.numel()
    with torch.inference_mode():
        for start in range(0, n, WINDOW):
            a, b = max(0, start - CONTEXT), min(n, start + WINDOW + CONTEXT)
            if b - a < RECEPTIVE:
                break
            logits, _ = model(stem[a:b][None].to(device))
            logits = logits[0].float()
            if star:
                probs = torch.cat([torch.log_softmax(logits[:, :-1], dim=-1), torch.zeros_like(logits[:, -1:])], dim=-1)
            else:
                probs = torch.log_softmax(logits, dim=-1)
            rows.append(probs.cpu()[(start - a) // HOP:(start - a) // HOP + (min(start + WINDOW, n) - start) // HOP])
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return torch.cat(rows), bundle.get_dict(star='*' if star else None)


def quiet_frames(stem):
    """Per 20 ms frame, whether the vocal stem is quiet (QUIET_DB below its 95th percentile)."""
    usable = stem[: stem.numel() // HOP * HOP]
    if not usable.numel():
        return []
    db = 20 * torch.log10(usable.reshape(-1, HOP).pow(2).mean(1).sqrt() + 1e-9)
    return (db < torch.quantile(db, 0.95) - QUIET_DB).tolist()


def longest_run(flags, lo, hi):
    best = run = 0
    for flag in flags[max(lo, 0):max(hi, 0)]:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def targets(ids, stars=None, star=None):
    """The aligner's target tokens: every word's letters in order, plus a star token at the start,
    at the end and after each word stars marks (never two in a row), when star is a label.
    Returns (tokens, owner): owner[t] is token t's word index, or None for a star."""
    tokens, owner = [], []

    def add_star():
        if star is not None and stars is not None and (not owner or owner[-1] is not None):
            tokens.append(star)
            owner.append(None)
    add_star()
    for index, word in enumerate(ids):
        tokens.extend(word)
        owner.extend([index] * len(word))
        if stars is not None and index < len(stars) and stars[index]:
            add_star()
    add_star()
    return tokens, owner


def place(emission, dictionary, words, quiet, stars=None):
    """Forced alignment of the known words; one entry per word, None for an empty word.
    stars: per word, whether the star token may follow it (dictionary must then hold '*')."""
    blank = dictionary.get('-', 0)
    star = dictionary.get('*') if stars is not None else None
    ids = [[dictionary[c] for c in word if c in dictionary and dictionary[c] not in (blank, star)] for word in words]
    tokens, owner = targets(ids, stars, star)
    out = [None] * len(words)
    if not any(o is not None for o in owner):
        return out
    labels, scores = torchaudio.functional.forced_align(
        emission[None].float().contiguous(), torch.tensor([tokens], dtype=torch.int32), blank=blank)
    spans = torchaudio.functional.merge_tokens(labels[0], scores[0].exp(), blank=blank)
    if len(spans) != len(tokens):
        raise RuntimeError(f'aligner returned {len(spans)} spans for {len(tokens)} tokens')
    letters = {}
    for span, index in zip(spans, owner):
        if index is not None:
            letters.setdefault(index, []).append(span)
    previous = 0
    frame = HOP / ALIGN_RATE
    for index in range(len(words)):
        if index not in letters:
            continue
        mine = letters[index]
        frames = sum(s.end - s.start for s in mine)
        start = mine[0].start
        out[index] = dict(start=round(start * frame, 3), end=round(mine[-1].end * frame, 3),
                          score=round(sum(s.score * (s.end - s.start) for s in mine) / max(frames, 1), 3),
                          quiet_before=round(longest_run(quiet, previous, start + 2) * frame, 3))
        previous = start
    return out


def run(mix, words, stars=None):
    device, began, timing = pick_device(), time.monotonic(), {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    stem = vocal_stem(mix, device)
    timing['separate_s'] = round(time.monotonic() - began, 1)
    began = time.monotonic()
    stem = torchaudio.functional.resample(stem.to(device), MIX_RATE, ALIGN_RATE).cpu()
    emission, dictionary = emissions(stem, device, star=stars is not None)
    result = place(emission, dictionary, words, quiet_frames(stem), stars)
    timing['align_s'] = round(time.monotonic() - began, 1)
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == 'cuda' else None
    return dict(words=result, seconds=round(mix.shape[-1] / MIX_RATE, 2), device=device.type,
                peak_gib=peak, timing=timing, frames=int(emission.shape[0]), stars=stars is not None)


def selftest():
    """The whole path on a synthetic signal, on whatever device exists (the image build runs it)."""
    torch.manual_seed(0)
    t = torch.arange(int(6 * MIX_RATE)) / MIX_RATE
    tone = 0.2 * torch.sin(2 * math.pi * 220 * t) * (torch.sin(2 * math.pi * 0.5 * t) > 0)
    mix = torch.stack([tone, 0.8 * tone]) + 0.01 * torch.randn(2, t.numel())
    result = run(mix, ['la', '', "don't", 'la'], stars=[True, False, True, False])
    words = result['words']
    assert len(words) == 4 and words[1] is None and result['stars'], words
    for word in (words[0], words[2], words[3]):
        assert 0 <= word['start'] <= word['end'] <= 6.1 and 0 <= word['score'] <= 1 and word['quiet_before'] >= 0, word
    assert words[0]['start'] <= words[2]['start'] <= words[3]['start'], words
    assert abs(result['frames'] - 300) <= 2, result['frames']
    # The same letters without the star token: the round-1 path, from the same emission's letters.
    emission, dictionary = emissions(torch.zeros(6 * ALIGN_RATE), torch.device('cpu'), star=True)
    plain = {k: v for k, v in dictionary.items() if k != '*'}
    assert dictionary['*'] == len(plain) and float(emission[:, -1].abs().max()) == 0.0, dictionary
    flat = place(emission[:, :-1], plain, ['la', '', 'la'], [False] * 300)
    assert flat[1] is None and flat[0]['start'] <= flat[2]['start'], flat
    print('align selftest ok', json.dumps({k: v for k, v in result.items() if k != 'words'}), flush=True)


def main():
    if sys.argv[1:] == ['--selftest']:
        selftest()
        return
    audio, words_path, out_path = sys.argv[1:4]
    try:
        with open(words_path, encoding='utf-8') as stream:
            spec = json.load(stream)
        stars = spec.get('star_after')
        if stars is not None and len(stars) != len(spec['words']):
            raise ValueError('star_after must have one entry per word')
        result = run(read_mix(audio), spec['words'], stars)
    except Exception as error:  # the worker reads the reason; a failed timing never fails a take
        with open(out_path, 'w', encoding='utf-8') as stream:
            json.dump(dict(error=f'{type(error).__name__}: {str(error)[:300]}'), stream)
        raise SystemExit(2)
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(result, stream)


if __name__ == '__main__':
    main()
