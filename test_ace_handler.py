"""Unit tests for ace_handler.py: CPU only, no torch, no network, no ffmpeg.

Run (isolated Python, from this folder):   python -I -m unittest discover -s . -p "test_ace_handler.py"
(python -I -m unittest test_ace_handler.py cannot find the module in isolated mode; the discover form can.)

A fake pipeline stands in for upstream's generate_music, a fake client for B2 and fake functions for ffmpeg and the
WAV writer. TestPinnedUpstream additionally reads the pinned upstream source (if a clone is on this machine) and
checks that every name this worker sends to it exists there; it is skipped when no clone is found."""
import ast
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ace_handler as ace  # noqa: E402

KEY_MP3 = re.compile(r'^ace/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/master\.mp3$')
KEY_WAV = re.compile(r'^ace/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/master\.wav$')
SECRET_CAPTION = 'ZEBRA-CAPTION-SENTINEL for the log check'
SECRET_LYRICS = 'KANGAROO-LYRICS-SENTINEL for the log check'


ACE_REAL_LINK = ace._link


def reset_state():
    ace.STATE.update(ckpt=None, built=set(), dit={}, llm=None, planner_tries=0, api=None, env=False, dirty=set(),
                     styles={})
    ace.RESOLVED.clear()


def silently(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def valid(**changes):
    request = {'caption': 'Warm acoustic folk, brushed drums', 'lyrics': '[Verse]\nword one word two', 'duration': 90}
    request.update(changes)
    return request


# ---------------------------------------------------------------- fakes

class FakeTensor:
    def __init__(self, seconds, channels=2, rate=48000):
        self.shape = (channels, int(seconds * rate))


class FakeS3:
    def __init__(self, fail=False):
        self.fail = fail
        self.uploads = []
        self.presigned = []
        self.objects = {}          # key -> bytes, what download_file can fetch
        self.downloads = []        # (bucket, key) in the order they were asked for
        self.broken = set()        # keys whose download raises half way (after writing some bytes)

    def download_file(self, bucket, key, filename):
        """boto3's S3.Client.download_file(Bucket, Key, Filename), positionally."""
        self.downloads.append((bucket, key))
        if key not in self.objects:
            raise OSError('404 not found')          # botocore raises ClientError; any exception is handled the same
        data = self.objects[key]
        if key in self.broken:
            Path(filename).write_bytes(data[:len(data) // 2])
            raise ConnectionError('connection reset')
        Path(filename).write_bytes(data)

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        if self.fail:
            raise OSError('upload refused')
        self.uploads.append({'path': path, 'bucket': bucket, 'key': key, 'extra': ExtraArgs,
                             'existed': Path(path).exists(), 'size': Path(path).stat().st_size})

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):
        self.presigned.append({'operation': operation, 'params': Params, 'expires': ExpiresIn})
        return f'https://storage.example/{Params["Key"]}?ttl={ExpiresIn}'


class Record:
    """Accepts any keyword fields, like the upstream dataclasses, and remembers them."""
    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeApi:
    GenerationParams = Record
    GenerationConfig = Record

    def __init__(self, takes=None, seconds=90, success=True, error=None, metadata=None, costs=None, raises=None,
                 spend=None):
        self.takes, self.seconds, self.success, self.error = takes, seconds, success, error
        self.metadata, self.costs, self.raises, self.spend = metadata, costs, raises, spend
        self.calls = []

    def generate_music(self, dit, llm, params, config, save_dir=None, progress=None):
        self.calls.append(SimpleNamespace(dit=dit, llm=llm, params=params, config=config, save_dir=save_dir))
        if progress is not None:
            progress(0.1, 'Phase 1: Generating CoT metadata (once for all items)...')
            progress(0.51, desc='Preparing inputs...')
        if self.spend is not None:
            self.spend()
        if self.raises is not None:
            raise self.raises
        if not self.success:
            return SimpleNamespace(success=False, audios=[], extra_outputs={}, error=self.error)
        count = self.takes if self.takes is not None else config.batch_size
        audios = [{'tensor': FakeTensor(self.seconds), 'sample_rate': 48000, 'path': '',
                   'params': {'seed': config.seeds[i]}} for i in range(count)]
        extra = {'lm_metadata': self.metadata, 'time_costs': self.costs if self.costs is not None else {}}
        return SimpleNamespace(success=True, audios=audios, extra_outputs=extra, error=None)


def fake_ffmpeg_factory(log):
    def fake_ffmpeg(*args):
        log.append(tuple(args))
        out = Path(args[-1])
        out.write_bytes(b'M' * 10 if out.suffix == '.mp3' else b'W' * 100)
    return fake_ffmpeg


def fake_write_wav(path, tensor, rate):
    Path(path).write_bytes(b'F' * 400)


class Harness(unittest.TestCase):
    """handler() end to end with every heavy part replaced."""

    def patch(self, obj, name, value):
        patcher = mock.patch.object(obj, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        # nothing here may read this machine's model cache, environment pins or real runpod library
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ('ACE_TURBO_DIR', 'ACE_SFT_DIR', 'ACE_MAIN_DIR', 'ACE_PLANNER_REWRITES_CAPTION'):
            os.environ.pop(name, None)
        nowhere = os.path.join(tempfile.gettempdir(), 'ace-test-no-such-cache')
        os.environ['HF_HUB_CACHE'] = nowhere
        self.patch(ace, 'CACHE_ROOT', nowhere)
        self.patch(ace, 'DOWNLOAD_CACHE', nowhere)
        self.progress_log = []
        fake_runpod = SimpleNamespace(serverless=SimpleNamespace(
            progress_update=lambda job, text: self.progress_log.append((job['id'], text))))
        modules = mock.patch.dict(sys.modules, {'runpod': fake_runpod})
        modules.start()
        self.addCleanup(modules.stop)
        self.s3 = FakeS3()
        self.api = FakeApi()
        self.ffmpeg_log = []
        self.dits = {}
        self.llm = SimpleNamespace(llm_initialized=True)
        self.planner_ok = True
        self.patch(ace, 'storage', lambda: (self.s3, 'sound-bucket'))
        self.engine = self.patch(ace, 'engine', mock.Mock(side_effect=lambda model: self.dits.setdefault(
            model, SimpleNamespace(model=object(), name=model))))
        self.planner = self.patch(ace, 'planner', mock.Mock(side_effect=lambda: self.llm if self.planner_ok else None))
        self.patch(ace, 'upstream', lambda: self.api)
        self.patch(ace, 'ffmpeg', fake_ffmpeg_factory(self.ffmpeg_log))
        self.patch(ace, 'write_float_wav', fake_write_wav)
        self.patch(ace, 'gpu_name', lambda: 'Fake A40')
        self.patch(ace, 'gpu_peak', lambda reset=False: None if reset else 12.5)

    def run_job(self, **changes):
        return silently(ace.handler, {'id': 'job-1', 'input': valid(**changes)})


# ---------------------------------------------------------------- pure helpers

class TestCaption(unittest.TestCase):
    def test_short_caption_is_kept_and_tidied(self):
        self.assertEqual(ace.trim_caption('  warm   lo-fi\n beats  '), ('warm lo-fi beats', None))

    def test_exactly_at_the_limit_is_untouched(self):
        text = ('a' * 9 + ' ') * 51 + 'b' * 2     # 512 characters
        self.assertEqual(len(text), 512)
        self.assertEqual(ace.trim_caption(text), (text, None))

    def test_long_caption_is_cut_at_a_word_boundary_with_a_note(self):
        words = [f'word{i}' for i in range(300)]
        trimmed, note = ace.trim_caption(' '.join(words))
        self.assertLessEqual(len(trimmed), 512)
        self.assertEqual(note, 'The music description was shortened to fit 512 characters.')
        self.assertEqual(trimmed.split(), words[:len(trimmed.split())])     # whole words only
        self.assertGreater(len(trimmed), 480)

    def test_a_limit_that_falls_on_a_space_keeps_the_whole_last_word(self):
        text = 'a' * 100 + ' ' + 'b' * 411 + ' ' + 'c' * 50        # the space is character 513
        self.assertEqual(ace.trim_caption(text)[0], 'a' * 100 + ' ' + 'b' * 411)

    def test_a_space_just_before_the_limit_drops_the_word_that_does_not_fit(self):
        self.assertEqual(ace.trim_caption('x' * 511 + ' tail')[0], 'x' * 511)

    def test_trailing_punctuation_is_dropped_after_a_cut(self):
        text = 'a' * 500 + ', ' + 'b' * 40
        trimmed, _ = ace.trim_caption(text)
        self.assertEqual(trimmed, 'a' * 500)

    def test_a_single_huge_word_is_cut_hard(self):
        trimmed, note = ace.trim_caption('w' * 900)
        self.assertEqual(len(trimmed), 512)
        self.assertTrue(note)


class TestKeyscale(unittest.TestCase):
    def test_forms_upstream_accepts(self):
        for text, want in (('A minor', 'A minor'), ('A Minor', 'A minor'), ('a MINOR', 'A minor'),
                           ('F# minor', 'F# minor'), ('f#min', 'F# minor'), ('Bb major', 'Bb major'),
                           ('bb maj', 'Bb major'), ('Am', 'A minor'), ('Bbm', 'Bb minor'), ('C# Major', 'C# major'),
                           ('F\u266f minor', 'F# minor'), ('B\u266d major', 'Bb major'), ('  G   major ', 'G major')):
            self.assertEqual(ace.normalize_keyscale(text), want, text)

    def test_not_a_key(self):
        for text in ('', 'minor', 'H major', 'A', 'Bb', 'sunny', 'A lydian', 'A minor chord', '12'):
            self.assertIsNone(ace.normalize_keyscale(text), text)


class TestSeedsAndSettings(unittest.TestCase):
    def test_seed_plus_index(self):
        self.assertEqual(ace.take_seeds(10, 3), [10, 11, 12])
        self.assertEqual(ace.take_seeds(0, 1), [0])

    def test_seeds_wrap_inside_the_range(self):
        self.assertEqual(ace.take_seeds(2147483647, 3), [2147483647, 0, 1])

    def test_no_seed_is_random_and_in_range(self):
        drawn = {ace.take_seeds(None, 1)[0] for _ in range(20)}
        self.assertGreater(len(drawn), 1)
        self.assertTrue(all(0 <= s < 2 ** 31 for s in drawn))
        seeds = ace.take_seeds(None, 4)
        self.assertEqual(seeds, [(seeds[0] + i) % 2 ** 31 for i in range(4)])

    def request(self, **changes):
        return ace.request_input(valid(**changes))

    def test_turbo_defaults_and_note(self):
        settings, notes = ace.settings_for(self.request())
        self.assertEqual(settings, {'steps': 8, 'guidance_scale': 1.0, 'shift': 3.0, 'dcw_enabled': True})
        self.assertEqual(notes, ['Made with the fast model in 8 steps.'])

    def test_sft_defaults_turn_dcw_off_explicitly(self):
        settings, notes = ace.settings_for(self.request(model='xl-sft'))
        self.assertEqual(settings, {'steps': 50, 'guidance_scale': 7.0, 'shift': 3.0, 'dcw_enabled': False})
        self.assertIs(settings['dcw_enabled'], False)
        self.assertEqual(notes, ['Made with the best-quality model in 50 steps at guidance 7.'])

    def test_overrides(self):
        settings, notes = ace.settings_for(self.request(model='xl-sft', steps=30, guidance_scale=5.5, shift=2))
        self.assertEqual((settings['steps'], settings['guidance_scale'], settings['shift']), (30, 5.5, 2.0))
        self.assertEqual(notes, ['Made with the best-quality model in 30 steps at guidance 5.5.'])
        settings, notes = ace.settings_for(self.request(steps=4, shift=1.5))
        self.assertEqual((settings['steps'], settings['shift']), (4, 1.5))

    def test_turbo_ignores_guidance_and_says_so(self):
        settings, notes = ace.settings_for(self.request(guidance_scale=9))
        self.assertEqual(settings['guidance_scale'], 1.0)
        self.assertEqual(notes, ['The fast model ignores the guidance setting.', 'Made with the fast model in 8 steps.'])
        _, notes = ace.settings_for(self.request(guidance_scale=1))
        self.assertEqual(notes, ['Made with the fast model in 8 steps.'])

    def test_guidance_13_on_sft_is_used_whole_and_on_turbo_is_ignored(self):
        settings, notes = ace.settings_for(self.request(model='xl-sft', guidance_scale=13))
        self.assertEqual(settings['guidance_scale'], 13.0)
        self.assertEqual(notes, ['Made with the best-quality model in 50 steps at guidance 13.'])
        settings, notes = ace.settings_for(self.request(model='xl-turbo', guidance_scale=13))
        self.assertEqual(settings['guidance_scale'], 1.0)
        self.assertEqual(notes, ['The fast model ignores the guidance setting.', 'Made with the fast model in 8 steps.'])

    def test_planner_temperature_needs_a_value_and_a_planner(self):
        asked = self.request(lm_temperature=1.3)
        self.assertEqual(ace.planner_temperature(asked, True), 1.3)
        self.assertIsNone(ace.planner_temperature(asked, False))
        self.assertIsNone(ace.planner_temperature(self.request(), True))
        self.assertEqual(ace.planner_temperature(self.request(lm_temperature=0), True), 0.0)    # zero is a value
        self.assertIsNotNone(ace.planner_temperature(self.request(lm_temperature=0), True))


class TestSmallHelpers(unittest.TestCase):
    def test_join_names(self):
        self.assertEqual(ace.join_names(['tempo']), 'tempo')
        self.assertEqual(ace.join_names(['tempo', 'key']), 'tempo and key')
        self.assertEqual(ace.join_names(['tempo', 'key', 'time signature']), 'tempo, key and time signature')

    def test_scrub_hides_the_callers_words_and_keeps_one_short_line(self):
        text = f'Failed on {SECRET_CAPTION}\nTraceback with {SECRET_LYRICS}'
        shown = ace.scrub(text, SECRET_CAPTION, SECRET_LYRICS)
        self.assertNotIn('ZEBRA', shown)
        self.assertNotIn('KANGAROO', shown)
        self.assertNotIn('Traceback', shown)
        self.assertLessEqual(len(ace.scrub('x' * 900)), 200)
        self.assertLessEqual(len(ace.scrub('x' * 900, limit=300)), 300)
        self.assertGreater(len(ace.scrub('x' * 900, limit=300)), 200)
        self.assertEqual(ace.scrub(None), '')
        self.assertEqual(ace.scrub('   '), '')

    def test_scrub_removes_web_links_and_hub_tokens_but_keeps_the_reason(self):
        text = 'Cannot load: https://huggingface.co/a/b?x=1 and http://10.0.0.1:8080/p, token hf_QwErTyUiOp12 end'
        self.assertEqual(ace.scrub(text), 'Cannot load: [link] and [link], token [token] end')
        self.assertEqual(ace.scrub('Out of memory while loading'), 'Out of memory while loading')
        self.assertEqual(ace.scrub('shelf_life_value and chf_x'), 'shelf_life_value and chf_x')   # not a token

    def test_plain_failure(self):
        self.assertEqual(ace.plain_failure('Insufficient free VRAM: need ~9.1 GB'), ace.TOO_BIG)
        self.assertEqual(ace.plain_failure('CUDA out of memory. Tried to allocate 2 GiB'), ace.TOO_BIG)
        self.assertEqual(ace.plain_failure('some other thing'), ace.FAILED)
        self.assertEqual(ace.plain_failure(None), ace.FAILED)

    def test_planned_only_reports_what_the_caller_left_open(self):
        meta = {'bpm': '96', 'keyscale': 'a minor', 'timesignature': '4/4', 'caption': 'private words',
                'duration': 100}
        open_request = ace.request_input(valid())
        self.assertEqual(ace.planned(meta, open_request), {'bpm': 96, 'keyscale': 'A minor', 'timesignature': '4'})
        given = ace.request_input(valid(bpm=120, keyscale='C major', timesignature=3))
        self.assertIsNone(ace.planned(meta, given))
        half = ace.request_input(valid(bpm=120))
        self.assertEqual(ace.planned(meta, half), {'keyscale': 'A minor', 'timesignature': '4'})

    def test_planned_ignores_nonsense(self):
        request = ace.request_input(valid())
        self.assertIsNone(ace.planned(None, request))
        self.assertIsNone(ace.planned({'bpm': 'N/A', 'keyscale': 'N/A', 'timesignature': 'N/A'}, request))
        self.assertIsNone(ace.planned({'bpm': 9000, 'keyscale': 5, 'timesignature': '7'}, request))
        self.assertIsNone(ace.planned({'bpm': 'fast'}, request))


# ---------------------------------------------------------------- request validation

class TestRequestInput(unittest.TestCase):
    def check(self, sentence, **changes):
        with self.assertRaises(ace.RequestError) as caught:
            ace.request_input(valid(**changes))
        self.assertEqual(str(caught.exception), sentence)

    def test_defaults(self):
        request = ace.request_input({'caption': 'jazz trio', 'duration': 120})
        self.assertEqual((request['mode'], request['model']), ('render', 'xl-turbo'))
        self.assertEqual((request['batch_size'], request['vocal_language'], request['thinking']), (1, 'unknown', True))
        self.assertIsNone(request['seed'])
        self.assertIsNone(request['bpm'])
        self.assertEqual((request['keyscale'], request['timesignature']), ('', ''))
        self.assertEqual(request['lyrics'], '[Instrumental]')
        self.assertTrue(request['instrumental'])
        self.assertEqual(request['duration'], 120.0)

    def test_every_field_set(self):
        request = ace.request_input({
            'mode': 'render', 'model': 'xl-sft', 'caption': ' piano ballad ', 'lyrics': '[Verse]\r\nsome words\r\n',
            'duration': 200.5, 'seed': 0, 'batch_size': 4, 'bpm': 72.0, 'keyscale': 'bb Major',
            'timesignature': '3/4', 'vocal_language': 'ES', 'thinking': False, 'steps': 40,
            'guidance_scale': 8, 'shift': 2.5})
        self.assertEqual(request['model'], 'xl-sft')
        self.assertEqual(request['caption'], 'piano ballad')
        self.assertEqual(request['lyrics'], '[Verse]\nsome words')
        self.assertFalse(request['instrumental'])
        self.assertEqual((request['duration'], request['seed'], request['batch_size']), (200.5, 0, 4))
        self.assertEqual((request['bpm'], request['keyscale'], request['timesignature']), (72, 'Bb major', '3'))
        self.assertEqual((request['vocal_language'], request['thinking']), ('es', False))
        self.assertEqual((request['steps'], request['guidance_scale'], request['shift']), (40, 8, 2.5))

    def test_null_means_absent(self):
        request = ace.request_input({'caption': 'x', 'duration': 30, 'seed': None, 'batch_size': None, 'bpm': None,
                                     'keyscale': None, 'timesignature': None, 'vocal_language': None,
                                     'thinking': None, 'steps': None, 'guidance_scale': None, 'shift': None,
                                     'lm_temperature': None, 'lyrics': None, 'model': None, 'mode': None})
        self.assertEqual((request['model'], request['batch_size'], request['thinking']), ('xl-turbo', 1, True))
        self.assertIsNone(request['lm_temperature'])

    def test_mode_and_model(self):
        self.check('Mode must be render or probe.', mode='fast')
        self.check('Model must be xl-turbo or xl-sft.', model='xl-base')
        self.check('Model must be xl-turbo or xl-sft.', model='')
        self.check('Model must be xl-turbo or xl-sft.', model=3)
        for value in (['xl-turbo'], {'name': 'xl-turbo'}, [], {}, True):      # unhashable values must not reach a dict lookup
            self.check('Model must be xl-turbo or xl-sft.', model=value)

    def test_odd_json_values_in_any_field_are_refused_in_plain_words_never_crash(self):
        fields = ('mode', 'model', 'caption', 'lyrics', 'duration', 'seed', 'batch_size', 'bpm', 'keyscale',
                  'timesignature', 'vocal_language', 'thinking', 'steps', 'guidance_scale', 'shift', 'lm_temperature',
                  'lora_key', 'lora_scale')
        values = ([], {}, ['xl-turbo'], {'a': 1}, [[1]], [None], True, False, 0, -1, 1.5, '', 'x', float('nan'),
                  float('inf'), 10 ** 30, -10 ** 30, 10 ** 400, -10 ** 400, ' ', '\x00', 2 ** 31)
        for field in fields:
            for value in values:
                for extra in ({}, {'model': 'xl-sft', 'lora_key': 'ace-loras/soul-real-v1'}):     # with a style, too
                    try:
                        ace.request_input(valid(**{**extra, field: value}))
                    except ace.RequestError:
                        pass
                    except Exception as error:      # any other type would reach the generic handler as a vague failure
                        self.fail(f'{field}={value!r} raised {type(error).__name__}')

    def test_caption(self):
        self.check('Describe the music in a few words.', caption='')
        self.check('Describe the music in a few words.', caption='   \n ')
        self.check('Describe the music in a few words.', caption=None)
        self.check('The music description must be text.', caption=12)
        self.check('The music description must be text.', caption=['a'])

    def test_long_caption_is_trimmed_not_refused(self):
        request = ace.request_input(valid(caption='word ' * 400))
        self.assertLessEqual(len(request['caption']), 512)
        self.assertEqual(request['notes'], ['The music description was shortened to fit 512 characters.'])

    def test_lyrics(self):
        self.check('Lyrics must be text.', lyrics=5)
        self.check('Lyrics must be text.', lyrics=['a'])
        self.check('Lyrics must be 4096 characters or fewer.', lyrics='a' * 4097)
        self.assertEqual(ace.request_input(valid(lyrics='a' * 4096))['lyrics'], 'a' * 4096)

    def test_instrumental_forms(self):
        for lyrics in ('', '   ', None, '[Instrumental]', '[instrumental]', ' [INSTRUMENTAL]\n'):
            request = ace.request_input(valid(lyrics=lyrics))
            self.assertEqual(request['lyrics'], '[Instrumental]', repr(lyrics))
            self.assertTrue(request['instrumental'])

    def test_section_tags_and_backing_vocals_pass_through(self):
        lyrics = '[Intro]\n\n[Verse 1]\nline one (oh oh)\n\n[Chorus]\nline two (hey)\n[Outro]'
        request = ace.request_input(valid(lyrics=lyrics))
        self.assertEqual(request['lyrics'], lyrics)
        self.assertFalse(request['instrumental'])

    def test_duration(self):
        sentence = 'Length must be between 10 and 600 seconds.'
        for value in (None, 9, 9.99, 601, 0, -5, '120', True, False, float('nan'), float('inf'), [30]):
            self.check(sentence, duration=value)
        with self.assertRaises(ace.RequestError) as caught:       # a length is required
            ace.request_input({'caption': 'x'})
        self.assertEqual(str(caught.exception), sentence)
        for value in (10, 600, 30.5):
            self.assertEqual(ace.request_input(valid(duration=value))['duration'], float(value))

    def test_seed(self):
        sentence = 'Seed must be a whole number from 0 to 2147483647.'
        for value in (-1, 2147483648, 1.5, '7', True, float('nan'), [1]):
            self.check(sentence, seed=value)
        for value, want in ((0, 0), (2147483647, 2147483647), (12.0, 12)):
            self.assertEqual(ace.request_input(valid(seed=value))['seed'], want)

    def test_batch_size(self):
        sentence = 'Batch size must be a whole number from 1 to 4.'
        for value in (0, 5, -1, 1.5, '2', True):
            self.check(sentence, batch_size=value)
        self.assertEqual(ace.request_input(valid(batch_size=4))['batch_size'], 4)

    def test_bpm(self):
        sentence = 'Tempo must be a whole number from 30 to 300 BPM.'
        for value in (29, 301, 0, 88.5, '88', True):
            self.check(sentence, bpm=value)
        self.assertEqual(ace.request_input(valid(bpm=30))['bpm'], 30)
        self.assertEqual(ace.request_input(valid(bpm=300))['bpm'], 300)

    def test_keyscale(self):
        self.check('Key must be text, such as A minor.', keyscale=5)
        request = ace.request_input(valid(keyscale='sunny'))
        self.assertEqual(request['keyscale'], '')
        self.assertEqual(request['notes'], ['The key was not understood, so it was left out.'])
        self.assertEqual(ace.request_input(valid(keyscale='a minor'))['keyscale'], 'A minor')

    def test_time_signature(self):
        sentence = 'Time signature must be 2, 3, 4 or 6.'
        for value in (5, '5', 7, 'waltz', 4.5, True, [4]):
            self.check(sentence, timesignature=value)
        for value, want in ((2, '2'), ('3', '3'), (4, '4'), ('6', '6'), ('4/4', '4'), ('6/8', '6'), (4.0, '4')):
            self.assertEqual(ace.request_input(valid(timesignature=value))['timesignature'], want, value)

    def test_vocal_language(self):
        sentence = 'Vocal language must be a short code such as en.'
        for value in ('english', 'xx', 5, ['en']):
            self.check(sentence, vocal_language=value)
        for value, want in (('en', 'en'), ('FR', 'fr'), ('yue', 'yue'), ('unknown', 'unknown'), (' ja ', 'ja')):
            self.assertEqual(ace.request_input(valid(vocal_language=value))['vocal_language'], want)

    def test_thinking(self):
        for value in ('yes', 1, 0, 'false'):
            self.check('Thinking must be true or false.', thinking=value)
        self.assertFalse(ace.request_input(valid(thinking=False))['thinking'])

    def test_advanced_ranges_follow_the_model(self):
        self.check('Steps must be a whole number from 1 to 8.', steps=9)
        self.check('Steps must be a whole number from 1 to 8.', steps=0)
        self.check('Steps must be a whole number from 4 to 100.', model='xl-sft', steps=101)
        self.check('Steps must be a whole number from 4 to 100.', model='xl-sft', steps=3)
        self.check('Steps must be a whole number from 1 to 8.', steps=4.5)
        self.check('Steps must be a whole number from 1 to 8.', steps='8')
        self.check('Guidance must be a number from 1 to 15.', guidance_scale=0.5)
        self.check('Guidance must be a number from 1 to 15.', guidance_scale=16)
        self.check('Guidance must be a number from 1 to 15.', guidance_scale='7')
        self.check('Shift must be a number from 1 to 5.', shift=0)
        self.check('Shift must be a number from 1 to 5.', shift=5.1)
        self.check('Shift must be a number from 1 to 5.', shift=True)
        self.assertEqual(ace.request_input(valid(steps=8))['steps'], 8)
        self.assertEqual(ace.request_input(valid(model='xl-sft', steps=100))['steps'], 100)

    def test_guidance_up_to_13_is_accepted_for_both_models(self):
        # the booth's Prompt guidance dial sends 7 (dial 1), 10 (dial 2) and 13 (dial 3) for xl-sft
        for model in ('xl-turbo', 'xl-sft'):
            for value in (1, 1.0, 7, 7.0, 8.5, 10, 13, 13.0, 15):
                got = ace.request_input(valid(model=model, guidance_scale=value))['guidance_scale']
                self.assertEqual(got, value, (model, value))
            self.check('Guidance must be a number from 1 to 15.', model=model, guidance_scale=15.1)

    def test_lm_temperature_is_optional(self):
        self.assertIsNone(ace.request_input(valid())['lm_temperature'])
        self.assertIsNone(ace.request_input(valid(lm_temperature=None))['lm_temperature'])

    def test_lm_temperature_accepts_the_whole_range_as_a_float(self):
        for value in (0.0, 0, 0.4, 0.85, 1, 1.3, 2, 2.0):
            for model in ('xl-turbo', 'xl-sft'):
                got = ace.request_input(valid(model=model, lm_temperature=value))['lm_temperature']
                self.assertEqual(got, value, (model, value))
                self.assertIs(type(got), float, (model, value))      # 0 and 1 arrive as 0.0 and 1.0, never as bool

    def test_lm_temperature_refusals_are_one_plain_sentence(self):
        sentence = 'Creative variation must be a number from 0 to 2.'
        for value in (-0.01, -1, 2.01, 2.5, 3, 100, 10 ** 30, 10 ** 400, -10 ** 400, float('nan'), float('inf'),
                      float('-inf'), True, False, '1', '0.85', '', ' ', [0.85], {'t': 0.85}, [], {}):
            self.check(sentence, lm_temperature=value)
        self.check(sentence, model='xl-sft', lm_temperature=2.01)

    def test_huge_whole_numbers_get_the_plain_sentence_not_an_overflow(self):
        self.assertEqual(ace._real(10 ** 400), 10 ** 400)           # math.isfinite would raise OverflowError on it
        self.check('Length must be between 10 and 600 seconds.', duration=10 ** 400)
        self.check('Guidance must be a number from 1 to 15.', guidance_scale=10 ** 400)
        self.check('Shift must be a number from 1 to 5.', shift=-10 ** 400)
        self.assertIsNone(ace._real(True))
        self.assertIsNone(ace._real(float('nan')))
        self.assertIsNone(ace._real(float('inf')))
        self.assertIsNone(ace._real('1'))

    def test_unknown_fields_are_ignored(self):
        request = ace.request_input(valid(unexpected='value', engine='yue2'))
        self.assertNotIn('unexpected', request)

    def test_validation_order_is_stable(self):
        with self.assertRaises(ace.RequestError) as caught:
            ace.request_input({'mode': 'nope', 'model': 'nope', 'caption': ''})
        self.assertEqual(str(caught.exception), 'Mode must be render or probe.')


# ---------------------------------------------------------------- the handler end to end

class TestRenderOutput(Harness):
    def test_single_take_shape(self):
        out = self.run_job(seed=7)
        self.assertNotIn('error', out)
        self.assertEqual((out['engine'], out['model'], out['mode']), ('ace', 'xl-turbo', 'render'))
        self.assertRegex(out['key'], KEY_MP3)
        self.assertRegex(out['wav_key'], KEY_WAV)
        self.assertEqual(out['key'].rsplit('/', 1)[0], out['wav_key'].rsplit('/', 1)[0])
        # which link goes with which file: the fake client builds each link from the Key it was asked to sign
        self.assertEqual(out['url'], f'https://storage.example/{out["key"]}?ttl=604800')
        self.assertEqual(out['wav_url'], f'https://storage.example/{out["wav_key"]}?ttl=604800')
        self.assertTrue(out['url'].split('?')[0].endswith('/master.mp3'))
        self.assertTrue(out['wav_url'].split('?')[0].endswith('/master.wav'))
        self.assertEqual(out['duration_s'], 90.0)
        self.assertEqual(out['bytes'], 10)
        self.assertEqual(out['seed'], 7)
        self.assertIsInstance(out['processing_ms'], int)
        self.assertIs(out['truncated'], False)
        self.assertEqual(out['gpu'], 'Fake A40')
        self.assertEqual(out['memory'], {'render_peak_gib': 12.5})
        self.assertEqual(out['features'], ['ace-xl-turbo', 'ace-xl-sft', 'batch', 'instrumental', 'probe', 'lora'])
        self.assertEqual(set(out['versions']), {'ace_step_commit', 'torch', 'worker'})
        self.assertEqual(out['versions']['ace_step_commit'], ace.UPSTREAM_COMMIT)
        self.assertTrue(all(isinstance(note, str) and note.endswith('.') for note in out['worker_notes']))
        self.assertNotIn('plan', out)

    def test_take_zero_is_duplicated_at_the_top_level(self):
        out = self.run_job(seed=100, batch_size=3)
        self.assertEqual(len(out['takes']), 3)
        first = out['takes'][0]
        self.assertEqual(set(first), {'index', 'seed', 'key', 'wav_key', 'url', 'wav_url', 'duration_s', 'bytes'})
        for name in ('key', 'wav_key', 'url', 'wav_url', 'duration_s', 'bytes', 'seed'):
            self.assertEqual(out[name], first[name], name)
        self.assertEqual([t['index'] for t in out['takes']], [0, 1, 2])
        self.assertEqual([t['seed'] for t in out['takes']], [100, 101, 102])
        prefixes = {t['key'].rsplit('/', 1)[0] for t in out['takes']}
        self.assertEqual(len(prefixes), 3)
        for take in out['takes']:
            self.assertRegex(take['key'], KEY_MP3)
            self.assertRegex(take['wav_key'], KEY_WAV)
            self.assertEqual(take['url'], f'https://storage.example/{take["key"]}?ttl=604800')        # mp3 link, mp3 file
            self.assertEqual(take['wav_url'], f'https://storage.example/{take["wav_key"]}?ttl=604800')  # wav link, wav file
            self.assertEqual(take['key'].rsplit('/', 1)[0], take['wav_key'].rsplit('/', 1)[0])

    def test_timing_keys(self):
        out = self.run_job()
        self.assertEqual(set(out['timing']), {'model_load_s', 'render_s', 'encode_s', 'upload_s'})
        for value in out['timing'].values():
            self.assertIsInstance(value, float)
        self.assertEqual(out['controls'], {'steps': 8, 'guidance_scale': 1.0, 'shift': 3.0})

    def test_uploads_and_links(self):
        out = self.run_job(batch_size=2)
        self.assertEqual(len(self.s3.uploads), 4)
        for upload in self.s3.uploads:
            self.assertEqual(upload['bucket'], 'sound-bucket')
            self.assertTrue(upload['existed'])
            self.assertEqual(upload['extra'], {'ContentType': 'audio/mpeg' if upload['key'].endswith('.mp3')
                                               else 'audio/wav'})
        self.assertEqual(sorted(u['key'] for u in self.s3.uploads),
                         sorted([t['key'] for t in out['takes']] + [t['wav_key'] for t in out['takes']]))
        self.assertEqual(len(self.s3.presigned), 4)
        for link in self.s3.presigned:
            self.assertEqual((link['operation'], link['expires']), ('get_object', 604800))
            self.assertEqual(link['params']['Bucket'], 'sound-bucket')
        # every uploaded file gets exactly one link, signed for that file's key
        self.assertEqual(sorted(link['params']['Key'] for link in self.s3.presigned),
                         sorted(upload['key'] for upload in self.s3.uploads))

    def test_temp_files_are_gone_afterwards(self):
        self.run_job()
        self.assertTrue(self.s3.uploads)
        for upload in self.s3.uploads:
            self.assertFalse(Path(upload['path']).exists())

    def test_audio_is_made_the_way_yue_makes_it(self):
        self.run_job()
        self.assertEqual(len(self.ffmpeg_log), 2)
        first, second = self.ffmpeg_log
        self.assertEqual((first[0], Path(first[1]).name, first[2:4], Path(first[4]).name, len(first)),
                         ('-i', 'audio.wav', ('-c:a', 'pcm_s24le'), 'master.wav', 5))
        self.assertEqual((second[0], Path(second[1]).name, second[2:6], Path(second[6]).name, len(second)),
                         ('-i', 'master.wav', ('-c:a', 'libmp3lame', '-b:a', '320k'), 'master.mp3', 7))

    def test_nothing_is_sent_to_save_dir(self):
        self.run_job()
        self.assertIsNone(self.api.calls[0].save_dir)


class TestTimingAndMemory(Harness):
    """The booth prices a song and sizes cheaper cards from these fields, so each one is pinned to a clock the test runs."""

    def setUp(self):
        super().setUp()
        self.clock = {'now': 1000.0}
        self.patch(ace.time, 'monotonic', lambda: self.clock['now'])

    def tick(self, seconds):
        self.clock['now'] += seconds

    def test_model_load_time_covers_the_model_and_the_planner(self):
        def slow_engine(model):
            self.tick(20.0)
            return self.dits.setdefault(model, SimpleNamespace(model=object(), name=model))

        def slow_planner():
            self.tick(7.0)
            return self.llm
        self.engine.side_effect = slow_engine
        self.planner.side_effect = slow_planner
        self.assertEqual(self.run_job()['timing']['model_load_s'], 27.0)
        self.assertEqual(self.run_job(thinking=False)['timing']['model_load_s'], 20.0)    # no planner, no planner time

    def test_encode_and_upload_times_add_up_over_the_takes(self):
        encode = fake_ffmpeg_factory(self.ffmpeg_log)

        def slow_ffmpeg(*args):
            self.tick(2.0)
            encode(*args)
        self.patch(ace, 'ffmpeg', slow_ffmpeg)
        upload = self.s3.upload_file

        def slow_upload(*args, **kwargs):
            self.tick(3.0)
            return upload(*args, **kwargs)
        self.s3.upload_file = slow_upload
        timing = self.run_job(batch_size=3)['timing']
        self.assertEqual(timing['encode_s'], 12.0)      # 3 takes x 2 ffmpeg runs x 2 s
        self.assertEqual(timing['upload_s'], 18.0)      # 3 takes x 2 files x 3 s

    def test_the_memory_peak_is_reset_after_loading_and_read_right_after_the_render(self):
        events = []

        def peak(reset=False):
            events.append('reset' if reset else 'read')
            return None if reset else 7.25
        self.patch(ace, 'gpu_peak', peak)
        self.engine.side_effect = lambda model: events.append('load') or self.dits.setdefault(
            model, SimpleNamespace(model=object(), name=model))
        self.planner.side_effect = lambda: events.append('planner') or self.llm
        self.api = FakeApi(spend=lambda: events.append('render'))
        out = self.run_job()
        self.assertEqual(events, ['load', 'planner', 'reset', 'render', 'read'])
        self.assertEqual(out['memory'], {'render_peak_gib': 7.25})


class TestGenerationRequest(Harness):
    def last(self):
        return self.api.calls[-1]

    def test_seed_and_batch_reach_upstream_so_it_cannot_reroll_them(self):
        self.run_job(seed=40, batch_size=3)
        call = self.last()
        self.assertEqual(call.config.seeds, [40, 41, 42])
        self.assertEqual(call.config.batch_size, 3)
        self.assertIs(call.config.use_random_seed, False)
        self.assertEqual(call.params.seed, 40)

    def test_without_a_seed_one_is_drawn_and_reported(self):
        out = self.run_job(batch_size=2)
        call = self.last()
        self.assertIs(call.config.use_random_seed, False)
        self.assertEqual(call.config.seeds[0], out['seed'])
        self.assertEqual([t['seed'] for t in out['takes']], call.config.seeds)

    def test_global_generators_are_seeded_for_the_planner(self):
        with mock.patch.object(ace, 'seed_everything') as seeded:
            self.run_job(seed=33, batch_size=2)
        seeded.assert_called_once_with(33)

    def test_metadata_goes_in_fields_not_in_the_caption(self):
        self.run_job(caption='Slow soul, 72 BPM, warm', bpm=72, keyscale='a Minor', timesignature='3',
                     vocal_language='es', duration=151)
        params = self.last().params
        self.assertEqual(params.caption, 'Slow soul, 72 BPM, warm')     # the caller's words, untouched
        self.assertEqual((params.bpm, params.keyscale, params.timesignature), (72, 'A minor', '3'))
        self.assertEqual((params.vocal_language, params.duration, params.task_type), ('es', 151.0, 'text2music'))

    def test_unset_metadata_is_left_for_the_planner(self):
        self.run_job()
        params = self.last().params
        self.assertEqual((params.bpm, params.keyscale, params.timesignature), (None, '', ''))
        self.assertEqual(params.vocal_language, 'unknown')

    def test_long_caption_is_trimmed_and_noted(self):
        out = self.run_job(caption='ambient texture ' * 80)
        self.assertLessEqual(len(self.last().params.caption), 512)
        self.assertIn('The music description was shortened to fit 512 characters.', out['worker_notes'])

    def test_instrumental_sends_the_literal_tag(self):
        out = self.run_job(lyrics='')
        params = self.last().params
        self.assertEqual(params.lyrics, '[Instrumental]')
        self.assertIs(params.instrumental, True)
        self.assertIs(out['instrumental'], True)
        out = self.run_job(lyrics='[Verse]\nnice and plain')
        self.assertEqual(self.last().params.lyrics, '[Verse]\nnice and plain')
        self.assertIs(self.last().params.instrumental, False)
        self.assertIs(out['instrumental'], False)

    def test_turbo_settings(self):
        self.run_job()
        params = self.last().params
        self.assertEqual((params.inference_steps, params.guidance_scale, params.shift), (8, 1.0, 3.0))
        self.assertIs(params.dcw_enabled, True)
        self.assertIs(self.last().dit, self.dits['xl-turbo'])
        self.engine.assert_called_with('xl-turbo')

    def test_dcw_strengths_follow_the_planner_as_in_upstreams_think_defaults(self):
        self.run_job()                                  # turbo with the planner: the Think pair
        params = self.last().params
        self.assertEqual((params.dcw_scaler, params.dcw_high_scaler), (0.02, 0.06))
        self.run_job(thinking=False)                    # no planner: the plain pair
        params = self.last().params
        self.assertEqual((params.dcw_scaler, params.dcw_high_scaler), (0.05, 0.02))
        self.planner_ok = False                         # planner wanted but not available: made without it
        self.run_job()
        params = self.last().params
        self.assertIs(params.thinking, False)
        self.assertEqual((params.dcw_scaler, params.dcw_high_scaler), (0.05, 0.02))

    def test_dcw_strengths_are_not_sent_when_dcw_is_off(self):
        self.run_job(model='xl-sft')
        params = self.last().params
        self.assertIs(params.dcw_enabled, False)
        self.assertFalse(hasattr(params, 'dcw_scaler'))
        self.assertFalse(hasattr(params, 'dcw_high_scaler'))

    def test_sft_settings_turn_dcw_off(self):
        out = self.run_job(model='xl-sft')
        params = self.last().params
        self.assertEqual((params.inference_steps, params.guidance_scale, params.shift), (50, 7.0, 3.0))
        self.assertIs(params.dcw_enabled, False)
        self.assertIs(self.last().dit, self.dits['xl-sft'])
        self.assertEqual(out['model'], 'xl-sft')
        self.assertEqual(out['controls'], {'steps': 50, 'guidance_scale': 7.0, 'shift': 3.0})

    def test_advanced_overrides_are_used(self):
        self.run_job(model='xl-sft', steps=24, guidance_scale=6, shift=2)
        params = self.last().params
        self.assertEqual((params.inference_steps, params.guidance_scale, params.shift), (24, 6.0, 2.0))

    def test_thinking_on_passes_the_planner_and_keeps_the_callers_words(self):
        self.run_job()
        call = self.last()
        self.assertIs(call.llm, self.llm)
        self.assertIs(call.params.thinking, True)
        self.assertIs(call.params.use_cot_metas, True)
        self.assertIs(call.params.use_cot_caption, False)
        self.assertIs(call.params.use_cot_language, False)

    def test_planner_may_rewrite_the_caption_only_when_switched_on(self):
        with mock.patch.dict(os.environ, {'ACE_PLANNER_REWRITES_CAPTION': '1'}):
            self.run_job()
        self.assertIs(self.last().params.use_cot_caption, True)

    def test_thinking_off_skips_the_planner_completely(self):
        out = self.run_job(thinking=False)
        call = self.last()
        self.assertIsNone(call.llm)
        self.assertIs(call.params.thinking, False)
        # upstream still runs its planner for any use_cot_* flag left on, so all three must be off
        self.assertEqual((call.params.use_cot_metas, call.params.use_cot_caption, call.params.use_cot_language),
                         (False, False, False))
        self.planner.assert_not_called()
        self.assertIn('The planner was skipped.', out['worker_notes'])
        self.assertIs(out['thinking'], False)
        self.assertNotIn('plan_s', out['timing'])

    def test_planner_missing_falls_back_without_it(self):
        self.planner_ok = False
        out = self.run_job()
        call = self.last()
        self.assertIsNone(call.llm)
        self.assertEqual((call.params.thinking, call.params.use_cot_metas), (False, False))
        self.assertIn('The planner was not available, so the song was made without it.', out['worker_notes'])
        self.assertIs(out['thinking'], False)

    def test_progress_reaches_runpod_in_plain_phrases(self):
        self.run_job()
        self.assertEqual(self.progress_log, [('job-1', 'Loading the music model'), ('job-1', 'Planning your song'),
                                             ('job-1', 'Rendering your song')])

    def test_a_loaded_model_is_not_announced_again(self):
        self.dits['xl-turbo'] = SimpleNamespace(model=object(), name='xl-turbo')
        ace.STATE['dit']['xl-turbo'] = self.dits['xl-turbo']
        self.run_job()
        self.assertNotIn('Loading the music model', [text for _job, text in self.progress_log])

    def test_progress_updates_never_break_a_render(self):
        sys.modules['runpod'].serverless.progress_update = mock.Mock(side_effect=ConnectionError('offline'))
        self.assertNotIn('error', self.run_job())


class TestPlanner(Harness):
    def test_plan_timing_and_notes(self):
        self.api = FakeApi(metadata={'bpm': '96', 'keyscale': 'A minor', 'timesignature': '4',
                                     'caption': 'private words', 'duration': 90},
                           costs={'lm_phase1_time': 5.0, 'lm_phase2_time': 40.0, 'lm_total_time': 45.04})
        out = self.run_job()
        self.assertEqual(out['plan'], {'bpm': 96, 'keyscale': 'A minor', 'timesignature': '4'})
        self.assertEqual(out['timing']['plan_s'], 45.0)
        self.assertIn('The planner chose the tempo, key and time signature.', out['worker_notes'])
        self.assertIs(out['thinking'], True)
        self.assertNotIn('private words', json.dumps(out))

    def test_plan_leaves_out_what_the_caller_set(self):
        self.api = FakeApi(metadata={'bpm': 96, 'keyscale': 'A minor', 'timesignature': '4'},
                           costs={'lm_total_time': 12.0})
        out = self.run_job(bpm=100)
        self.assertEqual(out['plan'], {'keyscale': 'A minor', 'timesignature': '4'})
        self.assertIn('The planner chose the key and time signature.', out['worker_notes'])

    def test_planner_that_decided_nothing_omits_the_plan(self):
        self.api = FakeApi(metadata={'bpm': 90, 'keyscale': 'C major', 'timesignature': '4'},
                           costs={'lm_total_time': 8.0})
        out = self.run_job(bpm=90, keyscale='C major', timesignature=4)
        self.assertNotIn('plan', out)
        self.assertIn('The planner shaped the song.', out['worker_notes'])
        self.assertEqual(out['timing']['plan_s'], 8.0)

    def test_render_time_excludes_the_plan(self):
        clock = {'now': 1000.0}
        self.api = FakeApi(costs={'lm_total_time': 30.0}, spend=lambda: clock.update(now=clock['now'] + 100.0))
        with mock.patch.object(ace.time, 'monotonic', lambda: clock['now']):
            out = self.run_job()
        self.assertEqual(out['timing']['plan_s'], 30.0)
        self.assertEqual(out['timing']['render_s'], 70.0)
        self.assertEqual(out['timing']['model_load_s'], 0.0)
        self.assertEqual(out['processing_ms'], 100000)

    def test_fewer_takes_than_asked_is_said_plainly(self):
        self.api = FakeApi(takes=2)
        out = self.run_job(batch_size=4)
        self.assertEqual(len(out['takes']), 2)
        self.assertIn('Only 2 of 4 takes fit in memory.', out['worker_notes'])


class TestPlannerTemperature(Harness):
    """The creative variation dial: lm_temperature reaches upstream only when the caller sets it and the planner runs."""

    NOT_USED = 'The creative variation setting was not used because the planner did not run.'

    def last(self):
        return self.api.calls[-1]

    def test_absent_means_not_passed(self):
        for changes in ({}, {'lm_temperature': None}, {'model': 'xl-sft'}, {'thinking': False}, {'batch_size': 3}):
            self.run_job(**changes)
            self.assertFalse(hasattr(self.last().params, 'lm_temperature'), changes)

    def test_a_request_without_it_gets_the_same_answer_as_before(self):
        out = self.run_job()
        self.assertEqual(out['controls'], {'steps': 8, 'guidance_scale': 1.0, 'shift': 3.0})
        self.assertEqual(out['worker_notes'], ['Made with the fast model in 8 steps.'])
        out = self.run_job(model='xl-sft')
        self.assertEqual(out['controls'], {'steps': 50, 'guidance_scale': 7.0, 'shift': 3.0})
        self.assertEqual(out['worker_notes'], ['Made with the best-quality model in 50 steps at guidance 7.'])

    def test_the_only_difference_is_the_one_field(self):
        self.run_job(seed=5, batch_size=2)
        plain = self.last()
        self.run_job(seed=5, batch_size=2, lm_temperature=1.3)
        varied = self.last()
        self.assertEqual({k: v for k, v in vars(varied.params).items() if k != 'lm_temperature'}, vars(plain.params))
        self.assertEqual(vars(varied.config), vars(plain.config))
        self.assertIs(varied.llm, plain.llm)
        self.assertIs(varied.dit, plain.dit)

    def test_it_passes_through_to_the_planner_for_both_models(self):
        for model in ('xl-turbo', 'xl-sft'):
            for value in (0.4, 0.85, 1.0, 1.3, 0.0, 2.0):
                out = self.run_job(model=model, lm_temperature=value)
                self.assertNotIn('error', out, (model, value))
                self.assertEqual(self.last().params.lm_temperature, value, (model, value))
                self.assertIs(type(self.last().params.lm_temperature), float, (model, value))
                self.assertIs(self.last().params.thinking, True)
                self.assertIs(self.last().llm, self.llm)

    def test_zero_and_two_are_real_values_not_absent(self):
        self.run_job(lm_temperature=0)
        self.assertTrue(hasattr(self.last().params, 'lm_temperature'))
        self.assertEqual(self.last().params.lm_temperature, 0.0)
        self.run_job(lm_temperature=2)
        self.assertEqual(self.last().params.lm_temperature, 2.0)

    def test_it_is_recorded_in_controls_and_worker_notes(self):
        out = self.run_job(lm_temperature=1.3)
        self.assertEqual(out['controls'], {'steps': 8, 'guidance_scale': 1.0, 'shift': 3.0, 'lm_temperature': 1.3})
        self.assertEqual(out['worker_notes'], ['Made with the fast model in 8 steps.',
                                               'The planner ran at temperature 1.3.'])
        for value, text in ((0, '0'), (0.4, '0.4'), (0.85, '0.85'), (2.0, '2')):
            out = self.run_job(lm_temperature=value)
            self.assertEqual(out['controls']['lm_temperature'], float(value))
            self.assertIn(f'The planner ran at temperature {text}.', out['worker_notes'])
            self.assertTrue(all(isinstance(note, str) and note.endswith('.') for note in out['worker_notes']))
        self.assertNotIn(self.NOT_USED, out['worker_notes'])

    def test_it_changes_nothing_else_in_the_answer(self):
        plain = self.run_job(seed=9)
        varied = self.run_job(seed=9, lm_temperature=1.3)
        for name in ('engine', 'model', 'seed', 'duration_s', 'bytes', 'truncated', 'instrumental', 'thinking', 'gpu',
                     'features', 'versions', 'memory'):
            self.assertEqual(varied[name], plain[name], name)
        self.assertEqual(set(varied), set(plain))
        self.assertEqual(set(varied['timing']), set(plain['timing']))

    def test_without_the_planner_it_is_not_sent_and_not_claimed(self):
        out = self.run_job(thinking=False, lm_temperature=1.3)
        self.assertFalse(hasattr(self.last().params, 'lm_temperature'))
        self.assertNotIn('lm_temperature', out['controls'])
        self.assertIn('The planner was skipped.', out['worker_notes'])
        self.assertIn(self.NOT_USED, out['worker_notes'])
        self.assertFalse([note for note in out['worker_notes'] if 'ran at temperature' in note])

    def test_a_planner_that_could_not_load_is_treated_the_same_way(self):
        self.planner_ok = False
        out = self.run_job(lm_temperature=1.3)
        self.assertNotIn('error', out)
        self.assertFalse(hasattr(self.last().params, 'lm_temperature'))
        self.assertNotIn('lm_temperature', out['controls'])
        self.assertIn('The planner was not available, so the song was made without it.', out['worker_notes'])
        self.assertIn(self.NOT_USED, out['worker_notes'])

    def test_no_note_about_it_when_the_caller_sent_nothing(self):
        for changes in ({}, {'thinking': False}):
            out = self.run_job(**changes)
            self.assertFalse([note for note in out['worker_notes'] if 'variation' in note or 'temperature' in note])

    def test_the_planner_still_gets_the_callers_other_settings(self):
        self.run_job(lm_temperature=0.4, bpm=90, keyscale='D minor', timesignature='3', seed=12)
        params = self.last().params
        self.assertEqual((params.bpm, params.keyscale, params.timesignature, params.seed), (90, 'D minor', '3', 12))
        self.assertEqual((params.use_cot_metas, params.use_cot_caption, params.use_cot_language), (True, False, False))

    def test_the_global_generators_are_still_seeded_for_the_planner(self):
        with mock.patch.object(ace, 'seed_everything') as seeded:
            self.run_job(seed=33, lm_temperature=1.3)
        seeded.assert_called_once_with(33)

    def test_both_dials_together_are_independent_of_each_other(self):
        out = self.run_job(model='xl-sft', guidance_scale=10, lm_temperature=0.4)
        params = self.last().params
        self.assertEqual((params.guidance_scale, params.lm_temperature), (10.0, 0.4))
        self.assertEqual(out['controls'], {'steps': 50, 'guidance_scale': 10.0, 'shift': 3.0, 'lm_temperature': 0.4})
        out = self.run_job(model='xl-turbo', guidance_scale=10, lm_temperature=1.3)     # turbo: guidance ignored, planner dial kept
        params = self.last().params
        self.assertEqual((params.guidance_scale, params.lm_temperature), (1.0, 1.3))
        self.assertEqual(out['controls'], {'steps': 8, 'guidance_scale': 1.0, 'shift': 3.0, 'lm_temperature': 1.3})
        self.assertIn('The fast model ignores the guidance setting.', out['worker_notes'])


class TestGuidanceDial(Harness):
    """The booth's Prompt guidance dial reaches xl-sft as guidance_scale 7, 10 or 13 and never reaches xl-turbo."""

    def last(self):
        return self.api.calls[-1]

    def test_best_quality_takes_the_dials_whole_range(self):
        for value in (7, 10, 13, 7.0, 8.5, 13.0):
            out = self.run_job(model='xl-sft', guidance_scale=value)
            self.assertNotIn('error', out, value)
            self.assertEqual(self.last().params.guidance_scale, float(value))
            self.assertEqual(out['controls']['guidance_scale'], float(value))

    def test_guidance_13_on_sft_end_to_end(self):
        out = self.run_job(model='xl-sft', guidance_scale=13)
        params = self.last().params
        self.assertEqual((params.guidance_scale, params.inference_steps, params.shift), (13.0, 50, 3.0))
        self.assertIs(params.dcw_enabled, False)
        self.assertIn('Made with the best-quality model in 50 steps at guidance 13.', out['worker_notes'])
        self.assertNotIn('lm_temperature', out['controls'])

    def test_the_fast_model_ignores_guidance_and_says_so(self):
        out = self.run_job(model='xl-turbo', guidance_scale=13)
        self.assertEqual(self.last().params.guidance_scale, 1.0)
        self.assertEqual(out['controls']['guidance_scale'], 1.0)
        self.assertIn('The fast model ignores the guidance setting.', out['worker_notes'])
        out = self.run_job(guidance_scale=13)               # the default model is the fast one
        self.assertEqual(out['controls']['guidance_scale'], 1.0)

    def test_without_guidance_each_model_keeps_its_own_default(self):
        out = self.run_job(model='xl-sft')
        self.assertEqual((self.last().params.guidance_scale, out['controls']['guidance_scale']), (7.0, 7.0))
        out = self.run_job()
        self.assertEqual((self.last().params.guidance_scale, out['controls']['guidance_scale']), (1.0, 1.0))
        self.assertNotIn('The fast model ignores the guidance setting.', out['worker_notes'])
        # an explicit 7 on sft is the same request as none
        self.run_job(model='xl-sft', guidance_scale=7, seed=3)
        explicit = dict(vars(self.last().params))
        self.run_job(model='xl-sft', seed=3)
        self.assertEqual(explicit, vars(self.last().params))


class TestFailures(Harness):
    def test_upstream_failure_becomes_one_plain_sentence(self):
        self.api = FakeApi(success=False, error='Traceback (most recent call last)\nRuntimeError: boom')
        out = self.run_job()
        self.assertEqual(out, {'error': 'ACE-Step could not finish this song. Your writing is saved.'})
        self.assertEqual(self.s3.uploads, [])

    def test_memory_failures_ask_for_a_shorter_song(self):
        for error in ('Insufficient free VRAM: need ~11.0 GB, only 6.0 GB available.', 'CUDA out of memory.'):
            self.api = FakeApi(success=False, error=error)
            self.assertEqual(self.run_job(), {'error': ace.TOO_BIG})

    def test_success_without_audio_is_a_failure(self):
        self.api = FakeApi(takes=0)
        self.assertEqual(self.run_job(), {'error': ace.FAILED})

    def test_an_exception_never_escapes_and_only_its_type_is_logged(self):
        self.api = FakeApi(raises=RuntimeError(f'exploded near {SECRET_CAPTION}'))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            out = ace.handler({'id': 'j', 'input': valid(caption=SECRET_CAPTION, lyrics=SECRET_LYRICS)})
        self.assertEqual(out, {'error': 'ACE-Step could not finish (RuntimeError). Your writing is saved.'})
        self.assertNotIn('ZEBRA', buffer.getvalue() + json.dumps(out))
        self.assertIn('RuntimeError', buffer.getvalue())

    def test_logs_and_answers_never_carry_the_callers_words(self):
        self.api = FakeApi(success=False, error=f'Bad input: {SECRET_CAPTION} / {SECRET_LYRICS}')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            out = ace.handler({'id': 'j', 'input': valid(caption=SECRET_CAPTION, lyrics=SECRET_LYRICS)})
        everything = buffer.getvalue() + json.dumps(out)
        self.assertNotIn('ZEBRA', everything)
        self.assertNotIn('KANGAROO', everything)
        self.assertIn('generation failed', buffer.getvalue())

    def test_out_of_memory_exception(self):
        class OutOfMemoryError(RuntimeError):
            pass
        self.api = FakeApi(raises=OutOfMemoryError('CUDA out of memory'))
        self.assertEqual(self.run_job(), {'error': ace.TOO_BIG})

    def test_memory_is_given_back_after_a_crash(self):
        self.api = FakeApi(raises=RuntimeError('x'))
        with mock.patch.object(ace, 'empty_cache') as emptied:
            self.run_job()
        emptied.assert_called()

    def test_upload_failure(self):
        self.s3.fail = True
        self.assertEqual(self.run_job(), {'error': 'ACE-Step could not finish (OSError). Your writing is saved.'})

    def test_missing_storage_settings_fail_before_any_gpu_work(self):
        def no_storage():
            raise KeyError('AWS_ENDPOINT_URL')
        self.patch(ace, 'storage', no_storage)
        out = self.run_job()
        self.assertEqual(out, {'error': 'ACE-Step could not finish (KeyError). Your writing is saved.'})
        self.engine.assert_not_called()
        self.assertEqual(self.api.calls, [])

    def test_a_worker_error_message_is_returned_as_is(self):
        self.engine.side_effect = ace.WorkerError('No graphics card is available on this worker. Try again in a moment.')
        self.assertEqual(self.run_job(), {'error': 'No graphics card is available on this worker. Try again in a moment.'})

    def test_model_load_failure(self):
        self.engine.side_effect = ace.WorkerError('The best-quality music model could not be loaded.')
        self.assertEqual(self.run_job(model='xl-sft'), {'error': 'The best-quality music model could not be loaded.'})
        self.assertEqual(self.api.calls, [])

    def test_validation_errors_do_no_work(self):
        out = self.run_job(duration=5)
        self.assertEqual(out, {'error': 'Length must be between 10 and 600 seconds.'})
        self.engine.assert_not_called()

    def test_the_handler_never_raises(self):
        for job in (None, 5, 'text', [], {}, {'input': None}, {'input': 5}, {'input': []}, {'input': 'x'},
                    {'input': {'caption': 5}}, {'input': {'caption': {'a': 1}, 'duration': 'x'}},
                    {'input': valid(duration=object())}, {'input': valid(lyrics=b'bytes')},
                    {'input': {'caption': 'x' * 10 ** 6, 'duration': 30}}):
            out = silently(ace.handler, job)
            self.assertIsInstance(out, dict, job)
            self.assertTrue('error' in out or 'takes' in out, job)

    def test_bad_inputs_get_the_right_sentences(self):
        self.assertEqual(silently(ace.handler, {'input': 5}), {'error': 'Send the request as named fields.'})
        self.assertEqual(silently(ace.handler, {'input': ['x']}), {'error': 'Send the request as named fields.'})
        self.assertEqual(silently(ace.handler, {}), {'error': 'Describe the music in a few words.'})
        self.assertEqual(silently(ace.handler, None), {'error': 'Describe the music in a few words.'})

    def test_a_list_or_dict_model_gets_the_plain_sentence_not_a_vague_failure(self):
        for value in (['xl-turbo'], {'name': 'xl-turbo'}):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                out = ace.handler({'id': 'j', 'input': valid(model=value)})
            self.assertEqual(out, {'error': 'Model must be xl-turbo or xl-sft.'}, value)
            self.assertEqual(buffer.getvalue(), '')       # the caller's mistake, not a worker failure: nothing logged
        self.engine.assert_not_called()


# ---------------------------------------------------------------- trained styles: the request

STYLE_KEY = 'ace-loras/soul-real-v1'


def styled(**changes):
    """A valid request for a styled song (Best quality, a style key)."""
    return valid(**{'model': 'xl-sft', 'lora_key': STYLE_KEY, **changes})


class TestStyleRequest(unittest.TestCase):
    def check(self, sentence, **changes):
        with self.assertRaises(ace.RequestError) as caught:
            ace.request_input(valid(**changes))
        self.assertEqual(str(caught.exception), sentence)

    def test_a_styled_request_carries_the_key_and_a_scale_of_one_by_default(self):
        request = ace.request_input(styled())
        self.assertEqual(request['style'], {'key': STYLE_KEY, 'scale': 1.0})
        self.assertIs(type(request['style']['scale']), float)
        self.assertEqual(request['model'], 'xl-sft')

    def test_without_a_style_nothing_about_a_style_is_in_the_request(self):
        for changes in ({}, {'lora_key': None}, {'lora_key': ''}, {'lora_key': None, 'lora_scale': None},
                        {'lora_scale': 0.5}, {'lora_scale': 1}, {'model': 'xl-sft', 'lora_scale': 1.5}):
            request = ace.request_input(valid(**changes))
            self.assertIsNone(request['style'], changes)
            self.assertEqual(request['notes'], [], changes)
            self.assertEqual(request['vocal_language'], 'unknown', changes)

    def test_valid_keys(self):
        for name in ('soul-real-v1', 'a', 'A', '0', 'x' * 60, 'a.b_c-d', 'v1.2', 'a.', 'Soul_Real.V1', 'a-'):
            request = ace.request_input(styled(lora_key='ace-loras/' + name))
            self.assertEqual(request['style']['key'], 'ace-loras/' + name, name)

    def test_every_malformed_key_gets_the_one_plain_sentence(self):
        keys = ('k', 'ace-loras', 'ace-loras/', 'ace-loras//', 'ace-loras/a/b', 'ace-loras/a/', 'ace-loras/../x',
                'ace-loras/..', 'ace-loras/.', 'ace-loras/.hidden', 'ace-loras/a..b', 'ace-loras/a/..', '../ace-loras/x',
                '/ace-loras/x', 'ace-loras/' + 'a' * 61, 'ace-loras/a b', 'ace-loras/a\n', 'ace-loras/a\x00',
                'ace-loras/é', 'ace-loras/a;b', 'ace-loras/a%2Fb', 'ace-loras\\x', 'ACE-LORAS/x', 'ace-loras/ x',
                ' ace-loras/x', 'ace-loras/x ', 'ace-loras/a:b', 'ace-loras/a*', 'other/soul-real-v1', 'ace/x',
                'ace-loras/a\r\n', 'ace-loras/%00', 's3://ace-loras/x', 'ace-loras/x?y=1', 'ace-loras/x#1')
        for key in keys:
            self.check(ace.ADAPTER_SENTENCE, model='xl-sft', lora_key=key)
            self.check(ace.ADAPTER_SENTENCE, lora_key=key)       # the key is judged before the model
        for value in (0, 1, False, True, ['ace-loras/x'], {'a': 1}, 1.5, b'ace-loras/x'):
            self.check(ace.ADAPTER_SENTENCE, model='xl-sft', lora_key=value)

    def test_a_null_or_empty_key_means_none(self):
        for value in (None, ''):
            self.assertIsNone(ace.request_input(valid(lora_key=value))['style'])

    def test_only_the_best_quality_model_may_carry_a_style(self):
        self.check(ace.STYLE_MODEL_SENTENCE, lora_key=STYLE_KEY)                          # model absent = xl-turbo
        self.check(ace.STYLE_MODEL_SENTENCE, model='xl-turbo', lora_key=STYLE_KEY)
        self.check(ace.STYLE_MODEL_SENTENCE, model=None, lora_key=STYLE_KEY, lora_scale=0)      # even at strength 0
        self.assertEqual(ace.STYLE_MODEL, 'xl-sft')

    def test_scale_range_and_types(self):
        for value in (0.01, 0.5, 1, 1.0, 1.25, 1.5, 0.6):
            self.assertEqual(ace.request_input(styled(lora_scale=value))['style']['scale'], float(value))
        for value in (-0.001, -1, 1.5000001, 1.51, 2, 100, 10 ** 30, 10 ** 400, -10 ** 400, float('nan'), float('inf'),
                      float('-inf'), True, False, '1', '0.5', '', ' ', [1], {'s': 1}, [], {}):
            self.check(ace.STYLE_SCALE_SENTENCE, model='xl-sft', lora_key=STYLE_KEY, lora_scale=value)

    def test_the_scale_is_checked_even_without_a_key(self):
        for value in (2, -1, float('nan'), True, 'x'):
            self.check(ace.STYLE_SCALE_SENTENCE, lora_scale=value)
        self.check(ace.STYLE_SCALE_SENTENCE, lora_key='', lora_scale=1.6)

    def test_the_sentences(self):
        self.assertEqual(ace.ADAPTER_SENTENCE, 'That style is not available right now.')
        self.assertEqual(ace.STYLE_SCALE_SENTENCE, 'Style strength must be a number from 0 to 1.5.')
        self.assertEqual(ace.STYLE_MODEL_SENTENCE, 'A style needs the best-quality model.')

    def test_a_strength_of_zero_is_a_plain_song_and_says_so(self):
        for value in (0, 0.0):
            request = ace.request_input(styled(lora_scale=value))
            self.assertIsNone(request['style'])
            self.assertEqual(request['notes'], ['The style strength was 0, so the song was made without the style.'])
            self.assertEqual(request['vocal_language'], 'unknown')       # a plain song keeps the plain default

    def test_order_of_checks_is_key_then_model_then_scale(self):
        self.check(ace.ADAPTER_SENTENCE, lora_key='k', lora_scale=9)
        self.check(ace.STYLE_MODEL_SENTENCE, lora_key=STYLE_KEY, lora_scale=9)
        self.check(ace.STYLE_SCALE_SENTENCE, model='xl-sft', lora_key=STYLE_KEY, lora_scale=9)

    def test_language_defaults_to_english_only_for_a_style(self):
        self.assertEqual(ace.request_input(styled())['vocal_language'], 'en')
        self.assertEqual(ace.request_input(styled(vocal_language=None))['vocal_language'], 'en')
        self.assertEqual(ace.request_input(styled(vocal_language=''))['vocal_language'], 'en')
        self.assertEqual(ace.request_input(styled())['notes'], [])
        self.assertEqual(ace.request_input(valid(model='xl-sft'))['vocal_language'], 'unknown')
        self.assertEqual(ace.request_input(valid())['vocal_language'], 'unknown')

    def test_a_language_the_caller_sent_is_kept_with_a_note_that_the_style_is_english(self):
        note = 'The style was taught on English songs, so another language may sound less like it.'
        for sent, want in (('es', 'es'), ('FR', 'fr'), ('unknown', 'unknown')):
            request = ace.request_input(styled(vocal_language=sent))
            self.assertEqual(request['vocal_language'], want)
            self.assertEqual(request['notes'], [note], sent)
        self.assertEqual(ace.request_input(styled(vocal_language='en'))['notes'], [])
        self.assertEqual(ace.request_input(styled(vocal_language=' EN '))['notes'], [])
        self.check('Vocal language must be a short code such as en.', model='xl-sft', lora_key=STYLE_KEY,
                   vocal_language='english')

    def test_the_other_fields_are_validated_the_same_with_a_style(self):
        self.check('Seed must be a whole number from 0 to 2147483647.', model='xl-sft', lora_key=STYLE_KEY, seed=-1)
        self.check('Steps must be a whole number from 4 to 100.', model='xl-sft', lora_key=STYLE_KEY, steps=3)
        request = ace.request_input(styled(steps=30, guidance_scale=9, shift=2))
        self.assertEqual((request['steps'], request['guidance_scale'], request['shift']), (30, 9, 2))


class TestStyleCaption(unittest.TestCase):
    def test_the_tag_goes_first_with_a_comma_and_a_space(self):
        self.assertEqual(ace.tagged_caption('kdsoulr', 'warm soul ballad'), ('kdsoulr, warm soul ballad', None))

    def test_a_caption_that_just_fits_is_not_cut(self):
        tag = 'kdsoulr'
        room = 512 - len(tag) - 2
        caption = ('word ' * 200)[:room - 1] + 'x'
        text, note = ace.tagged_caption(tag, caption)
        self.assertEqual(len(caption), room)
        self.assertEqual((text, note), (f'{tag}, {caption}', None))
        self.assertEqual(len(text), 512)

    def test_one_character_over_is_cut_at_a_word_and_says_so(self):
        tag = 'kdsoulr'
        room = 512 - len(tag) - 2
        caption = 'abcd ' * 120
        caption = caption[:room + 1]
        text, note = ace.tagged_caption(tag, caption)
        self.assertEqual(note, 'The music description was shortened to make room for the style.')
        self.assertLessEqual(len(text), 512)
        self.assertTrue(text.startswith('kdsoulr, abcd'))
        self.assertTrue(all(word == 'abcd' for word in text.split(', ', 1)[1].split()))     # no word was cut in half

    def test_the_tag_is_never_cut_whatever_the_lengths(self):
        for tag in ('k', 'kdsoulr', 't' * 40):
            for length in (1, 50, 400, 470, 471, 480, 500, 505, 510, 511, 512):
                for word in ('a', 'wordy', 'x' * 30):
                    caption = (word + ' ') * 600
                    caption = caption[:length].strip() or 'a'
                    text, note = ace.tagged_caption(tag, caption)
                    self.assertTrue(text.startswith(tag + ', '), (tag, length, word))
                    self.assertLessEqual(len(text), 512, (tag, length, word))
                    self.assertEqual(note is None, len(caption) <= 512 - len(tag) - 2, (tag, length, word))

    def test_a_caption_with_no_spaces_still_leaves_the_tag_whole(self):
        text, note = ace.tagged_caption('kdsoulr', 'x' * 512)
        self.assertTrue(text.startswith('kdsoulr, xxxx'))
        self.assertLessEqual(len(text), 512)
        self.assertIsNotNone(note)

    def test_trailing_punctuation_is_not_left_dangling(self):
        caption = 'warm soul, ' * 60
        text, _note = ace.tagged_caption('kdsoulr', caption)
        self.assertFalse(text.endswith((',', ' ', ';', ':', '-')))


class TestPlainCaptionPlanner(unittest.TestCase):
    class Llm:
        llm_initialized = True
        llm_backend = 'pt'

        def __init__(self):
            self.calls = []

        def generate_with_stop_condition(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {'success': True, 'audio_codes': 'codes'}

        def other(self):
            return 'other'

    def test_the_planner_reads_the_plain_caption(self):
        llm = self.Llm()
        proxy = ace.PlainCaptionPlanner(llm, 'plain words')
        result = proxy.generate_with_stop_condition(caption='kdtag, plain words', lyrics='[Verse]', infer_type='llm_dit',
                                                    temperature=1.3)
        self.assertEqual(result, {'success': True, 'audio_codes': 'codes'})
        self.assertEqual(llm.calls, [((), {'caption': 'plain words', 'lyrics': '[Verse]', 'infer_type': 'llm_dit',
                                           'temperature': 1.3})])

    def test_every_other_attribute_is_the_real_planners(self):
        llm = self.Llm()
        proxy = ace.PlainCaptionPlanner(llm, 'plain')
        self.assertIs(proxy.llm_initialized, True)
        self.assertEqual((proxy.llm_backend, proxy.other()), ('pt', 'other'))
        llm.llm_initialized = False
        self.assertIs(proxy.llm_initialized, False)
        with self.assertRaises(AttributeError):
            proxy.nothing_like_this
        with self.assertRaises(AttributeError):
            proxy._private_thing

    def test_a_missing_inner_planner_cannot_recurse_forever(self):
        proxy = ace.PlainCaptionPlanner.__new__(ace.PlainCaptionPlanner)       # no __init__: no _llm
        with self.assertRaises(AttributeError):
            proxy.anything


# ---------------------------------------------------------------- trained styles: the fold (order of events)

class ListParam:
    """A weight for the order-of-events tests: ints in a list, with a shape. The real arithmetic runs on torch tensors
    in TestFoldOnTorch further down; here only the bookkeeping is under test, so it runs on any Python."""

    def __init__(self, values, shape, dtype='bf16'):
        self.values, self.shape, self.dtype = list(values), tuple(shape), dtype

    def dim(self):
        return len(self.shape)


class ListDecoder:
    def __init__(self, params):
        self.params = params

    def get_parameter(self, path):
        if path not in self.params:
            raise AttributeError(path)
        return self.params[path]


class ListOps:
    """TorchOps on plain lists, with a log of what was asked and switches that make each step misbehave."""

    def __init__(self, events=None):
        self.events = [] if events is None else events
        self.folds = 0
        self.fold_fails_at = None        # the n-th fold call raises
        self.put_back_fails = 0          # this many put_back calls raise before they work
        self.put_back_garbles = 0        # this many put_back calls write wrong values before they work
        self.on_first_fold = None

    def mode(self):
        return contextlib.nullcontext()

    def supported(self, param):
        return param.dim() == 2 and param.dtype in ('bf16', 'fp16', 'fp32')

    def snapshot(self, param):
        self.events.append('snapshot')
        return list(param.values)

    def nbytes(self, saved):
        return 2 * len(saved)

    def checksum(self, param):
        return sum(param.values), sum(i * v for i, v in enumerate(param.values, 1))

    def fold(self, param, a, b, factor):
        self.folds += 1
        self.events.append(('fold', factor))
        if self.folds == 1 and self.on_first_fold:
            self.on_first_fold()
        if self.fold_fails_at == self.folds:
            raise RuntimeError('device fault')
        param.values = [v + 100 + i for i, v in enumerate(param.values)]

    def put_back(self, param, saved):
        self.events.append('put_back')
        if self.put_back_fails:
            self.put_back_fails -= 1
            raise RuntimeError('device fault')
        if self.put_back_garbles:
            self.put_back_garbles -= 1
            param.values = [v + 1 for v in saved]
            return
        param.values = list(saved)


class ListAdapter(ace.Adapter):
    def __init__(self, modules, rank=2, alpha=4.0):
        super().__init__('ace-loras/test-v1', Path('/nowhere'), 'a' * 64, 'kdtest', 'Test', rank, alpha, modules, 1)
        self.read = []

    def pair(self, module):
        self.read.append(module)
        return [[1.0, 2.0]], [[3.0], [4.0]]


SHAPES = {'q_proj': (4, 3), 'k_proj': (2, 3), 'v_proj': (2, 3), 'o_proj': (3, 4)}      # (out, in); the real ones are bigger


def make_decoder(layers=2):
    """(decoder, adapter) with the real module names (layers.N.self_attn|cross_attn.q|k|v|o_proj) and tiny shapes, plus
    weights the adapter does not target."""
    params, modules = {}, {}
    for layer in range(layers):
        for attn in ('self_attn', 'cross_attn'):
            for proj, (out, inner) in SHAPES.items():
                path = f'layers.{layer}.{attn}.{proj}'
                params[path + '.weight'] = ListParam(range(1, out * inner + 1), (out, inner))
                modules[path] = ((2, inner), (out, 2))
    params['layers.0.self_attn.q_norm.weight'] = ListParam([5, 6], (2,), 'fp32')
    params['proj_in.weight'] = ListParam(range(7, 13), (2, 3))
    params['encoder.layers.0.self_attn.q_proj.weight'] = ListParam(range(1, 13), (4, 3))      # the lyric encoder has q_proj too
    return ListDecoder(params), ListAdapter(modules)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class TestFoldBookkeeping(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        self.decoder, self.adapter = make_decoder()
        self.ops = ListOps()
        self.handler = SimpleNamespace(model=SimpleNamespace(decoder=self.decoder))
        self.pristine = self.weights()
        self.targets = [f'{module}.weight' for module in self.adapter.modules]
        patcher = mock.patch.object(ace, 'empty_cache')
        patcher.start()
        self.addCleanup(patcher.stop)

    def weights(self):
        return {name: list(param.values) for name, param in self.decoder.params.items()}

    def scope(self, scale=1.0, model='xl-sft', handler=None):
        return ace.apply_adapter(handler or self.handler, self.adapter, scale, model, self.ops)

    def refused(self, handler=None):
        with quiet(), self.assertRaises(ace.WorkerError) as caught:
            with self.scope(handler=handler):
                self.fail('the block must not run')
        self.assertEqual(str(caught.exception), ace.ADAPTER_SENTENCE)

    def test_weights_change_inside_and_come_back_exactly(self):
        with quiet(), self.scope() as fold:
            inside = self.weights()
            self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertEqual(len(self.targets), 16)
        for name in self.targets:
            self.assertNotEqual(inside[name], self.pristine[name], name)
        for name in set(self.pristine) - set(self.targets):          # q_norm, proj_in and the encoder's q_proj
            self.assertEqual(inside[name], self.pristine[name], name)
        self.assertEqual(self.weights(), self.pristine)
        self.assertIs(fold.restored, True)
        self.assertEqual(ace.STATE['dirty'], set())
        self.assertEqual(fold.saved, {})

    def test_only_the_decoders_own_weights_are_touched(self):
        with quiet(), self.scope():
            self.assertEqual(self.decoder.params['encoder.layers.0.self_attn.q_proj.weight'].values, list(range(1, 13)))

    def test_every_weight_is_copied_before_the_first_one_is_written(self):
        with quiet(), self.scope():
            pass
        kinds = [event if isinstance(event, str) else event[0] for event in self.ops.events]
        first = kinds.index('fold')
        self.assertEqual(kinds[:first], ['snapshot'] * 16)
        self.assertEqual(kinds[first:first + 16], ['fold'] * 16)
        self.assertEqual(kinds[first + 16:], ['put_back'] * 16)

    def test_the_dirty_mark_is_set_before_the_first_write(self):
        seen = []
        self.ops.on_first_fold = lambda: seen.append(set(ace.STATE['dirty']))
        with quiet(), self.scope():
            pass
        self.assertEqual(seen, [{'xl-sft'}])

    def test_the_fold_factor_is_alpha_over_rank_times_the_scale(self):
        for scale in (1.0, 0.6, 1.5, 0.0001):
            self.ops.events.clear()
            with quiet(), self.scope(scale):
                pass
            factors = [event[1] for event in self.ops.events if isinstance(event, tuple)]
            self.assertEqual(len(factors), 16)
            for factor in factors:
                self.assertAlmostEqual(factor, (4.0 / 2) * scale, places=12)

    def test_pairs_are_read_when_folded_each_once(self):
        self.assertEqual(self.adapter.read, [])
        with quiet(), self.scope():
            self.assertEqual(sorted(self.adapter.read), sorted(self.adapter.modules))
        self.assertEqual(len(self.adapter.read), 16)

    def test_restore_after_an_exception_in_the_block(self):
        for error in (RuntimeError('render failed'), ValueError('x'), MemoryError(), OSError('disk')):
            with quiet(), self.assertRaises(type(error)):
                with self.scope():
                    raise error
            self.assertEqual(self.weights(), self.pristine, repr(error))
            self.assertEqual(ace.STATE['dirty'], set(), repr(error))

    def test_restore_after_a_cancellation(self):
        for error in (KeyboardInterrupt(), SystemExit(1), asyncio.CancelledError()):
            with quiet(), self.assertRaises(type(error)):
                with self.scope():
                    raise error
            self.assertEqual(self.weights(), self.pristine, repr(error))
            self.assertEqual(ace.STATE['dirty'], set(), repr(error))

    def test_a_fold_that_fails_half_way_puts_back_what_it_wrote(self):
        for failing in (1, 5, 16):
            self.ops.folds = 0
            self.ops.fold_fails_at = failing
            with quiet(), self.assertRaises(RuntimeError):
                with self.scope():
                    self.fail('the block must not run')
            self.assertEqual(self.weights(), self.pristine, failing)
            self.assertEqual(ace.STATE['dirty'], set(), failing)

    def test_an_interrupt_during_the_fold_puts_back_too(self):
        def interrupt():
            raise KeyboardInterrupt()
        self.ops.on_first_fold = interrupt
        with quiet(), self.assertRaises(KeyboardInterrupt):
            with self.scope():
                self.fail('the block must not run')
        self.assertEqual(self.weights(), self.pristine)
        self.assertEqual(ace.STATE['dirty'], set())

    def test_a_restore_that_cannot_be_done_marks_the_model_dirty_and_says_so(self):
        self.ops.put_back_fails = 10 ** 6
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope() as fold:
            pass
        self.assertIs(fold.restored, False)
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertIn('RESTORE FAILED for 16 weights', buffer.getvalue())
        self.assertIn('RuntimeError', buffer.getvalue())
        self.assertEqual(fold.saved, {})            # the 3 GB pristine copy is not kept for a model that will be reloaded

    def test_a_failed_restore_does_not_hide_the_renders_own_error(self):
        self.ops.put_back_fails = 10 ** 6
        with quiet(), self.assertRaises(ValueError) as caught:
            with self.scope():
                raise ValueError('render failed')
        self.assertEqual(str(caught.exception), 'render failed')
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})

    def test_a_one_off_fault_while_restoring_is_tried_again(self):
        self.ops.put_back_fails = 1
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope() as fold:
            pass
        self.assertIs(fold.restored, True)
        self.assertEqual(self.weights(), self.pristine)
        self.assertEqual(ace.STATE['dirty'], set())
        self.assertIn('restore error: RuntimeError', buffer.getvalue())

    def test_a_restore_that_wrote_the_wrong_values_is_caught_by_the_check_and_redone(self):
        self.ops.put_back_garbles = 3
        with quiet(), self.scope() as fold:
            pass
        self.assertIs(fold.restored, True)
        self.assertEqual(self.weights(), self.pristine)
        self.assertEqual(ace.STATE['dirty'], set())
        self.assertEqual(self.ops.events.count('put_back'), 16 + 3)         # the three bad ones were put back again

    def test_a_restore_that_keeps_writing_the_wrong_values_leaves_the_model_dirty(self):
        self.ops.put_back_garbles = 10 ** 6
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope() as fold:
            pass
        self.assertIs(fold.restored, False)
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertIn('RESTORE FAILED', buffer.getvalue())

    def test_one_number_off_in_one_weight_of_sixteen_is_found_and_that_weight_written_again(self):
        calls = {'n': 0}
        real = self.ops.put_back

        def put_back(param, saved):
            calls['n'] += 1
            real(param, saved)
            if calls['n'] == 7:
                param.values[0] += 1          # one number, in one weight, on the first attempt only
        self.ops.put_back = put_back
        with quiet(), self.scope() as fold:
            pass
        self.assertIs(fold.restored, True)
        self.assertEqual(self.weights(), self.pristine)
        self.assertEqual(calls['n'], 17)           # 16 weights, and the one that failed the check once more

    def test_nothing_is_written_when_a_target_is_missing_or_does_not_fit(self):
        removed = self.decoder.params.pop('layers.1.cross_attn.v_proj.weight')
        self.refused()
        self.decoder.params['layers.1.cross_attn.v_proj.weight'] = removed
        removed.shape = (2, 4)                    # a different input width
        self.refused()
        removed.shape = (3, 3)
        self.refused()
        removed.shape = (2, 3, 1)
        self.refused()
        removed.shape, removed.dtype = (2, 3), 'int8'
        self.refused()
        for kind in ('snapshot', 'put_back'):
            self.assertNotIn(kind, self.ops.events)
        self.assertEqual([e for e in self.ops.events if isinstance(e, tuple)], [])
        self.assertEqual(ace.STATE['dirty'], set())

    def test_a_weight_two_pairs_share_is_refused(self):
        self.decoder.params['layers.1.cross_attn.v_proj.weight'] = self.decoder.params['layers.1.cross_attn.k_proj.weight']
        self.refused()
        self.assertEqual(self.ops.events, [])

    def test_a_decoder_that_already_carries_an_adapter_is_refused(self):
        self.decoder.peft_config = {'default': object()}
        self.refused()
        del self.decoder.peft_config
        self.handler.lora_loaded = True               # upstream's own add_lora was used on this handler
        self.refused()
        self.assertEqual(self.ops.events, [])

    def test_a_handler_without_a_decoder_is_refused(self):
        self.refused(SimpleNamespace(model=SimpleNamespace()))
        self.refused(SimpleNamespace(model=None))
        self.refused(SimpleNamespace())

    def test_a_model_that_is_already_dirty_is_not_folded_again(self):
        ace.STATE['dirty'].add('xl-sft')
        self.refused()
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})        # still marked: the refusal does not clear it
        self.assertEqual(self.weights(), self.pristine)
        self.assertEqual(self.ops.events, [])

    def test_the_dirty_mark_belongs_to_the_model_that_was_folded(self):
        ace.STATE['dirty'].add('xl-turbo')
        with quiet(), self.scope(model='xl-sft'):
            self.assertEqual(ace.STATE['dirty'], {'xl-turbo', 'xl-sft'})
        self.assertEqual(ace.STATE['dirty'], {'xl-turbo'})

    def test_the_memory_is_given_back_before_the_render_and_again_after_it(self):
        seen = []
        with mock.patch.object(ace, 'empty_cache', side_effect=lambda: seen.append('empty')) as emptied, quiet():
            with self.scope():
                seen.append('render')
        self.assertEqual(seen, ['empty', 'render', 'empty'])
        self.assertEqual(emptied.call_count, 2)
        seen.clear()
        with mock.patch.object(ace, 'empty_cache', side_effect=lambda: seen.append('empty')), quiet():
            with self.assertRaises(RuntimeError), self.scope():
                raise RuntimeError('render failed')
        self.assertEqual(seen, ['empty', 'empty'])

    def test_the_log_says_how_much_was_held_and_never_a_path_or_a_word_of_the_song(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope(0.6):
            pass
        log = buffer.getvalue()
        self.assertIn('ACE-Step style Test folded into 16 weights at strength 0.6', log)
        self.assertIn('GiB held on the CPU', log)
        self.assertIn('removed and the 16 weights verified', log)
        self.assertNotIn('/nowhere', log)

    def test_a_weight_object_replaced_during_the_render_is_restored_in_place(self):
        name = 'layers.1.cross_attn.k_proj.weight'
        with quiet(), self.scope() as fold:
            old = self.decoder.params[name]
            self.decoder.params[name] = ListParam(old.values, old.shape, old.dtype)       # same numbers, a new object
        self.assertIs(fold.restored, True)
        self.assertEqual(self.weights(), self.pristine)             # the live object was restored, not the old one
        self.assertNotEqual(old.values, self.pristine[name])        # the one that was replaced is dead and stays as it was
        self.assertEqual(fold.moved, {'layers.1.cross_attn.k_proj'})
        self.assertEqual(ace.STATE['dirty'], set())

    def test_a_decoder_replaced_during_the_render_is_restored_in_place(self):
        copy = ListDecoder({name: ListParam(param.values, param.shape, param.dtype)
                            for name, param in self.decoder.params.items()})
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope() as fold:
            self.handler.model.decoder = copy
        self.assertIs(fold.restored, True)
        self.assertEqual({name: list(param.values) for name, param in copy.params.items()}, self.pristine)
        self.assertEqual(len(fold.moved), 16)
        self.assertIn('the render replaced decoder weight objects', buffer.getvalue())

    def test_a_decoder_that_lost_the_weights_is_a_failed_restore_and_never_a_silent_one(self):
        for replacement in (ListDecoder({}), None):
            ace.STATE['dirty'].clear()
            self.handler.model.decoder = self.decoder
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), self.scope() as fold:
                self.handler.model.decoder = replacement
            self.assertIs(fold.restored, False)
            self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
            self.assertIn('RESTORE FAILED for 16 weights', buffer.getvalue())

    def test_a_second_fold_after_a_clean_restore_is_allowed(self):
        for _ in range(3):
            with quiet(), self.scope() as fold:
                self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
            self.assertIs(fold.restored, True)
            self.assertEqual(self.weights(), self.pristine)


# ---------------------------------------------------------------- trained styles: fetching and checking the files

STYLE_PREFIX = 'ace-loras/test-v1'
STYLE_MODULES = {'layers.0.self_attn.q_proj': ((2, 3), (4, 2)), 'layers.0.self_attn.k_proj': ((2, 3), (2, 2))}


def style_config(**changes):
    config = {'peft_type': 'LORA', 'r': 2, 'lora_alpha': 4, 'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
              'use_rslora': False, 'use_dora': False, 'bias': 'none', 'fan_in_fan_out': False, 'lora_bias': False,
              'rank_pattern': {}, 'alpha_pattern': {}, 'modules_to_save': None, 'task_type': 'FEATURE_EXTRACTION',
              'base_model_name_or_path': '/workspace/ckpt/acestep-v15-xl-sft'}
    config.update(changes)
    return config


class TestStyleFetch(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {'ACE_LORA_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        self.s3 = FakeS3()
        self.weights = b'W' * 5000
        self.said = []
        self.modules_error = None
        self.read = []

        def modules(path, rank):
            self.read.append((Path(path).name, rank))
            if self.modules_error is not None:
                raise self.modules_error
            return dict(STYLE_MODULES)
        patcher = mock.patch.object(ace, 'adapter_modules', modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.put()

    def put(self, serve=None, config=None, weights=None, raw_serve=None):
        weights = self.weights if weights is None else weights
        data = {'adapter_model_sha256': hashlib.sha256(self.weights).hexdigest(), 'trigger_tag': 'kdtest',
                'name': 'Test'}
        data.update(serve or {})
        data = {key: value for key, value in data.items() if value is not ...}        # ... drops a field
        self.s3.objects[f'{STYLE_PREFIX}/serve.json'] = raw_serve if raw_serve is not None else json.dumps(data).encode()
        self.s3.objects[f'{STYLE_PREFIX}/adapter_config.json'] = json.dumps(config or style_config()).encode()
        self.s3.objects[f'{STYLE_PREFIX}/adapter_model.safetensors'] = weights

    def fetch(self, key=STYLE_PREFIX):
        return silently(ace.fetch_adapter, self.s3, 'sound-bucket', key, self.said.append)

    def refused(self, expect_log=None):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(ace.WorkerError) as caught:
            ace.fetch_adapter(self.s3, 'sound-bucket', STYLE_PREFIX, self.said.append)
        self.assertEqual(str(caught.exception), ace.ADAPTER_SENTENCE)
        self.assertEqual(ace.STATE['styles'], {})
        self.assertFalse(list(Path(self.tmp.name).rglob('*.part')))
        if expect_log:
            self.assertIn(expect_log, buffer.getvalue())
        return buffer.getvalue()

    def test_a_good_style_is_fetched_checked_and_described(self):
        adapter = self.fetch()
        self.assertEqual((adapter.key, adapter.name, adapter.tag), (STYLE_PREFIX, 'Test', 'kdtest'))
        self.assertEqual(adapter.sha256, hashlib.sha256(self.weights).hexdigest())
        self.assertEqual((adapter.rank, adapter.alpha, adapter.scaling), (2, 4.0, 2.0))
        self.assertEqual(adapter.modules, STYLE_MODULES)
        self.assertEqual(adapter.folder, Path(self.tmp.name) / 'test-v1')
        self.assertEqual(adapter.size, 5000)
        self.assertEqual((adapter.folder / 'adapter_model.safetensors').read_bytes(), self.weights)
        self.assertTrue((adapter.folder / 'serve.json').is_file() and (adapter.folder / 'adapter_config.json').is_file())
        self.assertEqual(sorted(p.name for p in adapter.folder.iterdir()),
                         ['adapter_config.json', 'adapter_model.safetensors', 'serve.json'])     # no .part left behind
        self.assertIs(ace.STATE['styles'][STYLE_PREFIX], adapter)
        self.assertEqual(self.said, ['Loading the style'])
        self.assertEqual(self.read, [('adapter_model.safetensors.part', 2)])       # read for shape before it got its name

    def test_the_three_objects_are_asked_for_by_exact_key_with_the_same_client(self):
        self.fetch()
        self.assertEqual(self.s3.downloads, [('sound-bucket', f'{STYLE_PREFIX}/serve.json'),
                                             ('sound-bucket', f'{STYLE_PREFIX}/adapter_config.json'),
                                             ('sound-bucket', f'{STYLE_PREFIX}/adapter_model.safetensors')])

    def test_the_cache_is_reused_across_requests_of_the_same_worker(self):
        first = self.fetch()
        second = self.fetch()
        self.assertIs(first, second)
        self.assertEqual(len(self.s3.downloads), 3)
        self.assertEqual(self.said, ['Loading the style'])          # only the first request waited

    def test_a_cache_that_lost_its_file_or_changed_size_is_fetched_again(self):
        adapter = self.fetch()
        (adapter.folder / 'adapter_model.safetensors').unlink()
        again = self.fetch()
        self.assertEqual(len(self.s3.downloads), 6)
        self.assertTrue((again.folder / 'adapter_model.safetensors').is_file())
        (again.folder / 'adapter_model.safetensors').write_bytes(b'short')
        self.fetch()
        self.assertEqual(len(self.s3.downloads), 9)
        self.assertEqual((again.folder / 'adapter_model.safetensors').read_bytes(), self.weights)

    def test_two_styles_do_not_share_a_folder_or_a_cache_entry(self):
        for name in ('a', 'b'):
            for file in ('serve.json', 'adapter_config.json', 'adapter_model.safetensors'):
                self.s3.objects[f'ace-loras/{name}/{file}'] = self.s3.objects[f'{STYLE_PREFIX}/{file}']
        first, second = self.fetch('ace-loras/a'), self.fetch('ace-loras/b')
        self.assertNotEqual(first.folder, second.folder)
        self.assertEqual(set(ace.STATE['styles']), {'ace-loras/a', 'ace-loras/b'})

    def test_a_checksum_mismatch_is_refused_and_leaves_nothing_behind(self):
        self.s3.objects[f'{STYLE_PREFIX}/adapter_model.safetensors'] = b'X' * 5000
        log = self.refused('the adapter does not match its checksum')
        self.assertFalse((Path(self.tmp.name) / 'test-v1' / 'adapter_model.safetensors').exists())
        self.assertEqual(self.read, [])           # a file that failed its checksum is never even opened for shape
        self.assertNotIn(STYLE_PREFIX, log)

    def test_a_bad_download_never_replaces_a_good_cached_file(self):
        adapter = self.fetch()
        ace.STATE['styles'].clear()                 # a new worker process finding the old folder
        self.s3.objects[f'{STYLE_PREFIX}/adapter_model.safetensors'] = b'Y' * 5000
        self.refused()
        self.assertEqual((adapter.folder / 'adapter_model.safetensors').read_bytes(), self.weights)
        self.s3.objects[f'{STYLE_PREFIX}/adapter_model.safetensors'] = self.weights        # and a good one works after
        self.assertEqual(self.fetch().size, 5000)

    def test_a_missing_object_is_a_plain_refusal_for_each_of_the_three(self):
        for name in ('serve.json', 'adapter_config.json', 'adapter_model.safetensors'):
            self.put()
            del self.s3.objects[f'{STYLE_PREFIX}/{name}']
            log = self.refused('ACE-Step style could not be fetched: OSError')
            self.assertNotIn('404', log)

    def test_a_download_that_breaks_half_way_leaves_no_part_file(self):
        self.s3.broken.add(f'{STYLE_PREFIX}/adapter_model.safetensors')
        log = self.refused('could not be fetched: ConnectionError')
        self.assertFalse((Path(self.tmp.name) / 'test-v1' / 'adapter_model.safetensors').exists())
        self.assertNotIn('connection reset', log)           # only the type of the error is logged

    def test_any_exception_from_the_client_is_a_plain_refusal(self):
        class ClientError(Exception):
            pass
        for error in (ClientError('AccessDenied'), ValueError('x'), KeyError('k'), RuntimeError('r'), TimeoutError(),
                      PermissionError('p'), MemoryError(), ImportError('boto')):
            self.s3.download_file = mock.Mock(side_effect=error)
            log = self.refused(f'could not be fetched: {type(error).__name__}')
            self.assertNotIn('AccessDenied', log)

    def test_an_interrupt_during_the_download_goes_through_and_leaves_no_part_file(self):
        self.s3.download_file = mock.Mock(side_effect=KeyboardInterrupt())
        with quiet(), self.assertRaises(KeyboardInterrupt):
            ace.fetch_adapter(self.s3, 'sound-bucket', STYLE_PREFIX)
        self.assertFalse(list(Path(self.tmp.name).rglob('*.part')))
        self.assertEqual(ace.STATE['styles'], {})

    def test_a_stale_part_file_from_a_crash_is_overwritten(self):
        folder = Path(self.tmp.name) / 'test-v1'
        folder.mkdir()
        (folder / 'adapter_model.safetensors.part').write_bytes(b'junk' * 10)
        self.assertEqual(self.fetch().size, 5000)
        self.assertFalse(list(folder.glob('*.part')))

    def test_an_unwritable_cache_folder_is_a_plain_refusal(self):
        blocker = Path(self.tmp.name) / 'file'
        blocker.write_text('x')
        with mock.patch.dict(os.environ, {'ACE_LORA_DIR': str(blocker / 'inside')}):
            self.refused('could not be fetched')
        self.assertEqual(self.s3.downloads, [])

    def test_a_shape_check_that_fails_is_a_refusal_with_its_reason_and_the_file_is_not_kept(self):
        self.modules_error = ace.StyleProblem('the adapter has half a pair')
        self.refused('ACE-Step style refused: the adapter has half a pair')
        self.assertFalse((Path(self.tmp.name) / 'test-v1' / 'adapter_model.safetensors').exists())
        self.modules_error = OSError('disk')
        self.refused('could not be fetched: OSError')

    def test_serve_json_problems(self):
        sha = hashlib.sha256(self.weights).hexdigest()
        bad = [('not json', dict(raw_serve=b'{"adapter_model_sha256": ')), ('a list', dict(raw_serve=b'[1, 2]')),
               ('a string', dict(raw_serve=b'"x"')), ('empty', dict(raw_serve=b'')),
               ('no checksum', dict(serve={'adapter_model_sha256': ...})),
               ('short checksum', dict(serve={'adapter_model_sha256': sha[:63]})),
               ('long checksum', dict(serve={'adapter_model_sha256': sha + '0'})),
               ('checksum with a newline', dict(serve={'adapter_model_sha256': sha + '\n'})),
               ('checksum not hex', dict(serve={'adapter_model_sha256': 'g' * 64})),
               ('checksum a number', dict(serve={'adapter_model_sha256': 12345})),
               ('checksum of other bytes', dict(serve={'adapter_model_sha256': hashlib.sha256(b'other').hexdigest()})),
               ('no tag', dict(serve={'trigger_tag': ...})), ('blank tag', dict(serve={'trigger_tag': '   '})),
               ('tag too long', dict(serve={'trigger_tag': 't' * 41})),
               ('tag with a newline', dict(serve={'trigger_tag': 'kd\ntest'})),
               ('tag with a control character', dict(serve={'trigger_tag': 'kd\x00test'})),
               ('tag a number', dict(serve={'trigger_tag': 7})), ('tag a list', dict(serve={'trigger_tag': ['kd']})),
               ('too big', dict(raw_serve=b'{"pad": "' + b'x' * (1 << 20) + b'"}'))]
        for label, changes in bad:
            self.put(**changes)
            self.refused()
            self.assertFalse(list(Path(self.tmp.name).rglob('adapter_model.safetensors')), label)

    def test_serve_json_accepts_these_forms(self):
        sha = hashlib.sha256(self.weights).hexdigest()
        self.put(serve={'adapter_model_sha256': sha.upper()})
        adapter = self.fetch()
        self.assertEqual(adapter.sha256, sha)                  # compared and reported in lower case
        ace.STATE['styles'].clear()
        for serve, tag in (({'trigger_tag': ...}, None), ({'trigger_tag': ..., 'trigger_word': 'kdword'}, 'kdword'),
                           ({'trigger_tag': ..., 'custom_tag': 'kdcustom'}, 'kdcustom'),
                           ({'trigger_tag': 'first', 'trigger_word': 'second'}, 'first'),
                           ({'trigger_tag': '  padded  '}, 'padded'), ({'trigger_tag': 't' * 40}, 't' * 40)):
            self.put(serve=serve)
            if tag is None:
                self.refused()
                continue
            self.assertEqual(self.fetch().tag, tag)
            ace.STATE['styles'].clear()

    def test_the_display_name_falls_back_to_the_key(self):
        for serve, name in (({'name': ...}, 'test-v1'), ({'name': ''}, 'test-v1'), ({'name': 5}, 'test-v1'),
                            ({'name': 'n' * 41}, 'test-v1'), ({'name': 'a\nb'}, 'test-v1'), ({'name': ' Soul '}, 'Soul'),
                            ({'name': 'n' * 40}, 'n' * 40)):
            self.put(serve=serve)
            self.assertEqual(self.fetch().name, name, serve)
            ace.STATE['styles'].clear()

    def test_adapter_config_problems(self):
        bad = [style_config(peft_type='LOKR'), style_config(peft_type=None), style_config(r=0), style_config(r=-4),
               style_config(r='64'), style_config(r=True), style_config(r=2.0), style_config(lora_alpha=0),
               style_config(lora_alpha=-1), style_config(lora_alpha='128'), style_config(lora_alpha=True),
               style_config(lora_alpha=float('nan')), style_config(use_rslora=True), style_config(use_dora=True),
               style_config(use_qalora=True), style_config(lora_bias=True), style_config(fan_in_fan_out=True),
               style_config(bias='all'), style_config(bias='lora_only'), style_config(rank_pattern={'q_proj': 8}),
               style_config(alpha_pattern={'q_proj': 8}), style_config(modules_to_save=['norm']),
               style_config(target_parameters=['w']), style_config(alora_invocation_tokens=[1]),
               style_config(layer_replication=[[0, 1]]), style_config(target_modules=['gate_proj']),
               style_config(target_modules=['q_proj', 'gate_proj']), style_config(target_modules=[]),
               style_config(target_modules='q_proj'), style_config(target_modules=None)]
        for config in bad:
            self.put(config=config)
            self.refused()
        self.put(config=style_config(target_modules=['q_proj']))                 # a subset is fine
        self.assertEqual(self.fetch().rank, 2)

    def test_a_config_that_is_not_json_or_not_an_object_is_refused(self):
        for raw in (b'', b'[]', b'{', b'null', b'"x"'):
            self.put()
            self.s3.objects[f'{STYLE_PREFIX}/adapter_config.json'] = raw
            self.refused()

    def test_the_rank_the_settings_say_is_what_the_weights_are_read_against(self):
        self.put(config=style_config(r=8, lora_alpha=16))
        adapter = self.fetch()
        self.assertEqual((adapter.rank, adapter.alpha, adapter.scaling), (8, 16.0, 2.0))
        self.assertEqual(self.read, [('adapter_model.safetensors.part', 8)])

    def test_fractional_alpha_is_kept(self):
        self.put(config=style_config(r=4, lora_alpha=6.5))
        self.assertEqual(self.fetch().scaling, 1.625)

    def test_progress_is_announced_only_when_something_is_fetched(self):
        self.fetch()
        self.fetch()
        self.assertEqual(self.said, ['Loading the style'])
        self.assertIsNotNone(ace.fetch_adapter(self.s3, 'sound-bucket', STYLE_PREFIX))      # no callback is fine

    def test_the_log_says_what_was_fetched_and_nothing_about_the_bucket(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ace.fetch_adapter(self.s3, 'sound-bucket', STYLE_PREFIX)
        self.assertEqual(buffer.getvalue().strip(), 'ACE-Step style Test fetched: 2 pairs, rank 2')

    def test_the_cache_folder_comes_from_the_environment_or_the_default(self):
        self.assertEqual(ace.style_root(), Path(self.tmp.name))
        with mock.patch.dict(os.environ):
            os.environ.pop('ACE_LORA_DIR')
            self.assertEqual(ace.style_root(), Path('/tmp/ace-lora'))

    def test_file_sha256_matches_hashlib(self):
        path = Path(self.tmp.name) / 'blob'
        data = b'abc' * 1000003                    # more than one 1 MiB chunk
        path.write_bytes(data)
        self.assertEqual(ace.file_sha256(path), hashlib.sha256(data).hexdigest())


# ---------------------------------------------------------------- trained styles: a whole styled song, end to end

class FakePlanner:
    llm_initialized = True
    llm_backend = 'pt'

    def __init__(self):
        self.asked = []

    def generate_with_stop_condition(self, **kwargs):
        self.asked.append(kwargs)
        return {'success': True}


class StyledApi(FakeApi):
    """FakeApi that behaves like upstream's generate_music where it matters here: it asks the planner with
    params.caption (by keyword, as upstream does), hands the DiT params.caption, and notes the decoder's weights, the
    render lock and the dirty marks at that moment."""

    def __init__(self, decoder, events=None, **kwargs):
        super().__init__(**kwargs)
        self.decoder, self.events, self.seen = decoder, events, []

    def generate_music(self, dit, llm, params, config, save_dir=None, progress=None):
        if self.events is not None:
            self.events.append('generate')
        weights = {name: list(param.values) for name, param in self.decoder.params.items()}
        if llm is not None and getattr(llm, 'llm_initialized', False) and params.thinking:
            llm.generate_with_stop_condition(caption=params.caption or '', lyrics=params.lyrics, infer_type='llm_dit',
                                             temperature=getattr(params, 'lm_temperature', 0.85))
        self.seen.append(SimpleNamespace(weights=weights, dit_caption=params.caption, locked=ace.RENDER_LOCK.locked(),
                                         dirty=set(ace.STATE['dirty'])))
        return super().generate_music(dit, llm, params, config, save_dir, progress)


REAL_FETCH_ADAPTER = ace.fetch_adapter


class StyledHarness(Harness):
    def setUp(self):
        super().setUp()
        self.events = []
        self.decoder, self.adapter = make_decoder()
        self.ops = ListOps(self.events)
        self.pristine = self.weights()
        self.targets = [f'{module}.weight' for module in self.adapter.modules]
        self.llm = self.planner_obj = FakePlanner()
        self.api = StyledApi(self.decoder, self.events)
        self.dits['xl-sft'] = SimpleNamespace(model=SimpleNamespace(decoder=self.decoder), name='xl-sft')
        self.fetches, self.fetch_error, self.ops_made = [], False, []

        def make_ops():
            self.ops_made.append(self.ops)
            return self.ops

        def fetch(client, bucket, key, say=None):
            self.events.append('fetch')
            self.fetches.append((client, bucket, key))
            if self.fetch_error:
                raise ace.WorkerError(ace.ADAPTER_SENTENCE)
            return self.adapter
        self.patch(ace, 'TorchOps', make_ops)
        self.patch(ace, 'fetch_adapter', fetch)
        self.engine.side_effect = lambda model: (self.events.append('engine'), self.dits.setdefault(
            model, SimpleNamespace(model=object(), name=model)))[1]
        self.planner.side_effect = lambda: (self.events.append('planner'), self.llm if self.planner_ok else None)[1]
        self.patch(ace, 'gpu_peak', lambda reset=False: (self.events.append('reset' if reset else 'read'),
                                                         None if reset else 12.5)[1])
        real_encode = ace.encode_take
        self.patch(ace, 'encode_take', lambda audio, folder: (self.events.append('encode'), real_encode(audio, folder))[1])

    def weights(self):
        return {name: list(param.values) for name, param in self.decoder.params.items()}

    def styled(self, **changes):
        return self.run_job(**{'model': 'xl-sft', 'lora_key': STYLE_KEY, **changes})

    def last(self):
        return self.api.calls[-1]

    def folds(self):
        return [event[1] for event in self.events if isinstance(event, tuple) and event[0] == 'fold']


class TestStyledSong(StyledHarness):
    def test_a_styled_song_end_to_end(self):
        out = self.styled(seed=5)
        self.assertNotIn('error', out)
        self.assertEqual(out['model'], 'xl-sft')
        self.assertEqual(out['style'], {'key': STYLE_KEY, 'name': 'Test', 'scale': 1.0, 'tag': 'kdtest', 'sha256': 'a' * 64})
        self.assertEqual(self.fetches, [(self.s3, 'sound-bucket', STYLE_KEY)])
        seen = self.api.seen[0]
        self.assertEqual(seen.dit_caption, 'kdtest, Warm acoustic folk, brushed drums')          # the DiT: tagged
        self.assertEqual(self.planner_obj.asked[0]['caption'], 'Warm acoustic folk, brushed drums')      # the planner: plain
        for name in self.targets:
            self.assertNotEqual(seen.weights[name], self.pristine[name], name)               # folded while it rendered
        self.assertEqual(self.weights(), self.pristine)                                      # and back afterwards
        self.assertEqual((seen.dirty, ace.STATE['dirty']), ({'xl-sft'}, set()))
        self.assertEqual(self.last().params.vocal_language, 'en')
        self.assertEqual(self.last().params.caption, 'kdtest, Warm acoustic folk, brushed drums')
        self.assertIsInstance(self.last().llm, ace.PlainCaptionPlanner)
        self.assertEqual([t['seed'] for t in out['takes']], [5])
        json.dumps(out)

    def test_the_whole_path_with_the_real_fetch_and_a_fake_bucket(self):
        weights = b'W' * 3000
        prefix = {'serve.json': json.dumps({'adapter_model_sha256': hashlib.sha256(weights).hexdigest(),
                                            'trigger_tag': 'kdsoulr', 'name': 'Soul'}).encode(),
                  'adapter_config.json': json.dumps(style_config()).encode(), 'adapter_model.safetensors': weights}
        for name, data in prefix.items():
            self.s3.objects[f'{STYLE_KEY}/{name}'] = data
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {'ACE_LORA_DIR': td}):
            self.patch(ace, 'fetch_adapter', REAL_FETCH_ADAPTER)
            self.patch(ace, 'adapter_modules', lambda path, rank: dict(self.adapter.modules))
            self.patch(ace.Adapter, 'pair', lambda adapter, module: ([[1.0, 2.0]], [[3.0], [4.0]]))
            out = self.styled()
            self.assertNotIn('error', out)
            self.assertEqual(out['style'], {'key': STYLE_KEY, 'name': 'Soul', 'scale': 1.0, 'tag': 'kdsoulr',
                                            'sha256': hashlib.sha256(weights).hexdigest()})
            self.assertEqual(self.last().params.caption, 'kdsoulr, Warm acoustic folk, brushed drums')
            self.assertIn(('job-1', 'Loading the style'), self.progress_log)
            self.assertEqual(len(self.s3.downloads), 3)
            self.assertEqual((Path(td) / 'soul-real-v1' / 'adapter_model.safetensors').read_bytes(), weights)
            self.assertEqual(self.weights(), self.pristine)
            self.progress_log.clear()
            again = self.styled(seed=3)                      # the next styled song on the same worker: nothing is fetched
            self.assertNotIn('error', again)
            self.assertEqual(len(self.s3.downloads), 3)
            self.assertNotIn(('job-1', 'Loading the style'), self.progress_log)
            self.assertIsInstance(out['timing']['style_fetch_s'], float)
            self.assertLess(again['timing']['style_fetch_s'], 1.0)
            self.s3.objects[f'{STYLE_KEY}/adapter_model.safetensors'] = b'X' * 3000      # a damaged copy in the bucket
            ace.STATE['styles'].clear()
            self.assertEqual(self.styled(), {'error': ace.ADAPTER_SENTENCE})
            self.assertEqual((self.weights(), ace.STATE['dirty']), (self.pristine, set()))

    def test_the_answer_says_the_style_its_strength_and_its_tag(self):
        out = self.styled(lora_scale=0.6)
        self.assertEqual(out['style']['scale'], 0.6)
        self.assertEqual(out['controls'], {'steps': 50, 'guidance_scale': 7.0, 'shift': 3.0, 'style': 'Test',
                                           'style_scale': 0.6, 'style_tag': 'kdtest'})
        self.assertIn('Made in the Test style at strength 0.6, with its trigger word "kdtest" ahead of the music '
                      'description.', out['worker_notes'])
        self.assertTrue(all(isinstance(note, str) and note.endswith('.') for note in out['worker_notes']))
        self.assertEqual(set(out['timing']), {'model_load_s', 'render_s', 'encode_s', 'upload_s', 'plan_s', 'style_fetch_s',
                                              'style_apply_s', 'style_restore_s'} - {'plan_s'})
        self.assertIn('lora', out['features'])
        self.assertIs(type(out['style']['scale']), float)

    def test_only_the_caption_and_the_language_differ_from_the_same_song_without_a_style(self):
        self.styled(seed=9)
        with_style = dict(vars(self.last().params))
        self.run_job(model='xl-sft', seed=9)
        without = dict(vars(self.last().params))
        different = {key for key in set(with_style) | set(without) if with_style.get(key) != without.get(key)}
        self.assertEqual(different, {'caption', 'vocal_language'})
        self.assertEqual((with_style['inference_steps'], with_style['guidance_scale'], with_style['shift']), (50, 7.0, 3.0))
        self.assertIs(with_style['dcw_enabled'], False)
        self.assertEqual((with_style['thinking'], with_style['use_cot_metas'], with_style['use_cot_caption'],
                          with_style['use_cot_language']), (True, True, False, False))

    def test_the_planner_is_shown_the_plain_caption_even_when_it_is_told_to_think_harder(self):
        out = self.styled(lm_temperature=1.3)
        self.assertEqual(self.planner_obj.asked[0]['temperature'], 1.3)
        self.assertEqual(self.planner_obj.asked[0]['caption'], 'Warm acoustic folk, brushed drums')
        self.assertEqual(out['controls']['lm_temperature'], 1.3)
        self.assertEqual(out['controls']['style'], 'Test')

    def test_without_the_planner_the_dit_still_gets_the_tagged_caption(self):
        out = self.styled(thinking=False)
        self.assertIsNone(self.last().llm)
        self.assertEqual(self.last().params.caption, 'kdtest, Warm acoustic folk, brushed drums')
        self.assertEqual(self.planner_obj.asked, [])
        self.planner.assert_not_called()
        self.assertIn('The planner was skipped.', out['worker_notes'])
        self.planner_ok = False
        out = self.styled()
        self.assertIsNone(self.last().llm)
        self.assertIn('The planner was not available, so the song was made without it.', out['worker_notes'])
        self.assertEqual(out['style']['tag'], 'kdtest')

    def test_a_long_caption_gives_way_to_the_tag_never_the_other_way(self):
        out = self.styled(caption='word ' * 200)
        plain = self.planner_obj.asked[0]['caption']
        tagged = self.last().params.caption
        self.assertLessEqual(len(plain), 512)
        self.assertLessEqual(len(tagged), 512)
        self.assertTrue(tagged.startswith('kdtest, word'))
        self.assertGreater(len(plain), len(tagged) - 8)                    # the planner's version is the longer one
        self.assertIn('The music description was shortened to fit 512 characters.', out['worker_notes'])
        self.assertIn('The music description was shortened to make room for the style.', out['worker_notes'])
        short = self.styled(caption='x' * 100)
        self.assertNotIn('The music description was shortened to make room for the style.', short['worker_notes'])

    def test_the_planner_environment_switch_cannot_drop_the_trigger_word(self):
        with mock.patch.dict(os.environ, {'ACE_PLANNER_REWRITES_CAPTION': '1'}):
            self.styled()
            self.assertIs(self.last().params.use_cot_caption, False)
            self.run_job(model='xl-sft')
            self.assertIs(self.last().params.use_cot_caption, True)         # unchanged for a plain song

    def test_the_fast_model_is_refused_before_anything_happens(self):
        out = self.run_job(lora_key=STYLE_KEY)
        self.assertEqual(out, {'error': ace.STYLE_MODEL_SENTENCE})
        self.assertEqual((self.fetches, self.api.calls, self.events), ([], [], []))
        self.engine.assert_not_called()

    def test_an_unknown_or_damaged_style_is_refused_before_the_model_loads(self):
        self.fetch_error = True
        out = self.styled()
        self.assertEqual(out, {'error': 'That style is not available right now.'})
        self.assertEqual(self.events, ['fetch'])            # not the model, not the planner, not the weights
        self.engine.assert_not_called()
        self.planner.assert_not_called()
        self.assertEqual(self.api.calls, [])
        self.assertEqual((self.weights(), ace.STATE['dirty']), (self.pristine, set()))

    def test_a_malformed_key_never_reaches_the_fetch(self):
        for key in ('k', 'ace-loras/../x', 'ace-loras/a/b'):
            self.assertEqual(self.styled(lora_key=key), {'error': ace.ADAPTER_SENTENCE})
        self.assertEqual(self.fetches, [])

    def test_a_model_that_cannot_load_leaves_the_weights_alone(self):
        self.engine.side_effect = ace.WorkerError('The best-quality music model could not be loaded.')
        self.assertEqual(self.styled(), {'error': 'The best-quality music model could not be loaded.'})
        self.assertEqual((self.weights(), ace.STATE['dirty'], self.api.calls), (self.pristine, set(), []))

    def test_a_handler_without_a_decoder_is_a_plain_refusal(self):
        self.dits['xl-sft'] = SimpleNamespace(model=object(), name='xl-sft')
        self.assertEqual(self.styled(), {'error': ace.ADAPTER_SENTENCE})
        self.assertEqual((self.api.calls, ace.STATE['dirty']), ([], set()))

    def test_a_strength_of_zero_is_a_plain_song_with_no_fetch_and_no_fold(self):
        out = self.styled(lora_scale=0)
        self.assertNotIn('style', out)
        self.assertNotIn('style', out['controls'])
        self.assertFalse(any(key.startswith('style') for key in out['timing']))
        self.assertIn('The style strength was 0, so the song was made without the style.', out['worker_notes'])
        self.assertEqual(self.fetches, [])
        self.assertIs(self.last().llm, self.planner_obj)                        # the real planner, not the wrapper
        self.assertEqual(self.last().params.caption, 'Warm acoustic folk, brushed drums')
        self.assertEqual(self.last().params.vocal_language, 'unknown')
        self.assertEqual(self.ops_made, [])
        self.assertEqual(self.api.seen[0].weights, self.pristine)

    def test_the_strength_reaches_the_fold_as_alpha_over_rank_times_the_strength(self):
        self.styled(lora_scale=0.6)
        self.assertEqual(len(self.folds()), 16)
        for factor in self.folds():
            self.assertAlmostEqual(factor, 2.0 * 0.6, places=12)

    def test_a_batch_is_one_fold_and_every_take_comes_back(self):
        out = self.styled(batch_size=3, seed=10)
        self.assertEqual([t['seed'] for t in out['takes']], [10, 11, 12])
        self.assertEqual(len(self.folds()), 16)
        self.assertEqual(self.last().config.batch_size, 3)
        self.assertEqual(self.weights(), self.pristine)

    def test_a_plain_song_never_builds_the_weight_tools(self):
        self.run_job(model='xl-sft')
        self.run_job()
        self.assertEqual(self.ops_made, [])
        self.assertNotIn('style', self.last().params.caption)

    def test_a_request_without_style_fields_is_the_request_it_was(self):
        first = self.run_job(model='xl-sft', seed=4)
        params = dict(vars(self.last().params))
        second = self.run_job(model='xl-sft', seed=4, lora_key=None, lora_scale=None)
        self.assertEqual(dict(vars(self.last().params)), params)
        self.assertEqual({k: v for k, v in first.items() if k not in ('takes', 'key', 'wav_key', 'url', 'wav_url',
                                                                      'processing_ms')},
                         {k: v for k, v in second.items() if k not in ('takes', 'key', 'wav_key', 'url', 'wav_url',
                                                                       'processing_ms')})
        self.assertNotIn('style', first)
        self.assertIs(self.last().llm, self.planner_obj)

    def test_the_order_of_events(self):
        self.styled()
        kinds = [event if isinstance(event, str) else event[0] for event in self.events]
        first = {kind: kinds.index(kind) for kind in set(kinds)}
        last_snapshot = max(i for i, kind in enumerate(kinds) if kind == 'snapshot')
        self.assertLess(first['fetch'], first['engine'])         # a style that cannot be had costs no model load
        self.assertLess(first['engine'], first['planner'])
        self.assertLess(first['planner'], first['snapshot'])
        self.assertLess(last_snapshot, first['fold'])            # every weight copied before the first is written
        self.assertLess(max(i for i, kind in enumerate(kinds) if kind == 'fold'), first['reset'])
        self.assertLess(first['reset'], first['generate'])
        self.assertLess(first['generate'], first['read'])        # the peak is read before the weights are put back
        self.assertLess(first['read'], first['put_back'])
        self.assertLess(max(i for i, kind in enumerate(kinds) if kind == 'put_back'), first['encode'])    # then encode

    def test_the_render_lock_is_held_only_while_rendering(self):
        self.styled()
        self.assertIs(self.api.seen[-1].locked, True)
        self.assertIs(ace.RENDER_LOCK.locked(), False)
        self.run_job(model='xl-sft')
        self.assertIs(self.api.seen[-1].locked, True)           # plain songs wait their turn as well
        self.api = StyledApi(self.decoder, self.events, raises=RuntimeError('boom'))
        self.styled()
        self.assertIs(ace.RENDER_LOCK.locked(), False)
        self.api = StyledApi(self.decoder, self.events, raises=KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.styled()
        self.assertIs(ace.RENDER_LOCK.locked(), False)

    def test_probe_and_validation_do_not_wait_for_the_lock(self):
        with ace.RENDER_LOCK:
            self.assertEqual(self.styled(lora_key='k'), {'error': ace.ADAPTER_SENTENCE})
            out = silently(ace.handler, {'input': {'mode': 'probe'}})
            self.assertEqual(out['mode'], 'probe')

    def test_the_probe_lists_the_style_feature_and_is_not_bothered_by_a_key(self):
        out = silently(ace.handler, {'input': {'mode': 'probe', 'lora_key': STYLE_KEY}})
        self.assertEqual(out['mode'], 'probe')
        self.assertEqual(out['features'][-1], 'lora')
        self.assertEqual(self.fetches, [])

    def test_the_image_build_check_for_a_refused_style_still_holds(self):
        # Dockerfile.ace check 3 sends lora_key "k" (the fast model by default) and expects exactly this sentence
        out = silently(ace.handler, {'input': {'caption': 'x', 'duration': 60, 'lora_key': 'k'}})
        self.assertEqual(out, {'error': ace.ADAPTER_SENTENCE})
        dockerfile = Path(ace.__file__).with_name('Dockerfile.ace')
        if dockerfile.is_file():
            text = dockerfile.read_text(encoding='utf-8')
            self.assertIn("'lora_key': 'k'}}) == {'error': a.ADAPTER_SENTENCE}", text)

    def test_logs_never_carry_the_callers_words_or_the_tagged_caption(self):
        long_secret = SECRET_CAPTION + ' ' + ('more words ' * 60)
        for caption in (SECRET_CAPTION, long_secret):
            plain = ace.request_input(styled(caption=caption))['caption']
            tagged, _ = ace.tagged_caption('kdtest', plain)
            self.api = StyledApi(self.decoder, self.events, success=False,
                                 error=f'Bad input: {tagged} / {SECRET_LYRICS}')
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                out = ace.handler({'id': 'j', 'input': styled(caption=caption, lyrics=SECRET_LYRICS)})
            everything = buffer.getvalue() + json.dumps(out)
            self.assertNotIn('ZEBRA', everything)
            self.assertNotIn('KANGAROO', everything)
            self.assertEqual(out, {'error': ace.FAILED})
            self.assertEqual(self.weights(), self.pristine)

    def test_progress_says_what_a_styled_song_is_doing(self):
        self.styled()
        self.assertIn(('job-1', 'Planning your song'), self.progress_log)
        self.assertIn(('job-1', 'Rendering your song'), self.progress_log)


class TestAfterAStyledSong(StyledHarness):
    """The song after a styled one, styled or plain, runs on exactly the original weights whatever happened before."""

    def reset_case(self):
        self.ops.fold_fails_at, self.ops.folds = None, 0
        self.ops.put_back_fails = self.ops.put_back_garbles = 0
        self.s3.fail = False
        self.planner_obj = self.llm = FakePlanner()
        self.api = StyledApi(self.decoder, self.events)
        self.events.clear()

    def check_next_songs(self, label):
        self.assertEqual(self.weights(), self.pristine, label)
        self.assertEqual(ace.STATE['dirty'], set(), label)
        self.assertIs(ace.RENDER_LOCK.locked(), False, label)
        self.reset_case()
        out = self.run_job(model='xl-sft')
        self.assertNotIn('error', out, label)
        self.assertEqual(self.api.seen[0].weights, self.pristine, label)
        self.assertEqual(self.last().params.caption, 'Warm acoustic folk, brushed drums', label)
        self.assertIs(self.last().llm, self.planner_obj, label)
        self.assertNotIn('style', out, label)
        self.reset_case()
        again = self.styled()                              # and a styled one after it works again
        self.assertEqual(again['style']['key'], STYLE_KEY, label)
        self.assertEqual(self.weights(), self.pristine, label)

    def test_after_a_good_song(self):
        self.styled()
        self.check_next_songs('good')

    def test_after_upstream_reported_a_failure(self):
        self.api = StyledApi(self.decoder, self.events, success=False, error='Insufficient free VRAM: need 11 GB')
        self.assertEqual(self.styled(), {'error': ace.TOO_BIG})
        self.check_next_songs('upstream failure')

    def test_after_an_exception_in_the_render(self):
        self.api = StyledApi(self.decoder, self.events, raises=RuntimeError('boom'))
        self.assertEqual(self.styled(), {'error': 'ACE-Step could not finish (RuntimeError). Your writing is saved.'})
        self.check_next_songs('exception')

    def test_after_an_out_of_memory_in_the_render(self):
        class OutOfMemoryError(RuntimeError):
            pass
        self.api = StyledApi(self.decoder, self.events, raises=OutOfMemoryError('CUDA out of memory'))
        self.assertEqual(self.styled(), {'error': ace.TOO_BIG})
        self.check_next_songs('out of memory')

    def test_after_a_cancellation(self):
        for error in (KeyboardInterrupt(), SystemExit(1), asyncio.CancelledError()):
            self.reset_case()
            self.api = StyledApi(self.decoder, self.events, raises=error)
            with self.assertRaises(type(error)):
                self.styled()
            self.check_next_songs(type(error).__name__)

    def test_after_the_planner_died_inside_the_render(self):
        self.planner_obj.generate_with_stop_condition = mock.Mock(side_effect=RuntimeError('planner'))
        self.assertEqual(self.styled(), {'error': 'ACE-Step could not finish (RuntimeError). Your writing is saved.'})
        self.check_next_songs('planner')

    def test_after_the_fold_itself_failed_half_way(self):
        self.ops.fold_fails_at = 3
        self.assertEqual(self.styled(), {'error': 'ACE-Step could not finish (RuntimeError). Your writing is saved.'})
        self.assertEqual(self.api.calls, [])             # it never got as far as rendering
        self.check_next_songs('fold')

    def test_after_the_upload_failed_the_weights_were_already_back(self):
        self.s3.fail = True
        self.assertEqual(self.styled(), {'error': 'ACE-Step could not finish (OSError). Your writing is saved.'})
        self.check_next_songs('upload')

    def test_after_the_encoder_failed(self):
        with mock.patch.object(ace, 'encode_take', side_effect=OSError('ffmpeg')):
            self.assertEqual(self.styled(), {'error': 'ACE-Step could not finish (OSError). Your writing is saved.'})
        self.check_next_songs('encode')

    def test_a_restore_that_cannot_be_proven_marks_the_model_and_the_answer_says_so(self):
        self.ops.put_back_fails = 10 ** 6
        out = self.styled()
        self.assertNotIn('error', out)                                   # the song itself is fine
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertIn('The music model will be reloaded before the next song.', out['worker_notes'])
        self.assertIs(ace.RENDER_LOCK.locked(), False)

    def test_a_clean_restore_adds_no_such_note(self):
        out = self.styled()
        self.assertNotIn('The music model will be reloaded before the next song.', out['worker_notes'])

    def test_a_dirty_model_is_not_folded_again_but_the_real_engine_reloads_it_first(self):
        ace.STATE['dirty'].add('xl-sft')            # what a failed restore leaves behind; engine() is mocked in this harness
        out = self.styled()
        self.assertEqual(out, {'error': ace.ADAPTER_SENTENCE})
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertEqual(self.api.calls, [])


# ---------------------------------------------------------------- trained styles: the real arithmetic on real tensors

HAVE_TORCH = importlib.util.find_spec('torch') is not None
HAVE_SAFETENSORS = HAVE_TORCH and importlib.util.find_spec('safetensors') is not None
HAVE_PEFT = HAVE_TORCH and importlib.util.find_spec('peft') is not None
REAL_ADAPTER = Path(os.environ.get('ACE_TEST_ADAPTER_DIR', 'F:/KadeAI-work/ace/lora/results/adapters_unpacked/adapters/final'))


def torch_model(dtype, layers=2, hidden=8, query=8, kv=4, grad=True):
    """A stand-in for upstream's model: model.decoder.layers[N].self_attn|cross_attn.q|k|v|o_proj, model.encoder with an
    attention of its own (a lyric encoder has q_proj too), and weights the adapter must leave alone."""
    import torch
    from torch import nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, query, bias=False)
            self.k_proj = nn.Linear(hidden, kv, bias=False)
            self.v_proj = nn.Linear(hidden, kv, bias=False)
            self.o_proj = nn.Linear(query, hidden, bias=False)
            self.q_norm = nn.LayerNorm(query)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.cross_attn = Attn(), Attn()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Layer() for _ in range(layers))
            self.proj_in = nn.Conv1d(hidden, hidden, 2)
            self.condition_embedder = nn.Linear(hidden, hidden)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Layer() for _ in range(1))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder, self.encoder = Decoder(), Encoder()
    torch.manual_seed(11)
    model = Model().to(dtype)
    for param in model.parameters():
        param.requires_grad_(grad)
    return model


def write_torch_adapter(folder, model, rank=3, alpha=6.0, dtype=None, seed=5):
    """A PEFT-shaped adapter for model.decoder in `folder`; returns the Adapter the worker would hold for it."""
    import torch
    from safetensors.torch import save_file
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for name, module in model.decoder.named_modules():
        if name.endswith(ace.PROJECTIONS):
            out, inner = module.weight.shape
            a = torch.randn(rank, inner, generator=generator) * 0.3
            b = torch.randn(out, rank, generator=generator) * 0.3
            tensors[f'base_model.model.{name}.lora_A.weight'] = a.to(dtype) if dtype else a
            tensors[f'base_model.model.{name}.lora_B.weight'] = b.to(dtype) if dtype else b
    path = Path(folder) / 'adapter_model.safetensors'
    save_file(tensors, str(path))
    modules = ace.adapter_modules(path, rank)
    return ace.Adapter('ace-loras/torch-v1', Path(folder), 'b' * 64, 'kdtorch', 'Torch', rank, alpha, modules,
                       path.stat().st_size), tensors


def reference_weights(model, tensors, alpha, rank, scale):
    """PEFT's formula, W + (alpha / r) * scale * (B @ A), in float64: the independent answer."""
    import torch
    want = {}
    for name, module in model.decoder.named_modules():
        key = f'base_model.model.{name}.lora_A.weight'
        if key in tensors:
            a = tensors[key].double()
            b = tensors[f'base_model.model.{name}.lora_B.weight'].double()
            want[name + '.weight'] = module.weight.detach().double() + (alpha / rank) * scale * (b @ a)
    return want


def bits(tensor):
    import torch
    return tensor.detach().reshape(-1).view(torch.int16 if tensor.element_size() == 2 else torch.int32).clone()


@unittest.skipUnless(HAVE_SAFETENSORS, 'needs torch and safetensors')
class TestFoldOnTorch(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def build(self, dtype=None, adapter_dtype=None, grad=True, **kwargs):
        import torch
        self.model = torch_model(dtype or torch.bfloat16, grad=grad)
        self.adapter, self.tensors = write_torch_adapter(self.tmp.name, self.model, dtype=adapter_dtype, **kwargs)
        self.handler = SimpleNamespace(model=self.model)
        self.before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def scope(self, scale=1.0):
        return ace.apply_adapter(self.handler, self.adapter, scale, 'xl-sft')

    def unchanged(self, names=None):
        import torch
        for key, value in self.model.state_dict().items():
            if names is None or key in names:
                self.assertTrue(torch.equal(bits(value), bits(self.before[key])), key)

    def test_fp32_weights_match_the_reference_formula(self):
        import torch
        self.build(torch.float32)
        for scale in (1.0, 0.6, 1.5, 0.25):
            want = reference_weights(self.model, self.tensors, 6.0, 3, scale)
            with quiet(), self.scope(scale):
                for name, expected in want.items():
                    got = self.model.decoder.get_parameter(name).detach()
                    self.assertEqual(got.shape, expected.shape)
                    self.assertTrue(torch.allclose(got.double(), expected, rtol=1e-5, atol=1e-6), (scale, name))
            self.unchanged()

    def test_bf16_weights_are_the_reference_rounded_to_bf16(self):
        import torch
        self.build(torch.bfloat16)
        for scale in (1.0, 0.6, 1.5):
            want = reference_weights(self.model, self.tensors, 6.0, 3, scale)
            with quiet(), self.scope(scale):
                for name, expected in want.items():
                    got = self.model.decoder.get_parameter(name).detach()
                    self.assertIs(got.dtype, torch.bfloat16)
                    target = expected.to(torch.bfloat16)
                    # fp32 against fp64 can differ by one rounding step on a value that sits on a boundary; no more
                    self.assertTrue(((got.double() - target.double()).abs() <= 2 ** -7 * target.double().abs() + 1e-30).all(),
                                    (scale, name))
                    self.assertGreater(float((got == target).double().mean()), 0.99, (scale, name))
                    self.assertFalse(torch.equal(got, self.before['decoder.' + name]), name)    # it really changed it
            self.unchanged()

    def test_fp16_weights_work_too(self):
        import torch
        self.build(torch.float16)
        want = reference_weights(self.model, self.tensors, 6.0, 3, 1.0)
        with quiet(), self.scope():
            for name, expected in want.items():
                got = self.model.decoder.get_parameter(name).detach()
                self.assertIs(got.dtype, torch.float16)
                self.assertTrue(torch.allclose(got.double(), expected, rtol=2e-3, atol=2e-3), name)
        self.unchanged()

    def test_a_half_precision_adapter_file_is_read_in_fp32(self):
        import torch
        for adapter_dtype in (torch.bfloat16, torch.float16):
            self.build(torch.bfloat16, adapter_dtype=adapter_dtype)
            self.assertIs(next(iter(self.tensors.values())).dtype, adapter_dtype)
            want = reference_weights(self.model, self.tensors, 6.0, 3, 1.0)
            with quiet(), self.scope():
                for name, expected in want.items():
                    got = self.model.decoder.get_parameter(name).detach()
                    self.assertTrue(((got.double() - expected.to(torch.bfloat16).double()).abs()
                                     <= 2 ** -7 * expected.abs() + 1e-30).all(), name)
            self.unchanged()

    def test_parameters_keep_their_identity_dtype_and_flags(self):
        import torch
        self.build(torch.bfloat16)
        objects = {name: param for name, param in self.model.decoder.named_parameters()}
        with quiet(), self.scope():
            for name, param in self.model.decoder.named_parameters():
                self.assertIs(param, objects[name], name)
                self.assertTrue(param.requires_grad, name)
                self.assertFalse(param.is_inference(), name)
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())
        self.unchanged()

    def test_only_the_decoders_projection_weights_change(self):
        import torch
        self.build(torch.bfloat16)
        targets = {module + '.weight' for module in self.adapter.modules}
        self.assertEqual(len(targets), 2 * 2 * 4)                    # 2 layers x self/cross x q, k, v, o
        with quiet(), self.scope():
            state = self.model.state_dict()
            for key, value in state.items():
                same = torch.equal(bits(value), bits(self.before[key]))
                self.assertEqual(same, key not in {'decoder.' + t for t in targets}, key)

    def test_restore_is_bit_for_bit_after_success_exception_and_cancellation(self):
        import torch
        self.build(torch.bfloat16)
        with quiet(), self.scope() as fold:
            pass
        self.assertIs(fold.restored, True)
        self.unchanged()
        for error in (RuntimeError('boom'), KeyboardInterrupt(), SystemExit(1), asyncio.CancelledError()):
            with quiet(), self.assertRaises(type(error)):
                with self.scope():
                    raise error
            self.unchanged()
            self.assertEqual(ace.STATE['dirty'], set())

    def test_works_when_the_parameters_do_not_require_grad(self):
        import torch
        self.build(torch.bfloat16, grad=False)
        with quiet(), self.scope():
            self.assertFalse(torch.equal(bits(self.model.decoder.layers[0].self_attn.q_proj.weight),
                                         bits(self.before['decoder.layers.0.self_attn.q_proj.weight'])))
        self.unchanged()

    def test_a_plain_in_place_write_would_have_failed_which_is_why_inference_mode_is_used(self):
        import torch
        self.build(torch.bfloat16)
        weight = self.model.decoder.layers[0].self_attn.q_proj.weight
        with self.assertRaises(RuntimeError):
            weight.copy_(torch.zeros_like(weight))            # upstream never freezes the decoder: a plain write raises

    def test_works_when_the_parameters_are_inference_tensors(self):
        import torch
        with torch.inference_mode():
            self.model = torch_model(torch.bfloat16)
        self.assertTrue(self.model.decoder.layers[0].self_attn.q_proj.weight.is_inference())
        self.adapter, self.tensors = write_torch_adapter(self.tmp.name, self.model)
        self.handler = SimpleNamespace(model=self.model)
        self.before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        want = reference_weights(self.model, self.tensors, 6.0, 3, 1.0)
        with quiet(), self.scope():
            changed = [name for name in want if not torch.equal(self.model.decoder.get_parameter(name).detach(), self.before['decoder.' + name])]
            self.assertEqual(len(changed), len(want))
        self.unchanged()

    def test_a_weight_in_an_unsupported_number_format_is_refused_and_nothing_is_written(self):
        import torch
        from torch import nn
        self.build(torch.bfloat16)
        self.model.decoder.layers[1].cross_attn.o_proj.weight = nn.Parameter(
            torch.zeros(8, 8, dtype=torch.int8), requires_grad=False)           # like a quantised decoder
        before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        with quiet(), self.assertRaises(ace.WorkerError) as caught:
            with self.scope():
                self.fail('the block must not run')
        self.assertEqual(str(caught.exception), ace.ADAPTER_SENTENCE)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertEqual(ace.STATE['dirty'], set())

    def test_a_decoder_that_does_not_fit_the_adapter_is_refused(self):
        import torch
        self.build(torch.bfloat16)
        other = torch_model(torch.bfloat16, hidden=10)             # same names, other widths
        with quiet(), self.assertRaises(ace.WorkerError):
            with ace.apply_adapter(SimpleNamespace(model=other), self.adapter, 1.0, 'xl-sft'):
                self.fail('the block must not run')
        fewer = torch_model(torch.bfloat16, layers=1)               # the adapter names a layer this decoder lacks
        with quiet(), self.assertRaises(ace.WorkerError):
            with ace.apply_adapter(SimpleNamespace(model=fewer), self.adapter, 1.0, 'xl-sft'):
                self.fail('the block must not run')
        self.assertEqual(ace.STATE['dirty'], set())

    def test_the_checksum_sees_one_flipped_bit_a_swap_and_a_sign(self):
        import torch
        ops = ace.TorchOps()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            weight = torch.nn.Parameter(torch.randn(16, 12).to(dtype))
            alias = weight.detach()                                    # same storage, no autograd in the way
            raw = alias.view(-1).view(torch.int16 if weight.element_size() == 2 else torch.int32)
            base = ops.checksum(weight)
            self.assertEqual(ops.checksum(weight), base)
            raw[37] ^= 1                                               # the lowest mantissa bit of one number
            self.assertNotEqual(ops.checksum(weight), base, dtype)
            raw[37] ^= 1
            self.assertEqual(ops.checksum(weight), base, dtype)
            alias[2, 3], alias[7, 5] = alias[7, 5].clone(), alias[2, 3].clone()           # two numbers trade places
            self.assertNotEqual(ops.checksum(weight), base, dtype)
            alias[2, 3], alias[7, 5] = alias[7, 5].clone(), alias[2, 3].clone()
            self.assertEqual(ops.checksum(weight), base, dtype)
            alias[0, 0] = -alias[0, 0]                                 # a sign
            self.assertNotEqual(ops.checksum(weight), base, dtype)

    def test_the_snapshot_is_a_copy_and_the_ops_leave_no_grad_or_mode_behind(self):
        import torch
        ops = ace.TorchOps()
        weight = torch.nn.Parameter(torch.randn(4, 4).to(torch.bfloat16))
        with ops.mode():
            saved = ops.snapshot(weight)
            weight.copy_(torch.zeros(4, 4))
        self.assertFalse(torch.equal(saved, weight.detach()))
        self.assertEqual(ops.nbytes(saved), 4 * 4 * 2)
        self.assertEqual(saved.device.type, 'cpu')
        self.assertFalse(torch.is_inference_mode_enabled())
        self.assertTrue(ops.supported(weight))
        self.assertFalse(ops.supported(torch.nn.Parameter(torch.zeros(4), requires_grad=False)))      # a bias is not a weight
        self.assertFalse(ops.supported(torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)))

    def test_a_scale_of_zero_point_zero_one_barely_moves_a_bf16_weight_but_restores_it_exactly(self):
        import torch
        self.build(torch.bfloat16)
        with quiet(), self.scope(0.01):
            pass
        self.unchanged()

    def test_the_log_line_of_a_real_fold_names_the_weights_and_the_memory(self):
        import torch
        self.build(torch.bfloat16)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.scope():
            pass
        self.assertIn('folded into 16 weights at strength 1', buffer.getvalue())
        self.assertIn('GiB held on the CPU', buffer.getvalue())


@unittest.skipUnless(HAVE_PEFT and HAVE_SAFETENSORS, 'needs torch, safetensors and peft')
class TestFoldAgainstPeft(unittest.TestCase):
    """PEFT's own merge is the independent reference: the same adapter, the same scale, merged by PEFT."""

    def test_the_fold_equals_peft_merge_and_unload(self):
        import copy
        import torch
        from peft import LoraConfig, get_peft_model
        for dtype in (torch.float32, torch.bfloat16):
            for scale in (1.0, 0.6, 1.5):
                with tempfile.TemporaryDirectory() as td:
                    model = torch_model(dtype)
                    adapter, tensors = write_torch_adapter(td, model, rank=3, alpha=6.0)
                    donor = get_peft_model(copy.deepcopy(model.decoder).double(), LoraConfig(
                        r=3, lora_alpha=6, target_modules=list(ace.PROJECTIONS), lora_dropout=0.0, bias='none'))
                    renamed = {key.replace('base_model.model.', '', 1).replace('.lora_A.weight', '.lora_A.default.weight')
                               .replace('.lora_B.weight', '.lora_B.default.weight'): value.double()
                               for key, value in tensors.items()}
                    prefixed = {f'base_model.model.{key}': value for key, value in renamed.items()}
                    result = donor.load_state_dict(prefixed, strict=False)
                    self.assertEqual(result.unexpected_keys, [])
                    for module in donor.modules():
                        if isinstance(getattr(module, 'scaling', None), dict):
                            module.scaling['default'] *= scale          # upstream's set_lora_scale multiplies the same number
                    merged = donor.merge_and_unload()
                    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    with quiet(), ace.apply_adapter(SimpleNamespace(model=model), adapter, scale, 'xl-sft'):
                        for name, tensor in merged.state_dict().items():
                            got = model.decoder.state_dict()[name]
                            want = tensor.to(dtype)
                            if dtype is torch.float32:
                                self.assertTrue(torch.allclose(got, want, rtol=1e-5, atol=1e-6), (dtype, scale, name))
                            else:
                                self.assertTrue(((got.double() - want.double()).abs() <= 2 ** -7 * want.double().abs() + 1e-30).all(),
                                                (dtype, scale, name))
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(bits(value), bits(before[key])), key)


@unittest.skipUnless(HAVE_SAFETENSORS, 'needs torch and safetensors')
class TestAdapterFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'adapter_model.safetensors'

    def save(self, tensors, rank=2):
        from safetensors.torch import save_file
        save_file(tensors, str(self.path))
        return ace.adapter_modules(self.path, rank)

    def pair(self, module='layers.0.self_attn.q_proj', rank=2, out=4, inner=3, dtype=None):
        import torch
        return {f'base_model.model.{module}.lora_A.weight': torch.ones(rank, inner, dtype=dtype or torch.float32),
                f'base_model.model.{module}.lora_B.weight': torch.ones(out, rank, dtype=dtype or torch.float32)}

    def refused(self, tensors, rank=2):
        with self.assertRaises(ace.StyleProblem):
            self.save(tensors, rank)

    def test_a_pair_is_read_by_name_with_its_shapes_and_its_tensors(self):
        import torch
        tensors = {**self.pair(), **self.pair('layers.1.cross_attn.o_proj', out=3, inner=4)}
        modules = self.save(tensors)
        self.assertEqual(modules, {'layers.0.self_attn.q_proj': ((2, 3), (4, 2)),
                                   'layers.1.cross_attn.o_proj': ((2, 4), (3, 2))})
        adapter = ace.Adapter('ace-loras/t', Path(self.tmp.name), 'c' * 64, 't', 'T', 2, 4.0, modules, 1)
        a, b = adapter.pair('layers.1.cross_attn.o_proj')
        self.assertEqual((tuple(a.shape), tuple(b.shape)), ((2, 4), (3, 2)))
        self.assertTrue(torch.equal(a, tensors['base_model.model.layers.1.cross_attn.o_proj.lora_A.weight']))

    def test_names_this_worker_does_not_fold_are_refused(self):
        import torch
        extra = torch.ones(2, 3)
        for name in ('base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight',
                     'base_model.model.layers.0.mlp.gate_proj.lora_A.weight',
                     'base_model.model.layers.0.self_attn.q_proj.lora_embedding_A',
                     'base_model.model.layers.0.self_attn.q_proj.lora_magnitude_vector',
                     'base_model.model.layers.0.self_attn.q_proj.bias',
                     'layers.0.self_attn.q_proj.lora_A.weight',
                     'base_model.model.encoder.layers.0.self_attn.q_proj.lora_A.weight',
                     'base_model.model.layers.x.self_attn.q_proj.lora_A.weight',
                     'base_model.model.layers.0.self_attn.q_proj.lora_C.weight',
                     'base_model.model.layers.0.self_attn.q_proj.lora_A.weight.extra',
                     'base_model.model.layers.0.other_attn.q_proj.lora_A.weight',
                     'base_model.model.layers.0.self_attn.q_norm.lora_A.weight'):
            self.refused({**self.pair(), name: extra})

    def test_half_a_pair_an_empty_file_and_a_wrong_rank_are_refused(self):
        import torch
        half = self.pair()
        del half['base_model.model.layers.0.self_attn.q_proj.lora_B.weight']
        self.refused(half)
        self.refused({'unrelated': torch.ones(1)})
        self.refused(self.pair(rank=3))                                   # the settings say rank 2
        self.refused(self.pair(), rank=3)
        wrong = self.pair()
        wrong['base_model.model.layers.0.self_attn.q_proj.lora_B.weight'] = torch.ones(4, 5)      # B is out x r
        self.refused(wrong)
        flat = self.pair()
        flat['base_model.model.layers.0.self_attn.q_proj.lora_A.weight'] = torch.ones(6)
        self.refused(flat)

    def test_number_formats(self):
        import torch
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.assertEqual(len(self.save(self.pair(dtype=dtype))), 1, dtype)
        for dtype in (torch.int8, torch.int32, torch.float64, torch.uint8, torch.bool):
            self.refused(self.pair(dtype=dtype))

    def test_an_unreadable_file_raises_but_never_inside_a_render(self):
        self.path.write_bytes(b'not a safetensors file')
        with self.assertRaises(Exception):
            ace.adapter_modules(self.path, 2)              # fetch_adapter turns this into the plain refusal

    @unittest.skipUnless((REAL_ADAPTER / 'adapter_model.safetensors').is_file(), 'the trained adapter is not on this machine')
    def test_the_real_adapter_matches_the_real_decoder_layout(self):
        import torch
        from torch import nn
        modules = ace.adapter_modules(REAL_ADAPTER / 'adapter_model.safetensors', 64)
        self.assertEqual(len(modules), 32 * 2 * 4)
        self.assertEqual(ace.read_adapter_config(REAL_ADAPTER / 'adapter_config.json'), (64, 128.0))
        shapes = {}
        for module, (a, b) in modules.items():
            shapes.setdefault(module.rsplit('.', 1)[1], set()).add((a, b))
        self.assertEqual(shapes, {'q_proj': {((64, 2560), (4096, 64))}, 'k_proj': {((64, 2560), (1024, 64))},
                                  'v_proj': {((64, 2560), (1024, 64))}, 'o_proj': {((64, 4096), (2560, 64))}})
        # a decoder with the XL widths on the meta device (no memory): every pair finds its weight, with the right shape
        hidden, query, kv = 2560, 4096, 1024

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(hidden, query, bias=False)
                self.k_proj = nn.Linear(hidden, kv, bias=False)
                self.v_proj = nn.Linear(hidden, kv, bias=False)
                self.o_proj = nn.Linear(query, hidden, bias=False)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn, self.cross_attn = Attn(), Attn()

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(Layer() for _ in range(32))
        with torch.device('meta'):
            decoder = Decoder().to(torch.bfloat16)
        adapter = ace.Adapter('ace-loras/real', REAL_ADAPTER, 'd' * 64, 'kdsoulr', 'Soul', 64, 128.0, modules, 1)
        fold = ace.StyleFold(SimpleNamespace(model=SimpleNamespace(decoder=decoder)), adapter, 1.0, 'xl-sft')
        fold.resolve()
        self.assertEqual(len(fold.params), 256)
        held = sum(p.numel() * 2 for p in fold.params.values())
        self.assertEqual(held, 3_355_443_200)               # bf16 bytes the pristine copy takes on the CPU
        self.assertEqual(adapter.scaling, 2.0)


class TestProbe(Harness):
    def test_probe_resolves_models_and_does_no_gpu_work(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {'HF_HUB_CACHE': td}):
            with mock.patch.object(ace, 'CACHE_ROOT', os.path.join(td, 'none')), \
                    mock.patch.object(ace, 'DOWNLOAD_CACHE', os.path.join(td, 'none2')):
                make_dit(td, 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40)
                out = silently(ace.handler, {'input': {'mode': 'probe'}})
        self.assertEqual((out['engine'], out['mode']), ('ace', 'probe'))
        self.assertEqual(set(out['models']), {'turbo', 'sft', 'main'})
        self.assertEqual(out['models']['turbo']['source'], 'cached ' + 'a' * 12)
        self.assertEqual(out['models']['turbo']['revision'], 'a' * 40)
        self.assertEqual(out['models']['turbo']['repo'], 'ACE-Step/acestep-v15-xl-turbo')
        self.assertEqual(out['models']['sft'], {'repo': 'ACE-Step/acestep-v15-xl-sft', 'source': 'missing', 'revision': None})
        self.assertEqual(out['features'], list(ace.FEATURES))
        self.assertEqual(set(out['versions']), {'ace_step_commit', 'torch', 'worker'})
        self.assertEqual(out['gpu'], 'Fake A40')
        self.assertEqual(out['loaded'], {'xl-turbo': False, 'xl-sft': False, 'planner': False})
        self.engine.assert_not_called()
        self.planner.assert_not_called()
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.s3.uploads, [])

    def test_probe_never_downloads(self):
        with mock.patch.object(ace, 'fetch_model') as fetch:
            silently(ace.handler, {'input': {'mode': 'probe'}})
        fetch.assert_not_called()

    def test_probe_needs_no_caption_or_length(self):
        out = silently(ace.handler, {'input': {'mode': 'probe'}})
        self.assertNotIn('error', out)

    def test_probe_paths_are_not_in_the_answer(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {'HF_HUB_CACHE': td}):
            make_dit(td, 'ACE-Step/acestep-v15-xl-turbo', 'b' * 40)
            out = silently(ace.handler, {'input': {'mode': 'probe'}})
        self.assertNotIn(td, json.dumps(out))

    def test_unknown_mode(self):
        self.assertEqual(self.run_job(mode='warm'), {'error': 'Mode must be render or probe.'})


class TestProgress(unittest.TestCase):
    def test_phases_map_to_plain_phrases_once(self):
        sent = []
        progress = ace.make_progress({'id': 'x'}, sent.append)
        progress(0.1, 'Phase 1: Generating CoT metadata (once for all items)...')
        progress(0.12, 'Phase 1: Generating CoT metadata (once for all items)...')
        progress(0.5, 'Phase 2: Generating audio codes for 1 items...')
        progress(0.51, desc='Preparing inputs...')
        progress(0.6, desc='Generating music (steps 3/8)')
        progress(0.99, desc='Preparing audio data...')
        progress()
        self.assertEqual(sent, ['Planning your song', 'Writing the musical ideas', 'Rendering your song',
                                'Finishing the audio'])

    def test_a_failing_sender_never_breaks_the_render(self):
        def broken(text):
            raise RuntimeError('network')
        progress = ace.make_progress({'id': 'x'}, broken)
        progress(0.1, 'Phase 1: x')
        progress(0.1, desc='Phase 2: y')


# ---------------------------------------------------------------- audio encoding

class TestEncoding(unittest.TestCase):
    def test_ffmpeg_command_is_yues(self):
        with mock.patch.object(ace.subprocess, 'run') as run:
            ace.ffmpeg('-i', Path('a.wav'), '-c:a', 'libmp3lame', '-b:a', '320k', Path('b.mp3'))
        run.assert_called_once_with(['ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', 'a.wav', '-c:a', 'libmp3lame',
                                     '-b:a', '320k', 'b.mp3'], check=True, timeout=180, capture_output=True)

    def test_encode_take_makes_24_bit_wav_then_320k_mp3(self):
        log = []
        with tempfile.TemporaryDirectory() as td, mock.patch.object(ace, 'ffmpeg', fake_ffmpeg_factory(log)), \
                mock.patch.object(ace, 'write_float_wav', fake_write_wav):
            folder = Path(td)
            wav, mp3, seconds = ace.encode_take({'tensor': FakeTensor(12.5), 'sample_rate': 48000}, folder)
            self.assertEqual((wav.name, mp3.name), ('master.wav', 'master.mp3'))
            self.assertTrue(wav.exists() and mp3.exists())
            self.assertFalse((folder / 'audio.wav').exists())     # the float intermediate is removed
        self.assertEqual(seconds, 12.5)
        self.assertEqual(log[0], ('-i', folder / 'audio.wav', '-c:a', 'pcm_s24le', folder / 'master.wav'))
        self.assertEqual(log[1], ('-i', folder / 'master.wav', '-c:a', 'libmp3lame', '-b:a', '320k',
                                  folder / 'master.mp3'))

    def test_float_wav_is_written_samples_first(self):
        class Array:
            def __init__(self, shape, tag='raw'):
                self.shape, self.ndim, self.tag = shape, len(shape), tag

            @property
            def T(self):
                return Array(self.shape[::-1], 'transposed')

            def __getitem__(self, key):
                return Array((self.shape[0], 1), 'column')

        class Tensor:
            def __init__(self, shape):
                self.array = Array(shape)

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return self.array

        written = []
        fake = SimpleNamespace(write=lambda path, data, rate, subtype=None: written.append((path, data, rate, subtype)))
        with mock.patch.dict(sys.modules, {'soundfile': fake}):
            ace.write_float_wav(Path('x.wav'), Tensor((2, 96000)), 48000)
            ace.write_float_wav(Path('y.wav'), Tensor((96000,)), 48000)
        self.assertEqual((written[0][1].shape, written[0][2], written[0][3]), ((96000, 2), 48000, 'FLOAT'))
        self.assertEqual(written[1][1].shape, (96000, 1))


# ---------------------------------------------------------------- the model cache

def make_dit(root, repo, sha, shards=('model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'),
             skip=(), age=None):
    folder = Path(root) / ('models--' + repo.replace('/', '--')) / 'snapshots' / sha
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text('{}')
    (folder / 'silence_latent.pt').write_bytes(b'x')
    (folder / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {f'layer{i}': name for i, name in enumerate(shards)}}))
    for name in shards:
        if name not in skip:
            (folder / name).write_bytes(b'w')
    (folder / 'modeling_from_the_hub.py').write_text('# the hub copy, never used\n')
    if age is not None:
        os.utime(folder, (age, age))
    return folder


def make_main(root, sha, skip=(), age=None):
    folder = Path(root) / 'models--ACE-Step--Ace-Step1.5' / 'snapshots' / sha
    for relative in ace.MAIN_FILES:
        if relative not in skip:
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'm')
    folder.mkdir(parents=True, exist_ok=True)
    if age is not None:
        os.utime(folder, (age, age))
    return folder


class CacheCase(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'hub'
        self.root.mkdir()
        for name, value in (('CACHE_ROOT', str(Path(self.tmp.name) / 'no-volume')),
                            ('DOWNLOAD_CACHE', str(Path(self.tmp.name) / 'no-download'))):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        for env in ('ACE_TURBO_DIR', 'ACE_SFT_DIR', 'ACE_MAIN_DIR'):
            os.environ.pop(env, None)


class TestReadiness(CacheCase):
    def test_dit_ready(self):
        folder = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40)
        self.assertTrue(ace.dit_ready(folder))

    def test_dit_missing_a_shard_or_the_silence_latent_or_config(self):
        folder = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'b' * 40, skip=('model-00002-of-00002.safetensors',))
        self.assertFalse(ace.dit_ready(folder))
        for name in ('config.json', 'silence_latent.pt', 'model.safetensors.index.json'):
            folder = make_dit(self.root, 'ACE-Step/acestep-v15-xl-sft', name[:3] * 13 + 'c')
            (folder / name).unlink()
            self.assertFalse(ace.dit_ready(folder), name)

    def test_dit_with_a_single_weights_file(self):
        folder = self.root / 'single'
        folder.mkdir()
        (folder / 'config.json').write_text('{}')
        (folder / 'silence_latent.pt').write_bytes(b'x')
        self.assertFalse(ace.dit_ready(folder))
        (folder / 'model.safetensors').write_bytes(b'w')
        self.assertTrue(ace.dit_ready(folder))

    def test_a_broken_index_is_not_ready(self):
        folder = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'd' * 40)
        (folder / 'model.safetensors.index.json').write_text('{not json')
        self.assertFalse(ace.dit_ready(folder))
        (folder / 'model.safetensors.index.json').write_text('{"weight_map": {}}')
        self.assertFalse(ace.dit_ready(folder))
        self.assertFalse(ace.dit_ready(self.root / 'nowhere'))

    def test_main_ready(self):
        self.assertTrue(ace.main_ready(make_main(self.root, 'e' * 40)))
        self.assertFalse(ace.main_ready(make_main(self.root, 'f' * 40, skip=('vae/diffusion_pytorch_model.safetensors',))))
        self.assertFalse(ace.main_ready(make_main(self.root, '0' * 40, skip=('acestep-5Hz-lm-1.7B/model.safetensors',))))
        self.assertFalse(ace.main_ready(self.root / 'nowhere'))


class TestModelFolder(CacheCase):
    repo = 'ACE-Step/acestep-v15-xl-turbo'

    def test_pinned_snapshot_wins(self):
        pinned = make_dit(self.root, self.repo, 'a' * 40)
        make_dit(self.root, self.repo, 'b' * 40, age=9_999_999_999)
        self.assertEqual(ace.model_folder(pinned, self.repo, check=ace.dit_ready), (pinned, 'pinned'))

    def test_missing_pin_falls_back_to_the_newest_cached_snapshot(self):
        make_dit(self.root, self.repo, 'a' * 40, age=1_000_000)
        newest = make_dit(self.root, self.repo, 'b' * 40, age=2_000_000)
        make_dit(self.root, self.repo, 'c' * 40, age=500_000)
        pinned = self.root / ('models--' + self.repo.replace('/', '--')) / 'snapshots' / ('9' * 40)
        folder, how = ace.model_folder(pinned, self.repo, check=ace.dit_ready)
        self.assertEqual((folder, how), (newest, 'cached ' + 'b' * 12))

    def test_half_filled_pin_and_half_filled_cache_entries_are_skipped(self):
        broken_pin = make_dit(self.root, self.repo, 'a' * 40, skip=('model-00001-of-00002.safetensors',))
        make_dit(self.root, self.repo, 'b' * 40, skip=('model-00002-of-00002.safetensors',), age=3_000_000)
        good = make_dit(self.root, self.repo, 'c' * 40, age=1_000_000)
        self.assertEqual(ace.model_folder(broken_pin, self.repo, check=ace.dit_ready), (good, 'cached ' + 'c' * 12))

    def test_no_pin_looks_in_the_cache(self):
        good = make_dit(self.root, self.repo, 'a' * 40)
        self.assertEqual(ace.model_folder(None, self.repo, check=ace.dit_ready), (good, 'cached ' + 'a' * 12))

    def test_nothing_anywhere(self):
        self.assertEqual(ace.model_folder(self.root / 'x' / 'y' / 'z' / 'q', self.repo, check=ace.dit_ready),
                         (None, 'missing'))

    def test_other_repos_are_not_mistaken_for_this_one(self):
        make_dit(self.root, 'ACE-Step/acestep-v15-xl-sft', 'a' * 40)
        self.assertEqual(ace.model_folder(None, self.repo, check=ace.dit_ready), (None, 'missing'))

    def test_default_check_is_any_file(self):
        folder = self.root / 'models--o--n' / 'snapshots' / ('1' * 40)
        folder.mkdir(parents=True)
        self.assertEqual(ace.model_folder(None, 'o/n'), (None, 'missing'))
        (folder / 'f').write_text('x')
        self.assertEqual(ace.model_folder(None, 'o/n'), (folder, 'cached ' + '1' * 12))

    def test_explicit_roots(self):
        good = make_dit(self.root, self.repo, 'a' * 40)
        other = Path(self.tmp.name) / 'elsewhere'
        other.mkdir()
        self.assertEqual(ace.model_folder(None, self.repo, roots=[other], check=ace.dit_ready), (None, 'missing'))
        self.assertEqual(ace.model_folder(None, self.repo, roots=[other, self.root], check=ace.dit_ready)[0], good)


class TestResolveModels(CacheCase):
    def test_pinned_cached_and_downloaded_are_all_reported(self):
        turbo = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40)
        os.environ['ACE_TURBO_DIR'] = str(turbo)
        make_main(self.root, 'b' * 40)
        downloaded = make_dit(Path(self.tmp.name) / 'dl', 'ACE-Step/acestep-v15-xl-sft', 'c' * 40)
        os.environ['ACE_SFT_DIR'] = str(self.root / 'models--ACE-Step--acestep-v15-xl-sft' / 'snapshots' / ('c' * 40))
        with mock.patch.object(ace, 'fetch_model', return_value=downloaded) as fetch:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ace.resolve_models(download=True)
        self.assertEqual(ace.RESOLVED['turbo'], {'folder': turbo, 'how': 'pinned'})
        self.assertEqual(ace.RESOLVED['main']['how'], 'cached ' + 'b' * 12)
        self.assertEqual(ace.RESOLVED['sft'], {'folder': downloaded, 'how': 'downloaded'})
        fetch.assert_called_once_with('ACE-Step/acestep-v15-xl-sft', 'c' * 40, ace.DOWNLOAD_SKIP['sft'])
        self.assertEqual(os.environ['ACE_SFT_DIR'], str(downloaded))
        self.assertEqual(os.environ['ACE_TURBO_DIR'], str(turbo))
        self.assertIn('ACE-Step models:', buffer.getvalue())
        self.assertIn('acestep-v15-xl-sft downloaded', buffer.getvalue())

    def test_missing_with_download_off_never_fetches_and_is_retried_later(self):
        with mock.patch.object(ace, 'fetch_model', return_value=None) as fetch:
            silently(ace.resolve_models, download=False)
            fetch.assert_not_called()
        self.assertEqual({k: v['how'] for k, v in ace.RESOLVED.items()},
                         {'turbo': 'missing', 'sft': 'missing', 'main': 'missing'})
        good = make_dit(self.root, 'ACE-Step/acestep-v15-xl-sft', 'a' * 40)
        silently(ace.resolve_models, download=False, only=('sft',))
        self.assertEqual(ace.RESOLVED['sft'], {'folder': good, 'how': 'cached ' + 'a' * 12})

    def test_a_resolved_model_is_not_resolved_or_fetched_again(self):
        make_main(self.root, 'a' * 40)
        silently(ace.resolve_models, download=False, only=('main',))
        with mock.patch.object(ace, 'model_folder') as lookup, mock.patch.object(ace, 'fetch_model') as fetch:
            silently(ace.resolve_models, download=True, only=('main',))
        lookup.assert_not_called()
        fetch.assert_not_called()

    def test_only_limits_what_is_looked_up(self):
        with mock.patch.object(ace, 'fetch_model', return_value=None) as fetch:
            silently(ace.resolve_models, download=True, only=('sft',))
        self.assertEqual(set(ace.RESOLVED), {'sft'})
        self.assertEqual(fetch.call_count, 1)

    def test_an_incomplete_download_counts_as_missing(self):
        partial = make_dit(Path(self.tmp.name) / 'dl', 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40,
                           skip=('model-00001-of-00002.safetensors',))
        with mock.patch.object(ace, 'fetch_model', return_value=partial):
            silently(ace.resolve_models, download=True, only=('turbo',))
        self.assertEqual(ace.RESOLVED['turbo'], {'folder': None, 'how': 'missing'})

    def test_download_without_a_pin_asks_for_the_default_revision(self):
        with mock.patch.object(ace, 'fetch_model', return_value=None) as fetch:
            silently(ace.resolve_models, download=True, only=('main',))
        fetch.assert_called_once_with('ACE-Step/Ace-Step1.5', None, ace.DOWNLOAD_SKIP['main'])


def hub_modules(snapshot_download, constants, bound=False):
    """A stand-in huggingface_hub for sys.modules, shaped like the real package on a cold boot: it builds itself
    lazily, so `huggingface_hub.constants` is NOT an attribute until some code has imported that submodule
    (bound=False). The submodule is registered under its own name, which is all an explicit import needs."""
    fake = ModuleType('huggingface_hub')        # a real module object: `import a.b as c` reads its __name__
    fake.snapshot_download = snapshot_download
    if bound:
        fake.constants = constants
    return {'huggingface_hub': fake, 'huggingface_hub.constants': constants}


HUB = importlib.util.find_spec('huggingface_hub')

REAL_HUB_SCRIPT = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ['HF_HUB_OFFLINE'] = '1'
import ace_handler as ace
import huggingface_hub
seen = {}

def fake_download(repo, revision=None, cache_dir=None, ignore_patterns=None):
    # the real library is loaded, only the network call is replaced
    seen.update(env=os.environ['HF_HUB_OFFLINE'], flag=huggingface_hub.constants.HF_HUB_OFFLINE, repo=repo)
    return '/tmp/hub/snapshots/abc'

huggingface_hub.snapshot_download = fake_download
folder = ace.fetch_model('x/y', 'abc')
print(json.dumps({'folder': str(folder), 'seen': seen, 'env_after': os.environ['HF_HUB_OFFLINE'],
                  'flag_after': huggingface_hub.constants.HF_HUB_OFFLINE}))
"""


class TestFetchModel(unittest.TestCase):
    @unittest.skipIf(HUB is None, 'huggingface_hub is not installed on this machine (the image build has it)')
    def test_against_the_real_library_in_a_fresh_process(self):
        # The cold-boot order: nothing has imported huggingface_hub.constants before fetch_model runs. A stand-in
        # module cannot show that the real package binds the submodule only after somebody imports it.
        out = subprocess.run([sys.executable, '-I', '-B', '-c', REAL_HUB_SCRIPT, str(Path(__file__).resolve().parent)],
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.strip().splitlines()
        self.assertTrue(lines and lines[-1].startswith('{'), out.stdout)     # a swallowed failure prints a line instead
        result = json.loads(lines[-1])
        self.assertEqual(Path(result['folder']), Path('/tmp/hub/snapshots/abc'), out.stdout)
        self.assertEqual(result['seen'], {'env': '0', 'flag': False, 'repo': 'x/y'})
        self.assertEqual(result['env_after'], '1')
        self.assertIs(result['flag_after'], True)

    def test_works_whether_or_not_the_package_has_bound_its_constants_yet(self):
        for bound in (False, True):
            constants = SimpleNamespace(HF_HUB_OFFLINE=True)
            with mock.patch.dict(sys.modules, hub_modules(lambda *a, **k: '/tmp/x', constants, bound=bound)), \
                    mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
                self.assertEqual(ace.fetch_model('x/y', None), Path('/tmp/x'), bound)
            self.assertIs(constants.HF_HUB_OFFLINE, True, bound)

    def test_offline_mode_is_restored_after_a_download(self):
        seen = {}
        constants = SimpleNamespace(HF_HUB_OFFLINE=True)

        def snapshot_download(repo, revision=None, cache_dir=None, ignore_patterns=None):
            seen.update(repo=repo, revision=revision, cache_dir=cache_dir, ignore=ignore_patterns,
                        env=os.environ['HF_HUB_OFFLINE'], flag=constants.HF_HUB_OFFLINE)
            return '/tmp/hf-hub/models--x--y/snapshots/abc'
        with mock.patch.dict(sys.modules, hub_modules(snapshot_download, constants)), \
                mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
            folder = ace.fetch_model('x/y', 'abc', ('*.py',))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertEqual(folder, Path('/tmp/hf-hub/models--x--y/snapshots/abc'))
        self.assertEqual(seen, {'repo': 'x/y', 'revision': 'abc', 'cache_dir': '/tmp/hf-hub', 'ignore': ['*.py'],
                                'env': '0', 'flag': False})
        self.assertIs(constants.HF_HUB_OFFLINE, True)     # yue_handler left this False; here it is put back

    def test_the_hub_flag_follows_the_restored_environment_not_the_import_time_value(self):
        # huggingface_hub reads HF_HUB_OFFLINE once, at import. A first import inside fetch_model sees the
        # temporary '0', so the constant it holds afterwards must be rebuilt from the restored value.
        for restored, expected in (('1', True), ('0', False), (None, True)):
            constants = SimpleNamespace(HF_HUB_OFFLINE=False)
            env = {} if restored is None else {'HF_HUB_OFFLINE': restored}
            with mock.patch.dict(sys.modules, hub_modules(lambda *a, **k: '/tmp/x', constants)), \
                    mock.patch.dict(os.environ, env):
                if restored is None:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                ace.fetch_model('x/y', None)
                self.assertEqual(os.environ['HF_HUB_OFFLINE'], restored if restored is not None else '1')
            self.assertIs(constants.HF_HUB_OFFLINE, expected, restored)

    def test_a_failed_download_returns_none_and_restores_offline_mode(self):
        constants = SimpleNamespace(HF_HUB_OFFLINE=True)

        def snapshot_download(*args, **kwargs):
            raise ConnectionError('no route')
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, hub_modules(snapshot_download, constants)), \
                mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}), contextlib.redirect_stdout(buffer):
            self.assertIsNone(ace.fetch_model('x/y', None))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
        self.assertIs(constants.HF_HUB_OFFLINE, True)
        self.assertIn('ConnectionError', buffer.getvalue())
        self.assertNotIn('no route', buffer.getvalue())

    def test_a_missing_hub_library_is_handled(self):
        with mock.patch.dict(sys.modules, {'huggingface_hub': None}), mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
            self.assertIsNone(silently(ace.fetch_model, 'x/y', None))
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')


# ---------------------------------------------------------------- the checkpoints folder

def make_code(root):
    code = Path(root) / 'models'
    for variant, extra in (('xl_turbo', ['modeling_acestep_v15_xl_turbo.py']),
                           ('xl_sft', ['apg_guidance.py', 'modeling_acestep_v15_xl_base.py'])):
        folder = code / variant
        folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('')
        for name in ['configuration_acestep_v15.py'] + extra:
            (folder / name).write_text(f'# pinned upstream {variant} {name}\n')
    return code


def symlinks_allowed():
    with tempfile.TemporaryDirectory() as td:
        try:
            os.symlink(Path(td), Path(td) / 'probe', target_is_directory=True)
            return True
        except (OSError, NotImplementedError, AttributeError):
            return False


def copying_link(source, destination):
    source = Path(source)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


class TestBuildCheckpoints(CacheCase):
    def setUp(self):
        super().setUp()
        self.code = make_code(self.tmp.name)
        self.turbo = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40)
        self.sft = make_dit(self.root, 'ACE-Step/acestep-v15-xl-sft', 'b' * 40)
        self.main = make_main(self.root, 'c' * 40)
        self.target = Path(self.tmp.name) / 'ckpt'
        if not symlinks_allowed():
            patcher = mock.patch.object(ace, '_link', copying_link)
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, dits=True, main=True):
        ace.build_checkpoints(self.target, self.main if main else None,
                              {'acestep-v15-xl-turbo': (self.turbo, 'xl_turbo'),
                               'acestep-v15-xl-sft': (self.sft, 'xl_sft')} if dits else None, self.code)

    def test_layout_has_every_folder_upstream_looks_for(self):
        self.build()
        self.assertEqual(sorted(p.name for p in self.target.iterdir()),
                         sorted(['vae', 'Qwen3-Embedding-0.6B', 'acestep-5Hz-lm-1.7B', 'acestep-v15-turbo',
                                 'acestep-v15-xl-turbo', 'acestep-v15-xl-sft']))
        self.assertTrue((self.target / 'vae' / 'diffusion_pytorch_model.safetensors').exists())
        self.assertTrue((self.target / 'acestep-5Hz-lm-1.7B' / 'model.safetensors').exists())
        self.assertTrue((self.target / 'Qwen3-Embedding-0.6B' / 'model.safetensors').exists())

    def test_the_unused_2b_dit_is_a_placeholder_that_passes_upstreams_weights_check(self):
        self.build()
        placeholder = self.target / 'acestep-v15-turbo'
        self.assertEqual([p.name for p in placeholder.iterdir()], ['model.safetensors'])
        self.assertEqual((placeholder / 'model.safetensors').stat().st_size, 0)
        self.assertFalse(placeholder.is_symlink())

    def test_each_dit_has_weights_and_the_pinned_upstream_code(self):
        self.build()
        for name, snapshot, variant in (('acestep-v15-xl-turbo', self.turbo, 'xl_turbo'),
                                        ('acestep-v15-xl-sft', self.sft, 'xl_sft')):
            folder = self.target / name
            self.assertFalse(folder.is_symlink())
            self.assertTrue(ace.dit_ready(folder), name)
            for source in (self.code / variant).glob('*.py'):
                if source.name == '__init__.py':
                    self.assertFalse((folder / source.name).exists())
                    continue
                self.assertEqual((folder / source.name).read_text(), source.read_text(), source.name)
                self.assertFalse((folder / source.name).is_symlink())
            self.assertFalse((folder / 'modeling_from_the_hub.py').exists())     # the hub copy is never used

    def test_upstreams_code_sync_would_find_nothing_to_change(self):
        self.build()
        for name, variant in (('acestep-v15-xl-turbo', 'xl_turbo'), ('acestep-v15-xl-sft', 'xl_sft')):
            for source in (self.code / variant).glob('*.py'):
                if source.name != '__init__.py':
                    self.assertEqual((self.target / name / source.name).read_bytes(), source.read_bytes())

    def test_a_code_sync_write_cannot_reach_the_shared_cache(self):
        self.build()
        before = {p: p.read_bytes() for p in self.turbo.iterdir() if p.is_file()}
        # what upstream does when a file differs: shutil.copy2(source, destination)
        for source in (self.code / 'xl_turbo').glob('*.py'):
            if source.name != '__init__.py':
                shutil.copy2(source, self.target / 'acestep-v15-xl-turbo' / source.name)
        after = {p: p.read_bytes() for p in self.turbo.iterdir() if p.is_file()}
        self.assertEqual(before, after)

    def test_link_is_an_os_symlink_that_knows_folders_from_files(self):
        calls = []
        folder, file = self.main / 'vae', self.turbo / 'config.json'
        with mock.patch.object(ace, '_link', ACE_REAL_LINK), \
                mock.patch.object(ace.os, 'symlink', lambda *a, **k: calls.append((a, k))):
            ACE_REAL_LINK(folder, self.target / 'vae')
            ACE_REAL_LINK(file, self.target / 'config.json')
        self.assertEqual(calls, [((str(folder), str(self.target / 'vae')), {'target_is_directory': True}),
                                 ((str(file), str(self.target / 'config.json')), {'target_is_directory': False})])

    def test_python_files_are_never_linked_but_weights_and_json_are(self):
        linked = []

        def spy(source, destination):
            linked.append(Path(destination))
            copying_link(source, destination)
        with mock.patch.object(ace, '_link', spy):
            self.build()
        self.assertEqual([p for p in linked if p.suffix == '.py'], [])
        names = {p.name for p in linked}
        self.assertTrue({'vae', 'Qwen3-Embedding-0.6B', 'acestep-5Hz-lm-1.7B', 'config.json', 'silence_latent.pt',
                         'model.safetensors.index.json', 'model-00001-of-00002.safetensors',
                         'model-00002-of-00002.safetensors'} <= names)
        self.assertNotIn('acestep-v15-turbo', names)       # the placeholder is a real folder, nothing to link

    @unittest.skipUnless(symlinks_allowed(), 'symbolic links need extra rights on this Windows PC; run on Linux')
    def test_weights_are_links_into_the_snapshot_and_code_is_copied(self):
        self.build()
        for name in ('vae', 'Qwen3-Embedding-0.6B', 'acestep-5Hz-lm-1.7B'):
            self.assertTrue((self.target / name).is_symlink(), name)
        folder = self.target / 'acestep-v15-xl-turbo'
        for name in ('config.json', 'silence_latent.pt', 'model.safetensors.index.json',
                     'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'):
            self.assertTrue((folder / name).is_symlink(), name)
        self.assertFalse((folder / 'configuration_acestep_v15.py').is_symlink())

    def test_building_again_is_safe(self):
        self.build()
        self.build()
        self.assertTrue(ace.dit_ready(self.target / 'acestep-v15-xl-turbo'))
        self.assertTrue((self.target / 'vae').exists())

    def test_parts_can_be_built_separately(self):
        self.build(dits=False)
        self.assertFalse((self.target / 'acestep-v15-xl-turbo').exists())
        ace.build_checkpoints(self.target, None, {'acestep-v15-xl-sft': (self.sft, 'xl_sft')}, self.code)
        self.assertTrue(ace.dit_ready(self.target / 'acestep-v15-xl-sft'))
        self.assertTrue((self.target / 'vae').exists())

    def test_missing_upstream_code_is_a_plain_error(self):
        with self.assertRaises(ace.WorkerError) as caught:
            ace.build_checkpoints(self.target, None, {'acestep-v15-xl-turbo': (self.turbo, 'xl_turbo')},
                                  Path(self.tmp.name) / 'no-code')
        self.assertEqual(str(caught.exception), 'The music engine is not installed on this worker.')


class TestEnsureCheckpoints(CacheCase):
    def setUp(self):
        super().setUp()
        self.build = mock.Mock()
        for name, value in (('build_checkpoints', self.build),
                            ('upstream_models_dir', lambda: Path('/code')),
                            ('checkpoints_dir', lambda: Path('/ckpt'))):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_builds_main_and_turbo_once(self):
        ace.RESOLVED.update(turbo={'folder': Path('/t'), 'how': 'pinned'}, main={'folder': Path('/m'), 'how': 'pinned'})
        self.assertEqual(ace.ensure_checkpoints(('xl-turbo',)), Path('/ckpt'))
        self.build.assert_called_once_with(Path('/ckpt'), Path('/m'),
                                           {'acestep-v15-xl-turbo': (Path('/t'), 'xl_turbo')}, Path('/code'))
        ace.ensure_checkpoints(('xl-turbo',))
        self.assertEqual(self.build.call_count, 1)
        self.assertEqual(ace.STATE['built'], {'acestep-v15-xl-turbo', 'main'})

    def test_sft_is_added_later_without_rebuilding_the_rest(self):
        ace.RESOLVED.update(turbo={'folder': Path('/t'), 'how': 'pinned'}, main={'folder': Path('/m'), 'how': 'pinned'},
                            sft={'folder': Path('/s'), 'how': 'cached abc'})
        ace.ensure_checkpoints(('xl-turbo',))
        ace.ensure_checkpoints(('xl-sft',))
        self.assertEqual(self.build.call_args.args,
                         (Path('/ckpt'), None, {'acestep-v15-xl-sft': (Path('/s'), 'xl_sft')}, Path('/code')))

    def test_a_missing_model_is_a_plain_error(self):
        ace.RESOLVED.update(turbo={'folder': Path('/t'), 'how': 'pinned'}, main={'folder': Path('/m'), 'how': 'pinned'},
                            sft={'folder': None, 'how': 'missing'})
        with self.assertRaises(ace.WorkerError) as caught:
            ace.ensure_checkpoints(('xl-sft',))
        self.assertEqual(str(caught.exception), 'The best-quality music model is not available on this worker yet.')
        ace.RESOLVED['turbo'] = {'folder': None, 'how': 'missing'}
        with self.assertRaises(ace.WorkerError) as caught:
            ace.ensure_checkpoints(('xl-turbo',))
        self.assertEqual(str(caught.exception), 'The fast music model is not available on this worker yet.')
        ace.RESOLVED.pop('main')
        with self.assertRaises(ace.WorkerError) as caught:
            ace.ensure_checkpoints(())
        self.assertEqual(str(caught.exception), 'The music models are not available on this worker yet.')


# ---------------------------------------------------------------- loading and swapping models

class FakeDitHandler:
    created = []
    works = True       # set False to make the next handlers fail to initialise
    failure = 'Error initializing model: boom\n\nTraceback:\n...'

    def __init__(self):
        self.model = object()
        self.vae = self.text_encoder = self.text_tokenizer = self.config = self.silence_latent = object()
        self._base_decoder = None
        self.init_calls = []
        self.ok = FakeDitHandler.works
        FakeDitHandler.created.append(self)

    def initialize_service(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.ok:
            return '[OK] Model initialized successfully on cuda\nDtype: torch.bfloat16\nAttention: sdpa\n', True
        return self.failure, False


class FakeLlm:
    created = []
    failure = 'Error initializing 5Hz LM: boom'

    def __init__(self, ok=True):
        self.llm_initialized = False
        self.llm_backend = None
        self.ok = ok
        self.init_calls = []
        FakeLlm.created.append(self)

    def initialize(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.ok:
            self.llm_initialized, self.llm_backend = True, 'pt'
            return 'ok\nBackend: PyTorch', True
        return self.failure, False


class TestEngine(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        FakeDitHandler.created = []
        FakeDitHandler.works = True
        self.addCleanup(setattr, FakeDitHandler, 'works', True)
        self.events = []
        self.gib = 48.0
        modules = mock.patch.dict(sys.modules, {'acestep.model_downloader': None})     # the optional code-sync check
        modules.start()
        self.addCleanup(modules.stop)
        ace.STATE['ckpt'] = Path('/ckpt')
        ace.STATE['api'] = SimpleNamespace(AceStepHandler=FakeDitHandler, LLMHandler=FakeLlm)
        for name, value in (('require_gpu', lambda: None),
                            ('resolve_models', lambda **kw: self.events.append(('resolve', kw))),
                            ('ensure_checkpoints', lambda models=(), main=True: self.events.append(('build', models))),
                            ('gpu_total_gib', lambda: self.gib),
                            ('empty_cache', lambda: None)):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        real_load, real_release = ace.load_dit, ace.release_dit

        def load(model):
            self.events.append(('load', model))
            return real_load(model)

        def release(handler):
            self.events.append(('release', FakeDitHandler.created.index(handler)))
            return real_release(handler)
        for name, value in (('load_dit', load), ('release_dit', release)):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {'ACESTEP_PROJECT_ROOT': '/root'})
        env.start()
        self.addCleanup(env.stop)

    def loads(self):
        return [e for e in self.events if e[0] in ('load', 'release')]

    def test_load_dit_asks_upstream_for_a_plain_bf16_sdpa_cuda_load(self):
        handler = silently(ace.engine, 'xl-turbo')
        self.assertEqual(handler.init_calls, [dict(
            project_root='/root', config_path='acestep-v15-xl-turbo', device='cuda', use_flash_attention=False,
            compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None,
            prefer_source=None, use_mlx_dit=False)])

    def test_a_loaded_model_is_reused(self):
        first = silently(ace.engine, 'xl-turbo')
        second = silently(ace.engine, 'xl-turbo')
        self.assertIs(first, second)
        self.assertEqual(self.loads(), [('load', 'xl-turbo')])

    def test_a_big_card_keeps_both(self):
        turbo = silently(ace.engine, 'xl-turbo')
        sft = silently(ace.engine, 'xl-sft')
        self.assertIsNot(turbo, sft)
        self.assertEqual(self.loads(), [('load', 'xl-turbo'), ('load', 'xl-sft')])
        self.assertEqual(set(ace.STATE['dit']), {'xl-turbo', 'xl-sft'})
        self.assertIsNotNone(turbo.model)

    def test_a_card_under_40_gib_releases_turbo_before_loading_sft(self):
        self.gib = 31.8
        turbo = silently(ace.engine, 'xl-turbo')
        silently(ace.engine, 'xl-sft')
        self.assertEqual(self.loads(), [('load', 'xl-turbo'), ('release', 0), ('load', 'xl-sft')])
        self.assertEqual(set(ace.STATE['dit']), {'xl-sft'})
        self.assertIsNone(turbo.model)        # its weights were let go
        silently(ace.engine, 'xl-turbo')
        self.assertEqual(self.loads()[-2:], [('release', 1), ('load', 'xl-turbo')])
        self.assertEqual(set(ace.STATE['dit']), {'xl-turbo'})

    def test_a_card_that_cannot_be_measured_is_treated_as_small(self):
        self.gib = None
        silently(ace.engine, 'xl-turbo')
        silently(ace.engine, 'xl-sft')
        self.assertIn(('release', 0), self.events)

    def test_a_48_gb_card_as_torch_reports_it_is_not_small(self):
        self.gib = 44.99       # A40 and L40S report about 45 GiB
        silently(ace.engine, 'xl-turbo')
        silently(ace.engine, 'xl-sft')
        self.assertEqual([e[0] for e in self.loads()], ['load', 'load'])

    def test_resolution_and_layout_are_prepared_before_each_load(self):
        silently(ace.engine, 'xl-sft')
        self.assertEqual(self.events[0], ('resolve', {'download': True, 'only': ('sft', 'main')}))
        self.assertEqual(self.events[1], ('build', ('xl-sft',)))

    def test_a_handler_whose_model_was_lost_is_loaded_again_and_the_old_one_released(self):
        first = silently(ace.engine, 'xl-turbo')
        first.model = None
        second = silently(ace.engine, 'xl-turbo')
        self.assertIsNot(first, second)
        self.assertIsNone(first.vae)           # the leftovers were let go, not just forgotten
        self.assertIn(('release', 0), self.events)

    def test_a_failed_load_is_a_plain_error_and_leaves_nothing_registered(self):
        FakeDitHandler.works = False
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(ace.WorkerError) as caught:
            ace.engine('xl-sft')
        self.assertEqual(str(caught.exception), 'The best-quality music model could not be loaded.')
        self.assertEqual(ace.STATE['dit'], {})
        self.assertIn('xl-sft load failed: Error initializing model: boom', buffer.getvalue())
        self.assertNotIn('Traceback', buffer.getvalue())
        self.assertIsNone(FakeDitHandler.created[-1].model)       # the half-built handler was released

    def test_a_load_failure_line_keeps_the_reason_but_not_links_or_tokens(self):
        self.addCleanup(setattr, FakeDitHandler, 'failure', FakeDitHandler.failure)
        FakeDitHandler.works = False
        FakeDitHandler.failure = ('Error initializing model: 401 Client Error for url: '
                                  'https://huggingface.co/ACE-Step/x/resolve/main/config.json '
                                  'with token hf_AbCdEfGh1234567890 (HTTP://CDN.EXAMPLE/a?b=c)\nTraceback: more')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(ace.WorkerError):
            ace.engine('xl-turbo')
        line = buffer.getvalue()
        self.assertIn('xl-turbo load failed: Error initializing model: 401 Client Error for url: [link]', line)
        self.assertNotIn('huggingface.co', line)
        self.assertNotIn('hf_AbCd', line)
        self.assertNotIn('CDN.EXAMPLE', line)
        self.assertNotIn('Traceback', line)

    def test_a_card_is_checked_before_any_model_work(self):
        def no_card():
            raise ace.WorkerError('No graphics card is available on this worker. Try again in a moment.')
        with mock.patch.object(ace, 'require_gpu', no_card), self.assertRaises(ace.WorkerError) as caught:
            silently(ace.engine, 'xl-turbo')
        self.assertIn('No graphics card', str(caught.exception))
        self.assertEqual(self.events, [])           # nothing was resolved, built or loaded
        self.assertEqual(ace.STATE['dit'], {})

    def test_slow_attention_is_logged(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ace.log_status('xl-turbo', 'Dtype: torch.bfloat16\nAttention: eager\n')
            ace.log_status('xl-turbo', 'Dtype: torch.bfloat16\nAttention: sdpa\n')
        self.assertEqual(buffer.getvalue().count('WARNING'), 1)

    def test_release_dit_lets_go_of_everything(self):
        handler = FakeDitHandler()
        handler._base_decoder = object()
        silently(ace.release_dit, handler)
        for name in ('model', 'vae', 'text_encoder', 'text_tokenizer', 'config', 'silence_latent', '_base_decoder'):
            self.assertIsNone(getattr(handler, name), name)

    def test_a_dirty_model_is_released_and_loaded_again_before_it_is_used(self):
        first = silently(ace.engine, 'xl-sft')
        ace.STATE['dirty'].add('xl-sft')            # what a style that could not be removed cleanly leaves behind
        second = silently(ace.engine, 'xl-sft')
        self.assertIsNot(first, second)
        self.assertIsNone(first.model)              # the possibly adapted weights were let go, not just forgotten
        self.assertEqual(self.loads(), [('load', 'xl-sft'), ('release', 0), ('load', 'xl-sft')])
        self.assertEqual(ace.STATE['dirty'], set())
        self.assertIs(ace.STATE['dit']['xl-sft'], second)
        self.assertIs(silently(ace.engine, 'xl-sft'), second)         # afterwards it is simply reused
        self.assertEqual(len(self.loads()), 3)

    def test_the_other_model_is_left_alone_when_one_is_dirty(self):
        turbo = silently(ace.engine, 'xl-turbo')
        silently(ace.engine, 'xl-sft')
        ace.STATE['dirty'].add('xl-sft')
        self.assertIs(silently(ace.engine, 'xl-turbo'), turbo)
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})              # still marked until it is asked for
        self.assertEqual([e for e in self.loads() if e[0] == 'load'], [('load', 'xl-turbo'), ('load', 'xl-sft')])

    def test_a_dirty_model_that_is_not_loaded_any_more_is_just_loaded(self):
        ace.STATE['dirty'].add('xl-sft')
        handler = silently(ace.engine, 'xl-sft')
        self.assertIsNotNone(handler.model)
        self.assertEqual((self.loads(), ace.STATE['dirty']), ([('load', 'xl-sft')], set()))

    def test_a_reload_that_fails_keeps_the_mark_and_the_next_call_tries_again(self):
        silently(ace.engine, 'xl-sft')
        ace.STATE['dirty'].add('xl-sft')
        FakeDitHandler.works = False
        with quiet(), self.assertRaises(ace.WorkerError):
            ace.engine('xl-sft')
        self.assertEqual(ace.STATE['dirty'], {'xl-sft'})
        self.assertNotIn('xl-sft', ace.STATE['dit'])               # the adapted handler is gone for good
        FakeDitHandler.works = True
        self.assertIsNotNone(silently(ace.engine, 'xl-sft').model)
        self.assertEqual(ace.STATE['dirty'], set())

    def test_a_small_card_reloads_a_dirty_model_the_same_way(self):
        self.gib = 31.8
        silently(ace.engine, 'xl-sft')
        ace.STATE['dirty'].add('xl-sft')
        silently(ace.engine, 'xl-sft')
        self.assertEqual(self.loads(), [('load', 'xl-sft'), ('release', 0), ('load', 'xl-sft')])

    def test_the_reload_is_logged_without_any_text_of_the_song(self):
        silently(ace.engine, 'xl-sft')
        ace.STATE['dirty'].add('xl-sft')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ace.engine('xl-sft')
        self.assertIn('ACE-Step xl-sft: weights reloaded after a style could not be removed cleanly', buffer.getvalue())

    def test_a_model_that_is_not_dirty_is_never_reloaded_for_that_reason(self):
        first = silently(ace.engine, 'xl-sft')
        for _ in range(3):
            self.assertIs(silently(ace.engine, 'xl-sft'), first)
        self.assertEqual(self.loads(), [('load', 'xl-sft')])


class TestPlannerLoad(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        FakeLlm.created = []
        ace.STATE['ckpt'] = Path('/ckpt')
        ace.STATE['api'] = SimpleNamespace(LLMHandler=lambda: FakeLlm(self.ok))
        self.ok = True
        for name, value in (('require_gpu', lambda: None), ('resolve_models', lambda **kw: None),
                            ('ensure_checkpoints', lambda models=(), main=True: None)):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_loads_the_1_7b_planner_with_the_pt_backend(self):
        llm = silently(ace.planner)
        self.assertIs(llm, FakeLlm.created[0])
        self.assertEqual(llm.init_calls, [dict(checkpoint_dir=str(Path('/ckpt')), lm_model_path='acestep-5Hz-lm-1.7B',
                                               backend='pt', device='cuda', offload_to_cpu=False)])
        self.assertIs(silently(ace.planner), llm)
        self.assertEqual(len(FakeLlm.created), 1)

    def test_backend_knob(self):
        for value, want in (('vllm', 'vllm'), ('PT', 'pt'), ('nonsense', 'pt'), ('', 'pt')):
            with mock.patch.dict(os.environ, {'ACE_LM_BACKEND': value}):
                self.assertEqual(ace.lm_backend(), want)
        with mock.patch.dict(os.environ, {'ACE_LM_BACKEND': 'vllm'}):
            reset_state()
            ace.STATE['ckpt'] = Path('/ckpt')
            ace.STATE['api'] = SimpleNamespace(LLMHandler=lambda: FakeLlm(True))
            self.assertEqual(silently(ace.planner).init_calls[0]['backend'], 'vllm')

    def test_a_failed_load_gives_none_and_is_retried_once(self):
        self.ok = False
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertIsNone(ace.planner())
            self.assertIsNone(ace.planner())
            self.assertIsNone(ace.planner())
        self.assertEqual(len(FakeLlm.created), 2)
        self.assertIn('planner load failed: Error initializing 5Hz LM: boom', buffer.getvalue())
        self.assertNotIn('Traceback', buffer.getvalue())

    def test_the_planner_load_failure_line_drops_links_and_tokens(self):
        self.addCleanup(setattr, FakeLlm, 'failure', FakeLlm.failure)
        FakeLlm.failure = 'Error initializing 5Hz LM: cannot reach https://huggingface.co/x with hf_ZyXwVuTs9876543210'
        self.ok = False
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertIsNone(ace.planner())
        self.assertIn('planner load failed: Error initializing 5Hz LM: cannot reach [link] with [token]', buffer.getvalue())

    def test_an_exception_while_loading_gives_none(self):
        def broken():
            raise RuntimeError('boom')
        ace.STATE['api'] = SimpleNamespace(LLMHandler=broken)
        self.assertIsNone(silently(ace.planner))


class TestBoot(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.addCleanup(reset_state)
        self.events = []
        for name, value in (('require_gpu', lambda: self.events.append('gpu')),
                            ('resolve_models', lambda **kw: self.events.append(('resolve', kw))),
                            ('engine', lambda model: self.events.append(('engine', model))),
                            ('planner', lambda: self.events.append('planner'))):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_boot_loads_turbo_and_the_planner_and_only_looks_for_sft(self):
        silently(ace.boot)
        self.assertEqual(self.events, ['gpu', ('resolve', {'download': True, 'only': ('turbo', 'main')}),
                                       ('resolve', {'download': False, 'only': ('sft',)}),
                                       ('engine', 'xl-turbo'), 'planner'])

    def test_warm_reports_ready(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ace.warm()
        self.assertIn('ACE-Step ready at boot in', buffer.getvalue())

    def test_a_boot_failure_is_logged_and_never_raised(self):
        with mock.patch.object(ace, 'engine', side_effect=RuntimeError('no gpu: private path /x')):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                ace.warm()
        self.assertIn('ACE-Step boot load skipped: RuntimeError', buffer.getvalue())
        self.assertNotIn('private path', buffer.getvalue())

def fake_torch(available=True, name='NVIDIA A40', total_gib=44.99, peak_gib=3.0):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: available, get_device_name=lambda: name,
        get_device_properties=lambda index: SimpleNamespace(total_memory=int(total_gib * 2 ** 30)),
        reset_peak_memory_stats=lambda: calls.append('reset'), max_memory_allocated=lambda: int(peak_gib * 2 ** 30),
        empty_cache=lambda: calls.append('empty'), manual_seed_all=lambda seed: calls.append(('cuda seed', seed)))
    return SimpleNamespace(cuda=cuda, manual_seed=lambda seed: calls.append(('seed', seed)),
                           __version__='2.10.0+cu128'), calls


class TestCardHelpers(unittest.TestCase):
    sentence = 'No graphics card is available on this worker. Try again in a moment.'

    def test_no_torch_or_no_card_is_a_plain_error(self):
        with mock.patch.dict(sys.modules, {'torch': None}), self.assertRaises(ace.WorkerError) as caught:
            ace.require_gpu()
        self.assertEqual(str(caught.exception), self.sentence)
        torch, _ = fake_torch(available=False)
        with mock.patch.dict(sys.modules, {'torch': torch}), self.assertRaises(ace.WorkerError) as caught:
            ace.require_gpu()
        self.assertEqual(str(caught.exception), self.sentence)

    def test_a_card_passes(self):
        torch, _ = fake_torch()
        with mock.patch.dict(sys.modules, {'torch': torch}):
            ace.require_gpu()

    def test_name_memory_and_peak(self):
        torch, calls = fake_torch(name='NVIDIA A40', total_gib=44.99, peak_gib=21.456)
        with mock.patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(ace.gpu_name(), 'NVIDIA A40')
            self.assertAlmostEqual(ace.gpu_total_gib(), 44.99, places=1)
            self.assertIsNone(ace.gpu_peak(reset=True))
            self.assertEqual(ace.gpu_peak(), 21.46)
            self.assertEqual(ace.versions()['torch'], '2.10.0+cu128')
        self.assertEqual(calls, ['reset'])

    def test_everything_degrades_to_none_without_a_card(self):
        torch, _ = fake_torch(available=False)
        for modules in ({'torch': None}, {'torch': torch}):
            with mock.patch.dict(sys.modules, modules):
                self.assertIsNone(ace.gpu_name())
                self.assertIsNone(ace.gpu_peak())
                self.assertIsNone(ace.gpu_peak(reset=True))
                self.assertIsNone(ace.gpu_total_gib())
        with mock.patch.dict(sys.modules, {'torch': None}):
            self.assertIsNone(ace.versions()['torch'])

    def test_empty_cache_and_seed_use_torch_only_if_it_is_already_loaded(self):
        torch, calls = fake_torch()
        with mock.patch.dict(sys.modules, {'torch': torch}):
            ace.empty_cache()
            ace.seed_everything(5)
        self.assertEqual(calls, ['empty', ('seed', 5), ('cuda seed', 5)])
        saved = sys.modules.pop('torch', None)
        try:
            ace.empty_cache()
            ace.seed_everything(5)
            self.assertNotIn('torch', sys.modules)       # tidying up never imports torch
        finally:
            if saved is not None:
                sys.modules['torch'] = saved


class TestUpstreamImports(unittest.TestCase):
    def test_upstream_is_imported_lazily_with_quiet_logs_and_env(self):
        reset_state()
        self.addCleanup(reset_state)
        removed, added = [], []
        loguru = SimpleNamespace(logger=SimpleNamespace(remove=lambda *a: removed.append(a),
                                                        add=lambda *a, **k: added.append((a, k))))
        handler = SimpleNamespace(AceStepHandler='dit')
        inference = SimpleNamespace(GenerationConfig='config', GenerationParams='params', generate_music='gen')
        llm = SimpleNamespace(LLMHandler='llm')
        modules = {'loguru': loguru, 'acestep': SimpleNamespace(), 'acestep.handler': handler,
                   'acestep.inference': inference, 'acestep.llm_inference': llm}
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {}, clear=False):
            for name in ('ACESTEP_SAVE_MEMORY', 'ACESTEP_DISABLE_TQDM', 'HF_HUB_OFFLINE', 'ACESTEP_PROJECT_ROOT'):
                os.environ.pop(name, None)
            with mock.patch.object(Path, 'mkdir'):
                api = ace.upstream()
            self.assertEqual((api.AceStepHandler, api.LLMHandler, api.GenerationParams, api.GenerationConfig,
                              api.generate_music), ('dit', 'llm', 'params', 'config', 'gen'))
            self.assertIs(ace.upstream(), api)
            self.assertEqual(os.environ['ACESTEP_SAVE_MEMORY'], '1')
            self.assertEqual(os.environ['ACESTEP_DISABLE_TQDM'], '1')
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['ACESTEP_GENERATION_TIMEOUT'], '3600')
            self.assertEqual(os.environ['ACESTEP_CHECKPOINTS_DIR'], str(ace.checkpoints_dir()))
        self.assertEqual(removed, [()])
        # backtrace/diagnose off: loguru's defaults print variable values (the caption and lyrics) in tracebacks
        self.assertEqual(added[0][1], {'level': 'WARNING', 'backtrace': False, 'diagnose': False})
        self.assertEqual(len(added), 1)

    def test_existing_settings_are_not_overridden(self):
        reset_state()
        self.addCleanup(reset_state)
        with mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '0', 'ACESTEP_PROJECT_ROOT': '/custom'}), \
                mock.patch.object(Path, 'mkdir'):
            ace.configure_environment()
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '0')
            self.assertEqual(os.environ['ACESTEP_PROJECT_ROOT'], '/custom')

    def test_importing_the_module_pulls_in_no_heavy_library(self):
        here = str(Path(__file__).resolve().parent)
        code = ("import sys; sys.path.insert(0, %r); import ace_handler; "
                "print(sorted(m for m in ('torch', 'torchaudio', 'transformers', 'acestep', 'numpy', 'soundfile', "
                "'boto3', 'runpod', 'loguru', 'huggingface_hub', 'diffusers') if m in sys.modules))" % here)
        out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), '[]')


LOGURU = importlib.util.find_spec('loguru')

# Mirrors upstream's failure path: inference.py catches the error of a render and calls logger.exception(...), and the
# frame that failed holds the caption and lyrics. argv[2] picks the sink: the worker's own, or loguru's plain defaults.
LOG_SCRIPT = r"""
import sys
sys.path.insert(0, sys.argv[1])
import ace_handler as ace
from loguru import logger
if sys.argv[2] == 'worker':
    ace.quiet_logs()
else:
    logger.remove()
    logger.add(sys.stderr, level='WARNING')

def generate(captions, lyrics):
    try:
        return len(captions) / len(lyrics[:0])
    except Exception:
        logger.exception('Music generation failed')

logger.info('INFO-LINE-SENTINEL')
generate(sys.argv[3], sys.argv[4])
"""


@unittest.skipIf(LOGURU is None, 'loguru is not installed on this machine (the image build has it)')
class TestQuietLogsWithRealLoguru(unittest.TestCase):
    def logged(self, sink):
        # a real script file, not -c: loguru reads the source line of each frame to print the variables on it
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / 'fail_a_render.py'
            script.write_text(LOG_SCRIPT, encoding='utf-8')
            out = subprocess.run([sys.executable, '-I', '-B', str(script), str(Path(__file__).resolve().parent),
                                  sink, SECRET_CAPTION, SECRET_LYRICS], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout + out.stderr

    def test_a_failed_render_does_not_put_the_callers_words_in_the_log(self):
        text = self.logged('worker')
        self.assertIn('Music generation failed', text)       # the failure is still reported
        self.assertIn('ZeroDivisionError', text)
        self.assertNotIn('INFO-LINE-SENTINEL', text)          # and INFO lines stay out
        self.assertNotIn('ZEBRA', text)
        self.assertNotIn('KANGAROO', text)

    def test_the_check_can_see_a_leak_with_loguru_defaults(self):
        # control: with a plain WARNING sink the same traceback prints the variables, so the test above is meaningful
        text = self.logged('default')
        self.assertIn('ZEBRA', text)
        self.assertIn('KANGAROO', text)


class TestMisc(unittest.TestCase):
    def test_features_and_versions(self):
        self.assertEqual(ace.FEATURES, ('ace-xl-turbo', 'ace-xl-sft', 'batch', 'instrumental', 'probe', 'lora'))
        with mock.patch.dict(os.environ, {'ACE_STEP_COMMIT': 'abc', 'ACE_WORKER_VERSION': 'w9'}):
            self.assertEqual(ace.versions()['ace_step_commit'], 'abc')
            self.assertEqual(ace.versions()['worker'], 'w9')
        self.assertEqual(ace.UPSTREAM_COMMIT, 'ca1e85fe9430179831e6bc6be790c332190a3866')

    def test_storage_reads_the_same_settings_as_yue(self):
        made = {}
        fake_boto3 = SimpleNamespace(client=lambda service, **kw: made.update(service=service, **kw) or 'client')
        env = {'AWS_ENDPOINT_URL': 'https://s3.example', 'AWS_BUCKET_NAME': 'bucket'}
        with mock.patch.dict(sys.modules, {'boto3': fake_boto3}), mock.patch.dict(os.environ, env):
            os.environ.pop('AWS_REGION', None)
            self.assertEqual(ace.storage(), ('client', 'bucket'))
        self.assertEqual(made, {'service': 's3', 'endpoint_url': 'https://s3.example', 'region_name': 'us-east-005'})

    def test_the_log_never_prints_the_callers_words(self):
        source = Path(ace.__file__).read_text(encoding='utf-8')
        for number, line in enumerate(source.splitlines(), 1):
            if 'print(' in line and re.search(r'caption|lyrics|request\[', line):
                self.assertIn('scrub(', line, f'line {number} prints request text')

    def test_no_credentials_in_either_file(self):
        for path in (Path(ace.__file__), Path(__file__)):
            text = path.read_text(encoding='utf-8')
            self.assertNotRegex(text, r'AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}')
            self.assertNotIn('BEGIN ' + 'PRIVATE', text)
            self.assertNotIn('PRIVATE' + '_kade', text)      # the credentials file is never referenced

    def test_sentences_shown_to_people_name_no_one_and_no_folder(self):
        tree = ast.parse(Path(ace.__file__).read_text(encoding='utf-8'))
        docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                      and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        sentences = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and id(n) not in docstrings and ' ' in n.value]
        self.assertGreater(len(sentences), 40)
        for sentence in sentences:
            for word in ('Kade', 'Murdock', 'Desktop', 'Users', 'F:/', 'C:/'):
                self.assertNotIn(word, sentence)

    def test_every_sentence_returned_to_the_booth_is_short(self):
        for sentence in (ace.ADAPTER_SENTENCE, ace.STYLE_MODEL_SENTENCE, ace.STYLE_SCALE_SENTENCE, ace.TOO_BIG,
                         ace.FAILED):
            self.assertLessEqual(len(sentence), 100)


# ---------------------------------------------------------------- against the pinned upstream source

def find_upstream():
    for candidate in (os.environ.get('ACE_UPSTREAM_DIR'), '/opt/ace-step', 'F:/KadeAI-work/ace/upstream'):
        if candidate and (Path(candidate) / 'acestep' / 'inference.py').is_file():
            return Path(candidate)
    return None


UPSTREAM = find_upstream()


def parse(relative):
    return ast.parse((UPSTREAM / relative).read_text(encoding='utf-8'))


def class_node(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)


def function_node(tree, name, inside=None):
    scope = class_node(tree, inside) if inside else tree
    return next(n for n in ast.walk(scope) if isinstance(n, ast.FunctionDef) and n.name == name)


def dataclass_fields(tree, name):
    return {n.target.id for n in class_node(tree, name).body if isinstance(n, ast.AnnAssign)}


def parameters(function):
    args = function.args
    return {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs} - {'self'}


def literal(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, 'id', None) == name and node.value is not None:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


@unittest.skipIf(UPSTREAM is None, 'no clone of the pinned ACE-Step source on this machine')
class TestPinnedUpstream(unittest.TestCase):
    """Every name this worker sends to upstream exists in the pinned source (not just in a fake)."""

    def request(self, **changes):
        return ace.request_input(valid(**changes))

    def test_generation_fields_exist(self):
        tree = parse('acestep/inference.py')
        sent = {}

        class Params(Record):
            def __init__(self, **fields):
                sent['params'] = set(fields)

        class Config(Record):
            def __init__(self, **fields):
                sent['config'] = set(fields)
        api = SimpleNamespace(GenerationParams=Params, GenerationConfig=Config)
        request = self.request()
        settings, _ = ace.settings_for(request)
        ace.build_generation(api, request, settings, [1, 2], True)
        self.assertLessEqual(sent['params'], dataclass_fields(tree, 'GenerationParams'))
        self.assertLessEqual(sent['config'], dataclass_fields(tree, 'GenerationConfig'))
        self.assertTrue({'use_random_seed', 'seeds', 'batch_size'} <= sent['config'])

    def test_lm_temperature_is_a_generation_field_and_its_default_is_the_booths_midpoint(self):
        tree = parse('acestep/inference.py')
        sent = {}

        class Params(Record):
            def __init__(self, **fields):
                sent.update(fields)
        api = SimpleNamespace(GenerationParams=Params, GenerationConfig=Record)
        request = self.request(lm_temperature=1.3)
        settings, _ = ace.settings_for(request)
        ace.build_generation(api, request, settings, [1], True)
        self.assertEqual(sent['lm_temperature'], 1.3)
        self.assertLessEqual(set(sent), dataclass_fields(tree, 'GenerationParams'))
        default = next(ast.literal_eval(n.value) for n in class_node(tree, 'GenerationParams').body
                       if isinstance(n, ast.AnnAssign) and n.target.id == 'lm_temperature')
        self.assertEqual(default, 0.85)      # the booth's dial sits at 0.85 in the middle, so the middle sends nothing
        self.assertLessEqual(ace.LM_TEMPERATURE_MIN, default)
        self.assertLessEqual(default, ace.LM_TEMPERATURE_MAX)

    def test_the_planner_reads_lm_temperature_in_both_of_its_passes(self):
        inference = (UPSTREAM / 'acestep' / 'inference.py').read_text(encoding='utf-8')
        llm = (UPSTREAM / 'acestep' / 'llm_inference.py').read_text(encoding='utf-8')
        self.assertRegex(inference, r'Sampling temperature for the LLM \(0\.0.{1,3}2\.0\)')   # the dash is not ASCII
        self.assertIn('temperature=params.lm_temperature,', inference)
        # it goes to generate_with_stop_condition, which hands it to the metadata pass and to the audio-codes pass
        self.assertIn('"temperature": temperature,', llm)
        # and the pt backend's two sampling loops divide the logits by it (the codes pass runs with CFG, the metadata pass without)
        self.assertIn('self._sample_tokens(cfg_logits, temperature)', llm)
        self.assertIn('self._sample_tokens(next_token_logits, temperature)', llm)
        # the worker never sets upstream's separate per-phase temperatures, which would override this one
        self.assertIn('self.constrained_processor.metadata_temperature = None', llm)
        self.assertNotIn('metadata_temperature', (Path(__file__).resolve().parent / 'ace_handler.py').read_text(encoding='utf-8'))

    def test_guidance_range_and_the_turbo_override_match_upstream(self):
        controls = (UPSTREAM / 'acestep' / 'ui' / 'gradio' / 'interfaces' / 'generation_advanced_dit_controls.py'
                    ).read_text(encoding='utf-8')
        self.assertRegex(controls, r'guidance_scale = gr\.Slider\(\s*minimum=1\.0,\s*maximum=15\.0')
        handler = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'generate_music.py').read_text(encoding='utf-8')
        self.assertIn('if self.is_turbo_model() and guidance_scale != 1.0:', handler)
        sft = (UPSTREAM / 'acestep' / 'models' / 'xl_sft' / 'modeling_acestep_v15_xl_base.py').read_text(encoding='utf-8')
        self.assertIn('do_cfg_guidance = diffusion_guidance_scale > 1.0', sft)     # no upper cap in the sampler
        self.assertEqual(ace.request_input(valid(model='xl-sft', guidance_scale=15))['guidance_scale'], 15)

    def test_the_three_imports_exist_and_the_handlers_build_without_arguments(self):
        handler = parse('acestep/handler.py')
        llm = parse('acestep/llm_inference.py')
        inference = parse('acestep/inference.py')
        class_node(handler, 'AceStepHandler')
        class_node(llm, 'LLMHandler')
        function_node(inference, 'generate_music')
        class_node(inference, 'GenerationParams')
        class_node(inference, 'GenerationConfig')
        for tree, name in ((handler, 'AceStepHandler'), (llm, 'LLMHandler')):
            init = function_node(tree, '__init__', inside=name)
            required = len(init.args.args) - 1 - len(init.args.defaults)       # minus self, minus defaulted
            self.assertEqual(required, 0, name)

    def test_generate_music_signature_and_result(self):
        tree = parse('acestep/inference.py')
        self.assertTrue({'dit_handler', 'llm_handler', 'params', 'config', 'save_dir', 'progress'}
                        <= parameters(function_node(tree, 'generate_music')))
        self.assertTrue({'audios', 'extra_outputs', 'success', 'error'} <= dataclass_fields(tree, 'GenerationResult'))

    def test_defaults_this_worker_relies_on_are_still_upstreams(self):
        # use_random_seed=True is why the worker must pass False; thinking and the use_cot flags default on
        tree = parse('acestep/inference.py')
        defaults = {}
        for node in class_node(tree, 'GenerationConfig').body:
            if isinstance(node, ast.AnnAssign):
                defaults[node.target.id] = ast.literal_eval(node.value)
        self.assertIs(defaults['use_random_seed'], True)
        self.assertEqual(defaults['batch_size'], 2)
        params = {}
        for node in class_node(tree, 'GenerationParams').body:
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                try:
                    params[node.target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        self.assertEqual((params['use_cot_metas'], params['use_cot_caption'], params['use_cot_language']),
                         (True, True, True))

    def test_the_planner_is_skipped_only_when_all_three_cot_flags_and_thinking_are_off(self):
        text = (UPSTREAM / 'acestep' / 'inference.py').read_text(encoding='utf-8')
        self.assertIn('need_lm_for_cot = params.use_cot_caption or params.use_cot_language or params.use_cot_metas', text)
        self.assertIn('use_lm = (params.thinking or need_lm_for_cot) and llm_handler is not None', text)

    def test_time_cost_keys_the_worker_reads(self):
        text = (UPSTREAM / 'acestep' / 'inference.py').read_text(encoding='utf-8')
        self.assertIn('lm_total_time_costs', text)
        self.assertIn('"total_time": 0.0', text)
        self.assertIn('unified_time_costs[f"lm_{key}"]', text)
        self.assertIn('extra_outputs["lm_metadata"]', text)
        self.assertIn('extra_outputs["time_costs"]', text)

    def test_initialize_service_arguments_exist(self):
        tree = parse('acestep/core/generation/handler/init_service_orchestrator.py')
        known = parameters(function_node(tree, 'initialize_service'))
        handler = FakeDitHandler()
        reset_state()
        self.addCleanup(reset_state)
        ace.STATE['ckpt'] = Path('/ckpt')
        ace.STATE['api'] = SimpleNamespace(AceStepHandler=lambda: handler)
        with mock.patch.dict(os.environ, {'ACESTEP_PROJECT_ROOT': '/root'}), \
                mock.patch.dict(sys.modules, {'acestep.model_downloader': None}):
            silently(ace.load_dit, 'xl-turbo')
        self.assertLessEqual(set(handler.init_calls[0]), known)

    def test_planner_initialize_arguments_exist(self):
        tree = parse('acestep/llm_inference.py')
        known = parameters(function_node(tree, 'initialize', inside='LLMHandler'))
        reset_state()
        self.addCleanup(reset_state)
        ace.STATE['ckpt'] = Path('/ckpt')
        ace.STATE['api'] = SimpleNamespace(LLMHandler=lambda: FakeLlm(True))
        for name, value in (('require_gpu', lambda: None), ('resolve_models', lambda **kw: None),
                            ('ensure_checkpoints', lambda models=(), main=True: None)):
            patcher = mock.patch.object(ace, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        llm = silently(ace.planner)
        self.assertLessEqual(set(llm.init_calls[0]), known)
        self.assertTrue(hasattr(llm, 'llm_initialized'))
        self.assertIn('llm_initialized', (UPSTREAM / 'acestep' / 'llm_inference.py').read_text(encoding='utf-8'))

    def test_the_handler_attributes_released_exist(self):
        tree = parse('acestep/handler.py')
        init = function_node(tree, '__init__', inside='AceStepHandler')
        assigned = {t.attr for n in ast.walk(init) if isinstance(n, ast.Assign) for t in n.targets
                    if isinstance(t, ast.Attribute)}
        released = {'model', 'vae', 'text_encoder', 'text_tokenizer', 'config', 'silence_latent', '_base_decoder'}
        self.assertLessEqual(released, assigned)

    def test_checkpoint_names_and_repos_match_upstreams_registry(self):
        tree = parse('acestep/model_downloader.py')
        variants = literal(tree, '_CHECKPOINT_TO_VARIANT')
        registry = literal(tree, 'SUBMODEL_REGISTRY')
        for model, spec in ace.DIT_MODELS.items():
            self.assertEqual(variants[spec['folder']], spec['variant'], model)
            repo = next(r for _env, r, key in ace.MODELS if key == spec['key'])
            self.assertEqual(registry[spec['folder']], repo, model)
            code = UPSTREAM / 'acestep' / 'models' / spec['variant']
            self.assertTrue([p for p in code.glob('*.py') if p.name != '__init__.py'], model)
        self.assertEqual(set(literal(tree, 'MAIN_MODEL_COMPONENTS')), {ace.PLACEHOLDER_DIT, *ace.MAIN_PARTS})
        self.assertEqual(literal(tree, 'MAIN_MODEL_REPO'), next(r for _e, r, key in ace.MODELS if key == 'main'))
        self.assertIn(ace.LM_FOLDER, ace.MAIN_PARTS)
        self.assertEqual(literal(tree, 'DEFAULT_LM_MODEL'), ace.LM_FOLDER)

    def test_weight_check_upstream_uses_for_the_placeholder(self):
        text = (UPSTREAM / 'acestep' / 'model_downloader.py').read_text(encoding='utf-8')
        self.assertIn('"model.safetensors"', text)
        self.assertIn('any((model_path / filename).exists() for filename in weight_filenames)', text)

    def test_limits_and_languages_agree_with_upstream_constants(self):
        tree = parse('acestep/constants.py')
        self.assertEqual(set(literal(tree, 'VALID_LANGUAGES')), set(ace.VALID_LANGUAGES))
        self.assertEqual((literal(tree, 'BPM_MIN'), literal(tree, 'BPM_MAX')), (ace.BPM_MIN, ace.BPM_MAX))
        self.assertEqual((literal(tree, 'DURATION_MIN'), literal(tree, 'DURATION_MAX')),
                         (ace.DURATION_MIN, ace.DURATION_MAX))
        self.assertEqual(sorted(int(v) for v in set(ace.TIME_SIGNATURES.values())), literal(tree, 'VALID_TIME_SIGNATURES'))

    def test_dcw_strengths_are_upstreams_think_aware_defaults(self):
        tree = parse('acestep/ui/gradio/events/dcw_defaults.py')
        think, plain = literal(tree, 'THINK_DCW_DEFAULTS'), literal(tree, 'NON_THINK_DCW_DEFAULTS')
        self.assertEqual(ace.DCW_STRENGTHS[True], (think['scaler'], think['high_scaler']))
        self.assertEqual(ace.DCW_STRENGTHS[False], (plain['scaler'], plain['high_scaler']))
        self.assertEqual((think['mode'], plain['mode']), ('double', 'double'))     # the dataclass default mode, not sent
        fields = {}
        for node in class_node(parse('acestep/inference.py'), 'GenerationParams').body:
            if isinstance(node, ast.AnnAssign) and node.value is not None:
                fields[node.target.id] = node.value
        self.assertEqual(ast.literal_eval(fields['dcw_mode']), 'double')
        # GenerationParams carries only the non-Think pair; that is why the planner case is passed explicitly
        self.assertEqual((ast.literal_eval(fields['dcw_scaler']), ast.literal_eval(fields['dcw_high_scaler'])),
                         ace.DCW_STRENGTHS[False])

    def test_no_upstream_module_replaces_the_log_sink_the_worker_installs(self):
        # quiet_logs() runs before upstream is imported. A module that called logger.add/remove/configure while it
        # loads would undo it (only the command-line tool cli.py does that, and the worker never imports it).
        offenders = []
        for path in (UPSTREAM / 'acestep').rglob('*.py'):
            if re.search(r'\blogger\.(add|remove|configure)\(', path.read_text(encoding='utf-8', errors='replace')):
                offenders.append(str(path.relative_to(UPSTREAM)))
        self.assertEqual(offenders, [])

    def test_the_upstream_code_dirs_have_the_files_the_build_expects(self):
        turbo = {p.name for p in (UPSTREAM / 'acestep' / 'models' / 'xl_turbo').glob('*.py')}
        sft = {p.name for p in (UPSTREAM / 'acestep' / 'models' / 'xl_sft').glob('*.py')}
        self.assertTrue({'configuration_acestep_v15.py', 'modeling_acestep_v15_xl_turbo.py'} <= turbo)
        self.assertTrue({'configuration_acestep_v15.py', 'modeling_acestep_v15_xl_base.py', 'apg_guidance.py'} <= sft)

    def test_progress_markers_match_upstreams_strings(self):
        llm = (UPSTREAM / 'acestep' / 'llm_inference.py').read_text(encoding='utf-8')
        handler = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'generate_music.py').read_text(encoding='utf-8')
        payload = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'generate_music_payload.py').read_text(encoding='utf-8')
        self.assertIn('progress(0.1, f"Phase 1:', llm)
        self.assertIn('progress(0.5, f"Phase 2:', llm)
        self.assertIn('desc="Preparing inputs..."', handler)
        self.assertIn('desc="Preparing audio data..."', payload)

    def test_the_render_timeout_setting_is_the_one_upstream_reads(self):
        text = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'generate_music_execute.py').read_text(encoding='utf-8')
        self.assertIn('os.environ.get("ACESTEP_GENERATION_TIMEOUT", "600")', text)

    def test_gpu_memory_environment_names(self):
        text = (UPSTREAM / 'acestep' / 'gpu_config.py').read_text(encoding='utf-8')
        self.assertIn('SAVE_MEMORY_ENV = "ACESTEP_SAVE_MEMORY"', text)
        self.assertIn('os.environ.get("ACESTEP_CHECKPOINTS_DIR")', (UPSTREAM / 'acestep' / 'model_downloader.py').read_text(encoding='utf-8'))

    # ---- trained styles: what the plain-caption planner wrapper and the fold rest on (ACE_NOTES section 9)

    def test_the_planner_is_asked_by_keyword_and_nothing_else_of_it_is_read(self):
        function = function_node(parse('acestep/inference.py'), 'generate_music')
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'llm_handler'
                 and n.func.attr == 'generate_with_stop_condition']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].args, [])                       # PlainCaptionPlanner swaps a keyword, not a position
        keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(keywords['caption'], "params.caption or ''")
        touched = {n.attr for n in ast.walk(function) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id == 'llm_handler'}
        self.assertEqual(touched, {'llm_initialized', 'generate_with_stop_condition'})       # all the wrapper must forward

    def test_the_dit_is_given_the_caption_through_captions_and_only_the_planners_rewrite_changes_it(self):
        text = (UPSTREAM / 'acestep' / 'inference.py').read_text(encoding='utf-8')
        self.assertIn('dit_input_caption = params.caption\n', text)
        self.assertIn('"captions": dit_input_caption,', text)
        self.assertEqual(len(re.findall(r'^\s*dit_input_caption = ', text, re.M)), 3)     # start, planner rewrite, direct tasks
        self.assertRegex(text, r'if params\.use_cot_caption:\s+dit_input_caption = lm_generated_metadata\.get\("caption", dit_input_caption\)')
        self.assertRegex(text, r'if params\.task_type in DIRECT_CONDITIONING_TASKS:\s+dit_input_caption = ')
        self.assertIn('result = dit_handler.generate_music(**filtered_generate_kwargs)', text)

    def test_the_decoder_modules_have_the_names_the_adapter_was_trained_against(self):
        tree = parse('acestep/models/xl_sft/modeling_acestep_v15_xl_base.py')

        def assigned(class_name):
            init = function_node(tree, '__init__', inside=class_name)
            return {t.attr: ast.unparse(n.value) for n in ast.walk(init) if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self'}
        self.assertEqual(assigned('AceStepConditionGenerationModel')['decoder'], 'AceStepDiTModel(config)')
        self.assertIn('AceStepDiTLayer(config, layer_idx)', assigned('AceStepDiTModel')['layers'])
        layer = assigned('AceStepDiTLayer')
        self.assertIn('AceStepAttention(config=config, layer_idx=layer_idx)', layer['self_attn'])
        self.assertIn('AceStepAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)', layer['cross_attn'])
        attention = assigned('AceStepAttention')
        for name in ace.PROJECTIONS:
            self.assertTrue(attention[name].startswith('nn.Linear('), name)
        self.assertEqual(set(ace.PROJECTIONS) - set(attention), set())

    def test_the_adapters_key_pattern_is_what_peft_saves_for_the_decoder(self):
        lifecycle = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'lora' / 'lifecycle.py').read_text(encoding='utf-8')
        # upstream wraps model.decoder itself with PEFT, so adapter keys are relative to the decoder: layers.N....
        self.assertIn('decoder = self.model.decoder', lifecycle)
        self.assertIn('PeftModel.from_pretrained(\n                    decoder, lora_path', lifecycle)
        self.assertTrue(ace.ADAPTER_KEY.fullmatch('base_model.model.layers.31.cross_attn.o_proj.lora_B.weight'))
        self.assertIsNone(ace.ADAPTER_KEY.fullmatch('base_model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight'))

    def test_upstreams_own_unload_leaves_the_peft_layers_in_the_decoder(self):
        # Why the fold does not use add_lora / unload_lora. The probe in scratch/stage6/upstream_unload_probe.py runs this
        # code against real PEFT: after unload the output still differs from the plain model by exactly the adapter's amount.
        tree = parse('acestep/core/generation/handler/lora/lifecycle.py')
        unload = function_node(tree, 'unload_lora')
        called = {n.func.attr for n in ast.walk(unload) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertTrue({'get_base_model', 'load_state_dict'} <= called)
        self.assertFalse(called & {'unload', 'merge_and_unload', 'delete_adapter', 'disable_adapter_layers',
                                   'disable_adapter'}, called)
        self.assertEqual(ast.unparse(function_node(tree, 'add_lora')).count('self._base_decoder = {k: v.detach().cpu().clone() '
                                                                            'for k, v in state_dict.items()}'), 1)
        text = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'lora' / 'lifecycle.py').read_text(encoding='utf-8')
        self.assertIn('load_state_dict(self._base_decoder, strict=False)', text)       # the "Missing keys" lines

    def test_upstream_scales_an_adapter_only_up_to_one_and_by_multiplying_the_peft_scaling(self):
        controls = (UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler' / 'lora' / 'controls.py').read_text(encoding='utf-8')
        scaling = (UPSTREAM / 'acestep' / 'core' / 'lora' / 'scaling.py').read_text(encoding='utf-8')
        self.assertIn('scale_value = max(0.0, min(1.0, scale_value))', controls)        # 1.5 is this worker's own extension
        self.assertIn('module.scaling[adapter] = target.get("base_scale", module.scaling[adapter]) * scale', scaling)

    def test_upstream_never_freezes_the_decoder_and_loads_it_outside_inference_mode(self):
        folder = UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler'
        for path in sorted(folder.glob('init_service*.py')):
            text = path.read_text(encoding='utf-8')
            self.assertNotIn('requires_grad_(False)', text, path.name)
            self.assertNotIn('inference_mode', text, path.name)                  # so the weights are ordinary parameters
        loader = (folder / 'init_service_loader.py').read_text(encoding='utf-8')
        self.assertIn('self.model = self.model.to(device).to(self.dtype)', loader)
        self.assertIn('self.model.eval()', loader)

    def test_no_decoder_derived_state_outlives_one_generation_call(self):
        # A fold changes the weights between songs. That is only safe if nothing computed from the old weights is kept:
        # the cross-attention key/value cache is a local of generate_audio, and the model keeps no cache attributes.
        text = (UPSTREAM / 'acestep' / 'models' / 'xl_sft' / 'modeling_acestep_v15_xl_base.py').read_text(encoding='utf-8')
        self.assertIn('        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())\n        momentum_buffer', text)
        self.assertEqual(re.findall(r'self\.\w*(?:cache|past_key)\w*\s*=(?!=)', text), [])
        handler = UPSTREAM / 'acestep' / 'core' / 'generation' / 'handler'
        for path in [handler.parent.parent.parent / 'models' / 'xl_sft' / 'modeling_acestep_v15_xl_base.py',
                     *sorted(handler.glob('*.py'))]:
            if not path.name.endswith('_test.py'):                  # no captured CUDA graph holds the old weights either
                self.assertNotRegex(path.read_text(encoding='utf-8'), r'CUDAGraph|make_graphed_callables|cuda\.graph', path.name)


# ---------------------------------------------------------------- the built folder against upstream's own functions

def load_upstream_downloader():
    """The pinned model_downloader.py, loaded from its file under a private name. It imports only loguru, which is
    replaced by a do-nothing logger on a machine that does not have it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('ace_test_upstream_downloader',
                                                  UPSTREAM / 'acestep' / 'model_downloader.py')
    module = importlib.util.module_from_spec(spec)
    quiet = SimpleNamespace(**{name: (lambda *a, **k: None) for name in ('debug', 'info', 'warning', 'error')})
    stub = {} if importlib.util.find_spec('loguru') else {'loguru': SimpleNamespace(logger=quiet)}
    with mock.patch.dict(sys.modules, stub):
        spec.loader.exec_module(module)
    return module


@unittest.skipIf(UPSTREAM is None, 'no clone of the pinned ACE-Step source on this machine')
class TestLayoutPassesUpstreamsOwnChecks(CacheCase):
    """build_checkpoints with the real pinned code folders, judged by upstream's real model_downloader functions
    (on Linux, as in the image build, the links are real symbolic links)."""

    DITS = ('acestep-v15-xl-turbo', 'acestep-v15-xl-sft')

    def setUp(self):
        super().setUp()
        if not symlinks_allowed():
            patcher = mock.patch.object(ace, '_link', copying_link)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.turbo = make_dit(self.root, 'ACE-Step/acestep-v15-xl-turbo', 'a' * 40)
        self.sft = make_dit(self.root, 'ACE-Step/acestep-v15-xl-sft', 'b' * 40)
        self.main = make_main(self.root, 'c' * 40)
        self.target = Path(self.tmp.name) / 'ckpt'
        ace.build_checkpoints(self.target, self.main, {self.DITS[0]: (self.turbo, 'xl_turbo'),
                                                       self.DITS[1]: (self.sft, 'xl_sft')},
                              UPSTREAM / 'acestep' / 'models')
        self.downloader = load_upstream_downloader()

    def test_upstream_finds_every_model_it_looks_for_so_it_never_starts_a_download(self):
        self.assertTrue(self.downloader.check_main_model_exists(self.target))
        self.assertTrue(self.downloader.check_vae_exists('official', self.target))
        for name in self.DITS:
            self.assertTrue(self.downloader.check_model_exists(name, self.target), name)

    def test_upstream_sees_no_code_difference_so_its_sync_has_nothing_to_write(self):
        for name in self.DITS:
            self.assertEqual(self.downloader._check_code_mismatch(name, self.target), [], name)

    def test_the_mismatch_check_is_not_vacuous(self):
        victim = self.target / self.DITS[0] / 'modeling_acestep_v15_xl_turbo.py'
        victim.write_text('# changed\n')
        self.assertEqual(self.downloader._check_code_mismatch(self.DITS[0], self.target),
                         ['modeling_acestep_v15_xl_turbo.py'])
        victim.unlink()
        self.assertEqual(self.downloader._check_code_mismatch(self.DITS[0], self.target),
                         ['modeling_acestep_v15_xl_turbo.py'])

    def test_upstreams_own_code_sync_cannot_change_the_shared_cache(self):
        folders = (self.turbo, self.sft, self.main)
        before = {p: p.read_bytes() for folder in folders for p in folder.rglob('*') if p.is_file()}
        for name in self.DITS:
            self.assertTrue(self.downloader._sync_model_code_files(name, self.target), name)
        after = {p: p.read_bytes() for folder in folders for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        for name in self.DITS:
            self.assertEqual(self.downloader._check_code_mismatch(name, self.target), [], name)

    def test_the_code_folder_is_found_from_the_installed_package_or_from_the_clone_setting(self):
        import importlib.util
        models = UPSTREAM / 'acestep' / 'models'
        found = SimpleNamespace(submodule_search_locations=[str(UPSTREAM / 'acestep')])
        with mock.patch.object(importlib.util, 'find_spec', lambda name: found):
            self.assertEqual(ace.upstream_models_dir(), models)
        with mock.patch.object(importlib.util, 'find_spec', lambda name: None), \
                mock.patch.dict(os.environ, {'ACE_STEP_DIR': str(UPSTREAM)}):
            self.assertEqual(ace.upstream_models_dir(), models)


if __name__ == '__main__':
    unittest.main()
