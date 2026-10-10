"""Unit tests for ace_handler.py: CPU only, no torch, no network, no ffmpeg.

Run (isolated Python, from this folder):   python -I -m unittest discover -s . -p "test_ace_handler.py"
(python -I -m unittest test_ace_handler.py cannot find the module in isolated mode; the discover form can.)

A fake pipeline stands in for upstream's generate_music, a fake client for B2 and fake functions for ffmpeg and the
WAV writer. TestPinnedUpstream additionally reads the pinned upstream source (if a clone is on this machine) and
checks that every name this worker sends to it exists there; it is skipped when no clone is found."""
import ast
import contextlib
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
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ace_handler as ace  # noqa: E402

KEY_MP3 = re.compile(r'^ace/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/master\.mp3$')
KEY_WAV = re.compile(r'^ace/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/master\.wav$')
SECRET_CAPTION = 'ZEBRA-CAPTION-SENTINEL for the log check'
SECRET_LYRICS = 'KANGAROO-LYRICS-SENTINEL for the log check'


def reset_state():
    ace.STATE.update(ckpt=None, built=set(), dit={}, llm=None, planner_tries=0, api=None, env=False)
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
        self.assertEqual(ace.scrub(None), '')

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
        self.assertEqual((request['batch_size'], request['vocal_language'], request['thinking']), (1, 'en', True))
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
                                     'lyrics': None, 'model': None, 'mode': None})
        self.assertEqual((request['model'], request['batch_size'], request['thinking']), ('xl-turbo', 1, True))

    def test_mode_and_model(self):
        self.check('Mode must be render or probe.', mode='fast')
        self.check('Model must be xl-turbo or xl-sft.', model='xl-base')
        self.check('Model must be xl-turbo or xl-sft.', model='')
        self.check('Model must be xl-turbo or xl-sft.', model=3)

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
        self.assertTrue(out['url'].startswith('https://storage.example/ace/'))
        self.assertIn('ttl=604800', out['url'])
        self.assertIn('ttl=604800', out['wav_url'])
        self.assertEqual(out['duration_s'], 90.0)
        self.assertEqual(out['bytes'], 10)
        self.assertEqual(out['seed'], 7)
        self.assertIsInstance(out['processing_ms'], int)
        self.assertIs(out['truncated'], False)
        self.assertEqual(out['gpu'], 'Fake A40')
        self.assertEqual(out['memory'], {'render_peak_gib': 12.5})
        self.assertEqual(out['features'], ['ace-xl-turbo', 'ace-xl-sft', 'batch', 'instrumental', 'probe'])
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
        self.assertEqual(params.vocal_language, 'en')

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


class TestAdapterSeam(Harness):
    def test_lora_key_is_refused_before_any_work(self):
        out = self.run_job(lora_key='ace-loras/style.pt')
        self.assertEqual(out, {'error': 'Custom ACE styles are not available yet.'})
        self.engine.assert_not_called()
        self.planner.assert_not_called()
        self.assertEqual(self.api.calls, [])

    def test_any_value_other_than_null_or_empty_is_refused(self):
        for value in ('x', 0, False, True, ['a'], {'a': 1}):
            self.assertEqual(self.run_job(lora_key=value), {'error': ace.ADAPTER_SENTENCE}, repr(value))

    def test_null_or_empty_lora_key_is_the_same_as_none(self):
        for value in (None, ''):
            self.assertNotIn('error', self.run_job(lora_key=value))

    def test_probe_refuses_it_too(self):
        out = silently(ace.handler, {'input': {'mode': 'probe', 'lora_key': 'x'}})
        self.assertEqual(out, {'error': ace.ADAPTER_SENTENCE})

    def test_apply_adapter_is_a_seam_that_raises_today(self):
        with self.assertRaises(NotImplementedError) as caught:
            ace.apply_adapter(object(), 'ace-loras/style.pt')
        self.assertEqual(str(caught.exception), 'Custom ACE styles are not available yet.')

    def test_the_handler_maps_apply_adapter_but_not_other_not_implemented_errors(self):
        self.engine.side_effect = NotImplementedError(ace.ADAPTER_SENTENCE)
        self.assertEqual(self.run_job(), {'error': ace.ADAPTER_SENTENCE})
        self.engine.side_effect = NotImplementedError('operator not implemented for this device')
        self.assertEqual(self.run_job(),
                         {'error': 'ACE-Step could not finish (NotImplementedError). Your writing is saved.'})


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


class TestFetchModel(unittest.TestCase):
    def test_offline_mode_is_restored_after_a_download(self):
        seen = {}
        constants = SimpleNamespace(HF_HUB_OFFLINE=True)

        def snapshot_download(repo, revision=None, cache_dir=None, ignore_patterns=None):
            seen.update(repo=repo, revision=revision, cache_dir=cache_dir, ignore=ignore_patterns,
                        env=os.environ['HF_HUB_OFFLINE'], flag=constants.HF_HUB_OFFLINE)
            return '/tmp/hf-hub/models--x--y/snapshots/abc'
        fake = SimpleNamespace(constants=constants, snapshot_download=snapshot_download)
        with mock.patch.dict(sys.modules, {'huggingface_hub': fake}), mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
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
            fake = SimpleNamespace(constants=constants, snapshot_download=lambda *a, **k: '/tmp/x')
            env = {} if restored is None else {'HF_HUB_OFFLINE': restored}
            with mock.patch.dict(sys.modules, {'huggingface_hub': fake}), mock.patch.dict(os.environ, env):
                if restored is None:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                ace.fetch_model('x/y', None)
                self.assertEqual(os.environ['HF_HUB_OFFLINE'], restored if restored is not None else '1')
            self.assertIs(constants.HF_HUB_OFFLINE, expected, restored)

    def test_a_failed_download_returns_none_and_restores_offline_mode(self):
        constants = SimpleNamespace(HF_HUB_OFFLINE=True)

        def snapshot_download(*args, **kwargs):
            raise ConnectionError('no route')
        fake = SimpleNamespace(constants=constants, snapshot_download=snapshot_download)
        buffer = io.StringIO()
        with mock.patch.dict(sys.modules, {'huggingface_hub': fake}), mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}), \
                contextlib.redirect_stdout(buffer):
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
        return 'Error initializing model: boom\n\nTraceback:\n...', False


class FakeLlm:
    created = []

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
        return 'Error initializing 5Hz LM: boom', False


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
        self.assertIn('planner load failed', buffer.getvalue())
        self.assertNotIn('Traceback', buffer.getvalue())

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
        self.assertEqual(added[0][1], {'level': 'WARNING'})

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


class TestMisc(unittest.TestCase):
    def test_features_and_versions(self):
        self.assertEqual(ace.FEATURES, ('ace-xl-turbo', 'ace-xl-sft', 'batch', 'instrumental', 'probe'))
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
        for sentence in (ace.ADAPTER_SENTENCE, ace.TOO_BIG, ace.FAILED):
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


if __name__ == '__main__':
    unittest.main()
