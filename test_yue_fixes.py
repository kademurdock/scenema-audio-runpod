"""Nov 10 2026 fixes, CPU only: the v2 sampling map, where a trained style is folded, which weights
the sound stage reads, and the second look after a missing model file. No model, GPU or network.

Every request field here is optional; the first class proves that a request without them is built
exactly as it was before."""
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import torch
import yue_handler as yue


@dataclass(frozen=True)
class Config:
    ode_steps: int = 32


class OneCallPipe:
    """The pipeline as render_song used it before: one call, everything inside."""
    generation_config = Config()

    def __call__(self, **kwargs):
        return self.generation_config.ode_steps, kwargs


SONG = {'style': 'pop', 'lyrics': '[Verse]\nla la', 'abc': None, 'cot': 'full', 'seed': 7}


class OldRequestsTest(unittest.TestCase):
    def test_a_request_without_the_new_fields_samples_exactly_as_before(self):
        for variation in range(0, 101):
            score, singing = yue.sampling_plan(variation)
            self.assertEqual(score, {'temperature': round(.4 + .006 * variation, 3)})
            self.assertEqual(singing, {'temperature': round(.7 + .006 * variation, 3)})
        for variation, cot, guidance, sent in ((50, 'full', 1.0, 1.0), (80, 'off', 1.0, 1.01), (20, 'off', 1.4, 1.4)):
            steps, kwargs = yue.render_song(OneCallPipe(), dict(SONG, cot=cot),
                                            {'weirdness': variation, 'steps': 40, 'guidance': guidance}, 2000)
            self.assertEqual(steps, 40)
            self.assertEqual(kwargs, dict(SONG, cot=cot, cfg_scale=sent,
                                          abc_sampling={'temperature': round(.4 + .006 * variation, 3)},
                                          semantic_sampling={'temperature': round(.7 + .006 * variation, 3),
                                                             'max_tokens': 2000}))

    def test_the_defaults_are_the_old_behaviour(self):
        self.assertEqual(yue.sampling_option({}), 'v1')
        self.assertEqual(yue.style_scope({}), ())
        self.assertEqual(yue.sound_stage_option({}), 'styled')
        self.assertTrue(all(yue.style_selected(index, ()) for index in range(28 * 7)))

    def test_bad_values_are_refused_in_plain_words(self):
        for field, bad, check in (('sampling_map', 'v3', yue.sampling_option), ('sampling_map', 2, yue.sampling_option),
                                  ('lora_modules', 'head', yue.style_scope), ('lora_layers', [3], yue.style_scope),
                                  ('lora_layers', [5, 2], yue.style_scope), ('lora_layers', [0, 1.5], yue.style_scope),
                                  ('lora_layers', [-1, 4], yue.style_scope), ('lora_layers', 'all', yue.style_scope),
                                  ('lora_sound_stage', 'clean', yue.sound_stage_option),
                                  ('lora_sound_stage', True, yue.sound_stage_option)):
            with self.subTest(field=field, bad=bad), self.assertRaisesRegex(ValueError, field):
                check({field: bad})


class SamplingV2Test(unittest.TestCase):
    def test_up_to_fifty_nothing_changes(self):
        for variation in range(0, 51):
            for fixed in (False, True):
                self.assertEqual(yue.sampling_plan(variation, 'v2', fixed), yue.sampling_plan(variation, 'v1'))

    def test_above_fifty_the_variation_goes_into_the_score_and_the_singing_stays_clear(self):
        last = 1.0
        for variation in range(51, 101):
            score, singing = yue.sampling_plan(variation, 'v2')
            old_score, old_singing = yue.sampling_plan(variation, 'v1')
            self.assertEqual(score['temperature'], old_score['temperature'], 'the score is as bold as before')
            self.assertTrue(.9 < score['top_p'] <= .95 and 30 <= score['top_k'] <= 50)
            self.assertLess(singing['temperature'], old_singing['temperature'])
            self.assertTrue(last <= singing['temperature'] <= 1.1, 'it still rises, gently')
            last = singing['temperature']
        self.assertEqual(yue.sampling_plan(100, 'v2'), ({'temperature': 1.0, 'top_p': .95, 'top_k': 50}, {'temperature': 1.1}))
        self.assertEqual(yue.sampling_plan(70, 'v2')[1], {'temperature': 1.04})
        self.assertEqual(yue.sampling_plan(70, 'v1')[1], {'temperature': 1.12})

    def test_a_fixed_score_keeps_the_singing_at_the_models_default(self):
        for variation in (51, 60, 80, 100):
            self.assertEqual(yue.sampling_plan(variation, 'v2', fixed_score=True)[1], {'temperature': 1.0})

    def test_the_numbers_are_ones_yue2_accepts(self):
        # yue2.protocol.Sampling: 0 <= temperature <= 5, 0 < top_p <= 1, top_k a whole number >= 1
        for variation in range(0, 101):
            for fixed in (False, True):
                score, singing = yue.sampling_plan(variation, 'v2', fixed)
                self.assertTrue(0 <= score['temperature'] <= 5 and 0 <= singing['temperature'] <= 5)
                self.assertTrue(0 < score.get('top_p', .9) <= 1)
                self.assertIs(type(score.get('top_k', 30)), int)

    def test_render_song_sends_the_v2_numbers(self):
        _, kwargs = yue.render_song(OneCallPipe(), dict(SONG), {'weirdness': 80, 'steps': 32, 'guidance': 1.0},
                                    sampling='v2')
        self.assertEqual(kwargs['abc_sampling'], {'temperature': .88, 'top_p': .93, 'top_k': 42})
        self.assertEqual(kwargs['semantic_sampling'], {'temperature': 1.06})
        _, kwargs = yue.render_song(OneCallPipe(), dict(SONG), {'weirdness': 80, 'steps': 32, 'guidance': 1.0}, 900,
                                    sampling='v2', fixed_score=True)
        self.assertEqual(kwargs['semantic_sampling'], {'temperature': 1.0, 'max_tokens': 900})


class StyleScopeTest(unittest.TestCase):
    """Two fake layers in the trainer's order: attention q, k, v, o then mlp gate, up, down."""

    def setUp(self):
        self.linears = [torch.nn.Linear(3, 2, bias=False) for _ in range(14)]
        self.original = [lin.weight.detach().clone() for lin in self.linears]
        self.directory = tempfile.TemporaryDirectory()
        self.file = Path(self.directory.name) / 'style.pt'
        torch.save({'rank': 1, 'lora': [t for _ in range(14) for t in (torch.ones(1, 3), torch.ones(2, 1))]}, self.file)
        self.pipe = types.SimpleNamespace(_load_model=lambda: object())
        self.patch = mock.patch.object(yue, 'style_linears', return_value=self.linears)
        self.patch.start()
        yue.STYLE.update(applied=None, plain=None)

    def tearDown(self):
        self.patch.stop()
        yue.STYLE.update(applied=None, plain=None)
        self.directory.cleanup()

    def changed(self):
        return [index for index, (lin, original) in enumerate(zip(self.linears, self.original))
                if not torch.equal(lin.weight, original)]

    def apply(self, raw):
        wanted = yue.style_request(dict(raw, lora_key='yue2-loras/test.pt'))
        scope = yue.style_scope(raw)
        yue.apply_style(self.pipe, wanted + (scope,) if scope else wanted, lambda key: str(self.file))

    def test_the_scope_reads_as_modules_and_layers(self):
        self.assertEqual(yue.style_scope({'lora_modules': 'mlp'}), ('mlp', 0, None))
        self.assertEqual(yue.style_scope({'lora_layers': [4, 27]}), ('all', 4, 27))
        self.assertEqual(yue.style_scope({'lora_modules': 'attn', 'lora_layers': (0, 0)}), ('attn', 0, 0))
        picked = [index for index in range(28 * 7) if yue.style_selected(index, ('mlp', 26, None))]
        self.assertEqual(picked, [26 * 7 + 4, 26 * 7 + 5, 26 * 7 + 6, 27 * 7 + 4, 27 * 7 + 5, 27 * 7 + 6])

    def test_only_the_chosen_part_of_the_composer_takes_the_style(self):
        self.apply({})
        self.assertEqual(self.changed(), list(range(14)), 'no scope: the whole composer, as always')
        self.apply({'lora_modules': 'mlp'})
        self.assertEqual(self.changed(), [4, 5, 6, 11, 12, 13])
        self.apply({'lora_modules': 'attn'})
        self.assertEqual(self.changed(), [0, 1, 2, 3, 7, 8, 9, 10])
        self.apply({'lora_layers': [1, 1]})
        self.assertEqual(self.changed(), list(range(7, 14)))
        self.apply({'lora_modules': 'attn', 'lora_layers': [0, 0], 'lora_scale': .5})
        self.assertEqual(self.changed(), [0, 1, 2, 3])
        for index in (0, 1, 2, 3):
            torch.testing.assert_close(self.linears[index].weight, self.original[index] + .5)
        self.assertEqual(yue.STYLE['applied'], ('yue2-loras/test.pt', .5, ('attn', 0, 0)))

    def test_switching_scope_never_accumulates_and_plain_comes_back_whole(self):
        for raw in ({'lora_modules': 'mlp'}, {}, {'lora_layers': [0, 0]}, {'lora_modules': 'mlp'}):
            self.apply(raw)
        yue.apply_style(self.pipe, None, lambda key: str(self.file))
        self.assertEqual(self.changed(), [])


class FakeRequest:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def to_dict(self):
        return dict(self.__dict__)


class StagedPipe:
    """A pipeline whose four stages can be told apart, with a log of what ran in which order."""
    generation_config = Config()
    weights = {'mot': 'm', 'vae': 'v'}
    load_timing = {}

    def __init__(self, log):
        self.log = log

    def __call__(self, **kwargs):
        self.log.append('one call')
        return types.SimpleNamespace(kind='one call', kwargs=kwargs)

    def effective_config(self, request, abc_sampling, semantic_sampling):
        return {'abc': abc_sampling, 'semantic': semantic_sampling}

    def plan(self, request, abc_sampling):
        self.log.append('plan')
        return types.SimpleNamespace(timing={}, request=request)

    def generate_semantic(self, plan, sampling):
        self.log.append('singing tokens')
        return types.SimpleNamespace(timing={}, plan=plan, sampling=sampling)

    def synthesize(self, semantic):
        self.log.append(f'sound stage at {self.generation_config.ode_steps} steps')
        return 'latents'

    def decode(self, latents):
        self.log.append('decode')
        return 'audio'


def fake_yue2():
    """Stand-ins for the three yue2 names staged_song imports; the image's own test run uses the real ones."""
    result = lambda *fields: types.SimpleNamespace(kind='staged', fields=fields, semantic=fields[2], sample_rate=fields[1])
    return {'yue2': types.ModuleType('yue2'),
            'yue2.pipeline': types.SimpleNamespace(SongResult=result),
            'yue2.storage': types.SimpleNamespace(identity=lambda value: 'identity'),
            'yue2.protocol': types.SimpleNamespace(SongRequest=FakeRequest)}


class SoundStageTest(unittest.TestCase):
    def test_without_a_callback_the_song_is_one_pipeline_call_as_before(self):
        log = []
        result = yue.render_song(StagedPipe(log), dict(SONG), {'weirdness': 50, 'steps': 32, 'guidance': 1.0})
        self.assertEqual((result.kind, log), ('one call', ['one call']))

    def test_the_plain_weights_go_back_after_the_singing_and_before_the_sound_stage(self):
        log = []
        with mock.patch.dict(sys.modules, fake_yue2()):
            result = yue.render_song(StagedPipe(log), dict(SONG, cot='off'), {'weirdness': 50, 'steps': 48, 'guidance': 1.0},
                                     1234, between=lambda: log.append('PLAIN WEIGHTS BACK'))
        self.assertEqual(log, ['plan', 'singing tokens', 'PLAIN WEIGHTS BACK', 'sound stage at 48 steps', 'decode'])
        audio, rate, semantic, latents, config, weights, timing, identity = result.fields
        self.assertEqual((audio, rate, latents, weights, identity), ('audio', 48000, 'latents', StagedPipe.weights, 'identity'))
        # the same request and sampling one pipe(...) call would have had
        self.assertEqual(semantic.plan.request.to_dict(), dict(SONG, cot='off', cfg_scale=1.01))
        self.assertEqual(semantic.sampling, {'temperature': 1.0, 'max_tokens': 1234})
        self.assertEqual(config, {'abc': {'temperature': .7}, 'semantic': {'temperature': 1.0, 'max_tokens': 1234}})
        self.assertEqual(set(timing), {'abc', 'semantic', 'nar_seconds', 'vae_seconds', 'load', 'e2e_seconds'})
        self.assertEqual(StagedPipe.generation_config.ode_steps, 32, 'the steps setting does not leak into the next song')

    def test_a_failure_between_the_stages_still_puts_the_settings_back(self):
        def boom():
            raise RuntimeError('restore failed')
        pipe = StagedPipe([])
        with mock.patch.dict(sys.modules, fake_yue2()), self.assertRaises(RuntimeError):
            yue.render_song(pipe, dict(SONG), {'weirdness': 50, 'steps': 64, 'guidance': 1.0}, between=boom)
        self.assertEqual(pipe.generation_config.ode_steps, 32)
        self.assertNotIn('sound stage at 64 steps', pipe.log)


class WholeTakeTest(unittest.TestCase):
    """The handler end to end with every outside part faked: storage, the encoder, the pipeline."""

    def run_take(self, raw, pipe_error=None):
        log, uploads = [], []
        pipe = StagedPipe(log)
        pipe.tokenizer = types.SimpleNamespace(encode=lambda text: [1] * 10)

        def one_call(**kwargs):
            log.append('one call')
            return self.result(log)
        pipe.__class__ = type('Pipe', (StagedPipe,), {'__call__': lambda self, **kwargs: one_call(**kwargs)})

        def engine():
            if pipe_error and not log.count('second look'):
                raise pipe_error
            return pipe

        def recover(wait=None):
            log.append('second look')
        client = types.SimpleNamespace(upload_file=lambda path, bucket, key, ExtraArgs=None: uploads.append(key.split('/')[-1]),
                                       generate_presigned_url=lambda *a, **k: 'https://storage.invalid/' + k['Params']['Key'])
        fakes = dict(fake_yue2(), runpod=types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None)),
                     soundfile=types.SimpleNamespace(info=lambda path: types.SimpleNamespace(duration=61.0),
                                                     write=lambda path, audio, rate, subtype=None: Path(path).write_bytes(b'f')),
                     auk_handler=types.SimpleNamespace(ffmpeg=lambda *args: Path(args[-1]).write_bytes(b'x')))
        fakes['yue2.pipeline'] = types.SimpleNamespace(SongResult=lambda *fields: self.result(log, fields[2]))
        with mock.patch.dict(sys.modules, fakes), mock.patch.object(yue, 'engine', engine), \
                mock.patch.object(yue, 'storage', lambda: (client, 'bucket')), mock.patch.object(yue, 'resolve_models', lambda **k: None), \
                mock.patch.object(yue, 'recover_models', recover), mock.patch.object(yue, 'gpu_name', lambda: 'Test GPU'), \
                mock.patch.object(yue, 'apply_style', lambda pipe, wanted, fetch: log.append(('style', wanted))):
            return yue.handler({'input': raw}), log, uploads

    @staticmethod
    def result(log, semantic='tokens'):
        def save(work):
            Path(work).mkdir(parents=True, exist_ok=True)
            (Path(work) / 'audio.flac').write_bytes(b'f')
        return types.SimpleNamespace(save_artifacts=save, abc=None, truncated={'abc': False, 'semantic': False},
                                     semantic=semantic, sample_rate=48000)

    STYLED = {'style': 'kdtest, in the style of kdtest. English, soul', 'lyrics': '[verse]\nla la', 'cot': 'off', 'seed': 5,
              'lora_key': 'yue2-loras/test.pt', 'lora_scale': .8}

    def test_a_plain_song_and_an_old_styled_song_are_one_call_and_say_nothing_new(self):
        out, log, uploads = self.run_take({'style': 'pop', 'lyrics': '[Verse]\nla la', 'seed': 5})
        self.assertNotIn('error', out)
        self.assertEqual(log, [('style', None), 'one call'])
        self.assertFalse({'sampling_map', 'lora_sound_stage', 'lora_scope', 'styled_stage_url'} & set(out))
        out, log, uploads = self.run_take(self.STYLED)
        self.assertEqual(log, [('style', ('yue2-loras/test.pt', .8)), 'one call'])
        self.assertEqual(out['lora_sound_stage'], 'styled')
        self.assertEqual(sorted(uploads), ['master.mp3', 'master.wav'])
        for feature in ('sampling-v2', 'sound-stage', 'style-scope'):
            self.assertIn(feature, out['features'])

    def test_a_plain_sound_stage_restores_the_weights_between_the_stages(self):
        out, log, uploads = self.run_take(dict(self.STYLED, lora_sound_stage='plain', sampling_map='v2',
                                               lora_modules='mlp', lora_layers=[2, 27]))
        self.assertNotIn('error', out)
        scoped = ('yue2-loras/test.pt', .8, ('mlp', 2, 27))
        self.assertEqual(log, [('style', scoped), 'plan', 'singing tokens', ('style', None), 'sound stage at 32 steps', 'decode'])
        self.assertEqual((out['lora_sound_stage'], out['sampling_map'], out['lora_scope']), ('plain', 'v2', ['mlp', 2, 27]))
        self.assertNotIn('styled_stage_url', out)

    def test_both_saves_a_second_master_of_the_same_tokens_through_the_styled_stage(self):
        out, log, uploads = self.run_take(dict(self.STYLED, lora_sound_stage='both', steps=48))
        self.assertNotIn('error', out)
        wanted = ('yue2-loras/test.pt', .8)
        self.assertEqual(log, [('style', wanted), 'plan', 'singing tokens', ('style', None), 'sound stage at 48 steps', 'decode',
                               ('style', wanted), 'sound stage at 48 steps', 'decode'])
        self.assertIn('master-styled-stage.mp3', uploads)
        self.assertTrue(out['styled_stage_url'].endswith('/master-styled-stage.mp3'))
        self.assertIn('styled_stage_s', out['timing'])

    def test_the_sound_stage_choice_does_nothing_without_a_style(self):
        out, log, _ = self.run_take({'style': 'pop', 'lyrics': '[Verse]\nla la', 'seed': 5, 'lora_sound_stage': 'both'})
        self.assertEqual(log, [('style', None), 'one call'])
        self.assertNotIn('lora_sound_stage', out)

    def test_a_missing_model_file_gets_one_second_look_and_then_the_song(self):
        out, log, _ = self.run_take(self.STYLED, pipe_error=FileNotFoundError(2, 'No such file', '/cache/snapshots/abc/model.safetensors'))
        self.assertNotIn('error', out)
        self.assertEqual(log[0], 'second look')
        self.assertEqual(log.count('one call'), 1)

    def test_a_file_still_missing_is_named_in_the_error_and_not_tried_a_third_time(self):
        looks = []
        error = FileNotFoundError(2, 'No such file', '/cache/models--m-a-p--YuE2-3B/snapshots/abc123/model.safetensors')

        def engine():
            raise error
        fakes = {'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=lambda *a: None)),
                 'soundfile': types.SimpleNamespace(), 'auk_handler': types.SimpleNamespace(ffmpeg=None)}
        with mock.patch.dict(sys.modules, fakes), mock.patch.object(yue, 'engine', engine), \
                mock.patch.object(yue, 'storage', lambda: (None, 'bucket')), mock.patch.object(yue, 'resolve_models', lambda **k: None), \
                mock.patch.object(yue, 'recover_models', lambda wait=None: looks.append(1)):
            out = yue.handler({'input': self.STYLED})
        self.assertEqual(len(looks), 1)
        self.assertEqual(out, {'error': 'YuE2 could not finish (FileNotFoundError, abc123/model.safetensors). Your writing is saved.'})

    def test_any_other_failure_reads_as_it_always_did(self):
        out, log, _ = self.run_take(self.STYLED, pipe_error=RuntimeError('CUDA out of memory at https://secret.invalid/x'))
        self.assertEqual(out, {'error': 'YuE2 could not finish (RuntimeError). Your writing is saved.'})
        self.assertNotIn('second look', log)


class SecondLookTest(unittest.TestCase):
    def snapshot(self, hub, name='a' * 40):
        folder = hub / 'models--m-a-p--YuE2-3B' / 'snapshots' / name
        folder.mkdir(parents=True)
        (folder / 'config.json').write_text('{}')
        return folder

    def test_a_snapshot_with_a_download_still_open_is_not_whole(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self.snapshot(Path(td))
            self.assertTrue(yue._complete(folder))
            blobs = folder.parent.parent / 'blobs'
            blobs.mkdir()
            (blobs / 'abc.incomplete').write_text('')
            self.assertFalse(yue._complete(folder))
            self.assertTrue(yue._filled(folder), 'the first look would have taken it')
            (blobs / 'abc.incomplete').unlink()
            self.assertTrue(yue._complete(folder))
            self.assertFalse(yue._complete(folder / 'nothing-here'))

    def test_a_link_to_a_file_that_has_not_arrived_is_not_whole(self):
        with tempfile.TemporaryDirectory() as td:
            folder = self.snapshot(Path(td))
            try:
                os.symlink(str(Path(td) / 'blob-not-there'), str(folder / 'model.safetensors'))
            except (OSError, NotImplementedError):
                self.skipTest('this machine does not allow symbolic links')
            self.assertFalse(yue._complete(folder))

    def test_the_second_look_waits_for_a_cache_that_is_still_arriving(self):
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td)
            folder = self.snapshot(hub)
            blobs = folder.parent.parent / 'blobs'
            blobs.mkdir()
            (blobs / 'abc.incomplete').write_text('')
            naps = []

            def nap(seconds):
                naps.append(seconds)
                if len(naps) == 2:
                    (blobs / 'abc.incomplete').unlink()  # the cache finished arriving
            env = {'YUE_MODEL_DIR': str(folder), 'HF_HUB_CACHE': str(hub)}
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(yue.RESOLVED, {'done': False}), \
                    mock.patch.object(yue, 'MODELS', (('YUE_MODEL_DIR', 'm-a-p/YuE2-3B'),)):
                os.environ.pop('YUE_MODEL_DIR_PINNED', None)
                yue.resolve_models(download=False, strict=True, wait=60, sleep=nap)
                self.assertEqual(naps, [5, 5])
                self.assertTrue(yue.RESOLVED['done'])
                self.assertEqual(os.environ['YUE_MODEL_DIR'], str(folder))

    def test_a_cache_that_never_arrives_is_reported_missing_not_waited_for_forever(self):
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td)
            folder = self.snapshot(hub)
            blobs = folder.parent.parent / 'blobs'
            blobs.mkdir()
            (blobs / 'abc.incomplete').write_text('')
            env = {'YUE_MODEL_DIR': str(folder), 'HF_HUB_CACHE': str(hub)}
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(yue.RESOLVED, {'done': False}), \
                    mock.patch.object(yue, 'MODELS', (('YUE_MODEL_DIR', 'm-a-p/YuE2-3B'),)):
                os.environ.pop('YUE_MODEL_DIR_PINNED', None)
                yue.resolve_models(download=False, strict=True, wait=0, sleep=lambda s: self.fail('no wait was allowed'))
                self.assertFalse(yue.RESOLVED['done'])

    def test_the_pinned_folder_is_remembered_after_the_variable_points_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            hub = Path(td)
            other = self.snapshot(hub, 'b' * 40)
            pinned = hub / 'models--m-a-p--YuE2-3B' / 'snapshots' / ('c' * 40)
            env = {'YUE_MODEL_DIR': str(pinned), 'HF_HUB_CACHE': str(hub)}
            with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(yue.RESOLVED, {'done': False}), \
                    mock.patch.object(yue, 'MODELS', (('YUE_MODEL_DIR', 'm-a-p/YuE2-3B'),)):
                os.environ.pop('YUE_MODEL_DIR_PINNED', None)
                yue.resolve_models(download=False)
                self.assertEqual(os.environ['YUE_MODEL_DIR'], str(other))
                pinned.mkdir()
                (pinned / 'config.json').write_text('{}')  # the pinned snapshot arrived later
                yue.RESOLVED['done'] = False
                yue.resolve_models(download=False, strict=True, wait=0)
                self.assertEqual(os.environ['YUE_MODEL_DIR'], str(pinned))

    def test_recover_models_drops_the_pipeline_and_the_style_record(self):
        seen = []
        with mock.patch.object(yue, 'PIPE', 'old pipeline'), mock.patch.dict(yue.STYLE, {'applied': ('k', 1.0), 'plain': [1]}), \
                mock.patch.dict(yue.RESOLVED, {'done': True}), \
                mock.patch.object(yue, 'resolve_models', lambda **k: seen.append((k, yue.RESOLVED['done'], yue.PIPE))):
            yue.recover_models(wait=3)
            self.assertEqual(seen, [({'strict': True, 'wait': 3}, False, None)])
            self.assertEqual(yue.STYLE, {'applied': None, 'plain': None})

    def test_the_missing_file_is_named_shortly_and_safely(self):
        self.assertEqual(yue.missing_file(FileNotFoundError(2, 'x', '/a/b/snapshots/abc/model.safetensors')), 'abc/model.safetensors')
        self.assertEqual(yue.missing_file(FileNotFoundError('no filename')), '')
        self.assertEqual(yue.missing_file(FileNotFoundError(2, 'x', '/tmp/a b?/we ird&.pt')), 'a_b_/we_ird_.pt')
        self.assertLessEqual(len(yue.missing_file(FileNotFoundError(2, 'x', '/a/' + 'z' * 300))), 80)


if __name__ == '__main__':
    unittest.main()
