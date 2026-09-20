import unittest
from dataclasses import dataclass
from yue_handler import sound_controls, render_song, request_input, style_request


@dataclass(frozen=True)
class Config:
    ode_steps: int = 32


class Pipeline:
    generation_config = Config()

    def __call__(self, **kwargs):
        return self.generation_config.ode_steps, kwargs


class ControlsTest(unittest.TestCase):
    def test_defaults_preserve_release_sound(self):
        pipe = Pipeline()
        steps, args = render_song(pipe, {'seed': 17}, sound_controls({}))
        self.assertEqual(steps, 32)
        self.assertEqual(args, {'seed': 17, 'cfg_scale': 1.0,
                               'abc_sampling': {'temperature': .7},
                               'semantic_sampling': {'temperature': 1.0}})

    def test_overrides_reach_pipeline_and_reset(self):
        pipe = Pipeline()
        controls = sound_controls({'weirdness': 100, 'steps': 64, 'guidance': 2})
        steps, args = render_song(pipe, {'seed': 42}, controls)
        self.assertEqual(steps, 64)
        self.assertEqual(args['cfg_scale'], 2)
        self.assertEqual(args['semantic_sampling']['temperature'], 1.3)
        self.assertEqual(pipe.generation_config.ode_steps, 32)
        class Broken(Pipeline):
            def __call__(self, **kwargs):
                raise RuntimeError('fixture')
        broken = Broken()
        with self.assertRaises(RuntimeError):
            render_song(broken, {}, controls)
        self.assertEqual(broken.generation_config.ode_steps, 32)

    def test_bad_controls_are_rejected_before_gpu_load(self):
        for key, value in [('steps', 0), ('steps', 65), ('steps', 16.5),
                           ('weirdness', 101), ('weirdness', True),
                           ('guidance', float('nan')), ('guidance', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                sound_controls({key: value})


class StyleTest(unittest.TestCase):
    def test_score_free_song_keeps_upstream_guidance(self):
        _, args = render_song(Pipeline(), {'seed': 1, 'cot': 'off'}, sound_controls({}))
        self.assertEqual(args['cfg_scale'], 1.01)

    def test_style_key_and_strength_are_checked(self):
        self.assertIsNone(style_request({}))
        self.assertEqual(style_request({'lora_key': 'yue2-loras/kids-step1200.pt'}), ('yue2-loras/kids-step1200.pt', 1.0))
        for bad in ({'lora_key': '../secrets.pt'}, {'lora_key': 'yue2-loras/a.pt', 'lora_scale': 9}):
            with self.assertRaises(ValueError):
                style_request(bad)

    def test_score_free_refuses_a_cover(self):
        ok = request_input({'style': 'soulful', 'lyrics': 'la', 'cot': 'off'})
        self.assertEqual(ok['cot'], 'off')
        with self.assertRaises(ValueError):
            request_input({'style': 'soulful', 'lyrics': 'la', 'cot': 'off', 'abc': 'X:1'})


if __name__ == '__main__':
    unittest.main()
