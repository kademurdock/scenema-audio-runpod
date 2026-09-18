import unittest
from dataclasses import dataclass
from yue_handler import sound_controls, render_song


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


if __name__ == '__main__':
    unittest.main()
