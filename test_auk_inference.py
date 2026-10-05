"""No model downloads: a failed sampler must not poison a reused engine."""
import types
import unittest
from unittest.mock import patch

import auk_handler as worker


class Cache:
    def __init__(self):
        self.text_cond = None
        self.text_uncond = None

    def clear_cache(self):
        self.text_cond = None
        self.text_uncond = None


class ReusedEngine:
    def __init__(self):
        self.model = types.SimpleNamespace(transformer=Cache())
        self.fail_next = True
        self.seen = []

    def generate(self, messages, **kwargs):
        cache = self.model.transformer
        # AuK reuses these projections within ODE sampling and does not
        # verify whether an existing projection belongs to this request.
        if cache.text_cond is None:
            cache.text_cond = messages
        if cache.text_uncond is None:
            cache.text_uncond = 'unconditional projection'
        self.seen.append(cache.text_cond)
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError('synthetic sampling failure')
        return cache.text_cond, kwargs


class InferenceTests(unittest.TestCase):
    def test_failed_request_cannot_condition_the_next_request(self):
        runtime = ReusedEngine()
        with patch.object(worker, 'engine', return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, 'synthetic sampling failure'):
                worker.generate('first instruction', seed=1)
            self.assertIsNone(runtime.model.transformer.text_cond)
            self.assertIsNone(runtime.model.transformer.text_uncond)
            result = worker.generate('different instruction', seed=2)
        self.assertEqual(runtime.seen, ['first instruction', 'different instruction'])
        self.assertEqual(result, ('different instruction', {'seed': 2}))
        self.assertIsNone(runtime.model.transformer.text_cond)
        self.assertIsNone(runtime.model.transformer.text_uncond)

    def test_clears_preexisting_conditioning_before_sampling(self):
        runtime = ReusedEngine()
        runtime.fail_next = False
        runtime.model.transformer.text_cond = 'stale instruction'
        runtime.model.transformer.text_uncond = 'stale unconditional projection'
        with patch.object(worker, 'engine', return_value=runtime):
            result = worker.generate('current instruction', nfe=32, cfg_strength=2.0)
        self.assertEqual(result, ('current instruction', {'nfe': 32, 'cfg_strength': 2.0}))
        self.assertIsNone(runtime.model.transformer.text_cond)
        self.assertIsNone(runtime.model.transformer.text_uncond)


if __name__ == '__main__':
    unittest.main()
