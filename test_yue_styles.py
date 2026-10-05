"""Small CPU tensor checks for adapters; no model download or GPU required."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yue_handler as yue


class StyleWeightsTest(unittest.TestCase):
    def setUp(self):
        self.linears = [torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(2, 3, bias=False)]
        self.original = [lin.weight.detach().clone() for lin in self.linears]
        self.checkpoint = {'rank': 1, 'targets': 'ar self_attn qkvo + mlp gate/up/down',
                           'lora': [torch.ones(1, 3), torch.ones(2, 1), torch.ones(1, 2), torch.ones(3, 1)]}
        self.directory = tempfile.TemporaryDirectory()
        self.file = Path(self.directory.name) / 'style.pt'
        self.pipe = types.SimpleNamespace(_load_model=lambda: object())
        self.patch = mock.patch.object(yue, 'style_linears', return_value=self.linears)
        self.patch.start()
        yue.STYLE.update(applied=None, plain=None)

    def tearDown(self):
        self.patch.stop()
        yue.STYLE.update(applied=None, plain=None)
        self.directory.cleanup()

    def apply(self, scale, checkpoint=None):
        torch.save(checkpoint or self.checkpoint, self.file)
        wanted = ('yue2-loras/test.pt', scale) if scale else None
        yue.apply_style(self.pipe, wanted, lambda key: str(self.file))

    def test_scale_switch_and_plain_restore_do_not_accumulate(self):
        for scale in (1, .5, .25, 1, 0):
            self.apply(scale)
            for lin, original in zip(self.linears, self.original):
                torch.testing.assert_close(lin.weight, original + scale)

    def test_bad_later_tensor_cannot_partially_replace_previous_style(self):
        self.apply(.5)
        for corrupt in ('shape', 'nan', 'rank'):
            bad = copy.deepcopy(self.checkpoint)
            if corrupt == 'shape':
                bad['lora'][-1] = torch.ones(4, 1)
            elif corrupt == 'nan':
                bad['lora'][-1][0, 0] = float('nan')
            else:
                bad['rank'] = 2
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.apply(.75, bad)
            for lin, original in zip(self.linears, self.original):
                torch.testing.assert_close(lin.weight, original + .5)
            self.assertEqual(yue.STYLE['applied'], ('yue2-loras/test.pt', .5))

    def test_same_shape_decoder_files_are_rejected(self):
        for metadata in ({'kind': 'DECODER (NAR, sound stage) LoRA'}, {'io': {}},
                         {'head': '/training/head.pt'}, {'targets': 'nar_self_attn'}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'sound-stage'):
                self.apply(1, {**self.checkpoint, **metadata})
            for lin, original in zip(self.linears, self.original):
                torch.testing.assert_close(lin.weight, original)

    def test_overflow_restores_every_plain_weight(self):
        self.apply(.5)
        bad = copy.deepcopy(self.checkpoint)
        bad['lora'][-1].fill_(3e38)
        bad['lora'][-2].fill_(3e38)
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.apply(.75, bad)
        for lin, original in zip(self.linears, self.original):
            torch.testing.assert_close(lin.weight, original)
        self.assertIsNone(yue.STYLE['applied'])
        self.apply(.5)
        for lin, original in zip(self.linears, self.original):
            torch.testing.assert_close(lin.weight, original + .5)


if __name__ == '__main__':
    unittest.main()
