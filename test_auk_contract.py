import unittest

from auk_contract import plan


class ContractTests(unittest.TestCase):
    def test_long_speech_keeps_every_word_in_order(self):
        words = " ".join(f"word{n}" for n in range(1201))
        for pace in (0.5, 1, 3):
            pieces = plan({"prompt": words, "pace": pace})
            self.assertEqual(" ".join(p["text"] for p in pieces), words)
            self.assertTrue(all(p["seconds"] <= 19 for p in pieces))

    def test_xml_directions_are_not_spoken(self):
        pieces = plan({"prompt": '<speak voice="warm">Hello &amp; welcome.<action>sad</action>Goodbye.<sound>rain</sound></speak>'})
        self.assertEqual([p["text"] for p in pieces], ["Hello & welcome.", "Goodbye."])
        self.assertIn("sad", pieces[1]["instruction"])

    def test_edit_retains_exact_instruction_and_duration(self):
        instruction = 'Replace "Tuesday" with "Thursday".'
        p = plan({"auk_task": "edit", "instruction": instruction,
                  "reference_voice_url": "https://example.test/a", "gen_seconds": 9})
        self.assertEqual(p[0]["instruction"], instruction)
        self.assertEqual(p[0]["seconds"], 9)

    def test_bad_requests_do_not_reach_the_model(self):
        for values in ({"auk_task": "edit"}, {"prompt": "<speak></speak>"},
                       {"prompt": "Hi", "pace": float("nan")},
                       {"prompt": "Hi", "seed": -1},
                       {"prompt": '<!DOCTYPE speak><speak>Hi</speak>'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan(values)


if __name__ == "__main__":
    unittest.main()
