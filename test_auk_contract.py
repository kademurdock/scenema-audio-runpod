import unittest

from auk_contract import plan, model_instruction


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
        self.assertEqual(model_instruction(p[0], has_reference=True), instruction)

    def test_reference_speech_never_appends_voice_or_stage_directions(self):
        import json
        pieces = plan({"prompt": '<speak voice="distinct accent and youthful timbre">' +
                       ('A synthetic sentence for the speaker. ' * 24) +
                       '<action>emotion and studio instructions</action>She said, &quot;Good morning.&quot;</speak>'})
        self.assertGreater(len(pieces), 2)
        for piece in pieces:
            instruction = model_instruction(piece, has_reference=True)
            self.assertEqual(instruction, 'Say the following with the same voice: ' + json.dumps(piece['text'], ensure_ascii=False))
            self.assertNotIn('distinct accent', instruction)
            self.assertNotIn('studio instructions', instruction)
            self.assertNotIn('following description', instruction)

    def test_designed_voice_has_separate_description_and_spoken_content(self):
        piece = plan({"prompt": 'She said, "Hello."', "voice_description": 'Warm with a Southern accent'})[0]
        self.assertEqual(model_instruction(piece), 'Generate speech based on the following description: "Warm with a Southern accent". The content to speak is: "She said, \\"Hello.\\"".')

    def test_voice_sample_keeps_the_description_away_from_heard_speech(self):
        import json
        script = ('<speak voice="A small child with a thick accent. Studio audio.">'
                  'The thunder rolled outside like a warning drum. I stood at the window. ' +
                  ('Another synthetic sentence for the speaker. ' * 30) + '</speak>')
        plain = plan({"prompt": script})
        sampled = plan({"prompt": script, "voice_sample": True})
        self.assertTrue(sampled[0]["sample"])
        self.assertEqual(sampled[0]["text"], "The thunder rolled outside like a warning drum. I stood at the window.")
        self.assertIn("thick accent", sampled[0]["instruction"])
        heard = [p for p in sampled if not p.get("sample")]
        self.assertEqual(heard, plain)
        self.assertEqual(" ".join(p["text"] for p in heard), " ".join(p["text"] for p in plain))
        for piece in heard:
            instruction = model_instruction(piece, has_reference=True)
            self.assertEqual(instruction, 'Say the following with the same voice: ' + json.dumps(piece['text'], ensure_ascii=False))
            self.assertNotIn('thick accent', instruction)

    def test_voice_sample_is_off_unless_asked_and_never_with_a_clip_or_edit(self):
        script = '<speak voice="Warm">Hello there, friend. How are you today?</speak>'
        self.assertFalse(any(p.get("sample") for p in plan({"prompt": script})))
        self.assertFalse(any(p.get("sample") for p in plan({"prompt": script, "voice_sample": "yes"})))
        with_clip = plan({"prompt": script, "voice_sample": True, "reference_voice_url": "https://example.test/a"})
        self.assertFalse(any(p.get("sample") for p in with_clip))
        edit = plan({"auk_task": "edit", "instruction": "Make it brighter.", "voice_sample": True,
                     "reference_voice_url": "https://example.test/a"})
        self.assertFalse(any(p.get("sample") for p in edit))

    def test_voice_sample_of_a_short_line_is_the_whole_line(self):
        sampled = plan({"prompt": "Hi.", "voice_sample": True})
        self.assertEqual([p["text"] for p in sampled], ["Hi.", "Hi."])
        self.assertTrue(sampled[0]["sample"])
        self.assertNotIn("sample", sampled[1])

    def test_bad_requests_do_not_reach_the_model(self):
        for values in ({"auk_task": "edit"}, {"prompt": "<speak></speak>"},
                       {"prompt": "Hi", "pace": float("nan")},
                       {"prompt": "Hi", "seed": -1},
                       {"prompt": '<!DOCTYPE speak><speak>Hi</speak>'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                plan(values)


if __name__ == "__main__":
    unittest.main()
