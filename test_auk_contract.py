import unittest

from auk_contract import plan, model_instruction, edit_windows


class ContractTests(unittest.TestCase):
    def test_edit_windows_preserve_range_and_duration(self):
        for total in (0.2, 14, 42, 251.73):
            windows = edit_windows(total, {})
            self.assertEqual(windows[0]['start'], 0)
            self.assertEqual(windows[-1]['end'], total)
            self.assertAlmostEqual(sum(w['seconds'] for w in windows), total)
            for i, w in enumerate(windows):
                self.assertLessEqual(w['end'] - w['start'] + w['seconds'], 28.00001)
                if i:
                    self.assertEqual(windows[i - 1]['end'], w['start'])
        windows = edit_windows(90, {'edit_start': 10, 'edit_end': 30, 'gen_seconds': 40})
        self.assertEqual(windows[0]['start'], 10)
        self.assertEqual(windows[-1]['end'], 30)
        self.assertAlmostEqual(sum(w['seconds'] for w in windows), 40)

    def test_bad_edit_range_rejected(self):
        for values in ({'edit_start': -1}, {'edit_end': 91}, {'edit_start': 10, 'edit_end': 5},
                       {'edit_start': 90}, {'edit_start': 90.01, 'edit_end': 90.02},
                       {'gen_seconds': float('nan')}, {'gen_seconds': 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                edit_windows(90, values)

    def test_rounded_end_uses_actual_source_duration(self):
        windows = edit_windows(90, {'edit_start': 89, 'edit_end': 90.02})
        self.assertEqual(windows, [{'start': 89, 'end': 90, 'seconds': 1}])

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

    def test_plain_voice_change_uses_timbre_template_without_changing_requested_traits(self):
        cases = (
            ('Turn this voice into a man.', 'a man'),
            ('Make this voice sound like a twelve year old child.', 'a twelve year old child'),
            ('Please change the voice to an older woman with a Southern accent.', 'an older woman with a Southern accent'),
            ('Convert this speaking voice into a light, clear youthful voice!', 'a light, clear youthful voice'),
        )
        for instruction, description in cases:
            with self.subTest(instruction=instruction):
                piece = plan({'auk_task': 'edit', 'instruction': instruction,
                              'reference_voice_url': 'https://example.test/a'})[0]
                self.assertEqual(piece['instruction'], instruction)
                self.assertEqual(model_instruction(piece, has_reference=True),
                                 'Keep the spoken content unchanged and change the timbre to: ' +
                                 '"' + description + '".')

    def test_edit_preparation_leaves_other_tasks_and_combined_requests_intact(self):
        instructions = (
            'Keep the spoken content unchanged and change the timbre to: "a youthful voice".',
            'Replace "Tuesday" with "Thursday".',
            'Change the emotion to happy.',
            'Give this voice a strong Irish accent.',
            'Make this voice sound like a child. Remove the noise.',
            'Make this voice sound like a child, then replace "Tuesday" with "Thursday".',
            'Make this voice sound like a child and sing different words.',
            'Make this voice sound like a child, lower the pitch by two semitones.',
            'Make this voice sound like   .',
        )
        for instruction in instructions:
            with self.subTest(instruction=instruction):
                self.assertEqual(model_instruction({'instruction': instruction}, has_reference=True), instruction)

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
