# Private bootstrap-reference diagnostics

The normal speech path makes a private voice-setting sample before the audible
performance. That reference used to disappear at the end of the worker job, so a
failed output could not be compared with the actual conditioning sample.

`AUK_DIAGNOSTIC_USER_ID` is absent by default. An operator may set it to one
verified owner's 24-character account ID. Only a speech job whose trusted bridge
`out_prefix` exactly matches that ID, whose private voice sample is enabled, and
which has no imported reference can retain the actual cropped conditioning WAV.
Request text and request-side retention flags cannot turn this on. Editing and
other accounts retain their existing behavior.

The optional sample is a private sibling of that take's WAV and MP3 in the same
B2 bucket. It never enters the audible concatenation or gallery asset event, and
no prompt, sample, URL or user detail is printed to worker logs. The worker's
completed response supplies its owner ID, key, SHA-256, duration and seven-day
signed URL. The bridge continues projecting ordinary playback fields, so these
diagnostics do not appear in the gallery or ordinary Sound Booth API. An operator
must match the completed worker response to that authenticated owner's job before
reading it through the private provider API.
Link expiry does not delete the object: it follows the ordinary stored-master
retention policy. Operators must remove it with the take's private stored files
when it is no longer needed; do not enable this under a claimed deletion policy
that has not been verified.

This consumes no extra generation or transcription call. It is diagnostic
preparation, not a demonstrated cure for wrong words or missed edits. It does not
upload data for training. The feature is not enabled or deployed by this source
change. Existing projects and takes are not replayed.
