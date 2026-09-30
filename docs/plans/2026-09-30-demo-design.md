# Light interactive demo

The repository Demo entry opens the project page at `#demo`. The page presents the recorded, synchronized player first and waits for an explicit Start demo click. The original MP4 remains available in the experiment details.

Use a light neutral background, white method panels, and a restrained green accent. Put playback controls above the comparison. Emphasize method names, elapsed time, transcripts, accepted draft tokens, and acoustic progress. Move configuration and measurement explanations into a closed details section. Token states use symbols as well as color.

Keep original trace data, method finishing times, and token verification unchanged. Pause, resume, reset, seek, four playback speeds, and replay remain available. A fractional range-input endpoint is normalized to the exact finish. On phones, stack the methods and use wider token chips. Preserve the old `#interactive` anchor and recording mode.

Validation: check manual start, pause stability, seek, reset, replay, all speeds, distinct method finish times, matching final transcripts, closed/open experiment details, and 320/390 px layouts in the browser. Check JavaScript syntax, local asset paths, trace contracts, and Git whitespace. Python checks are separate from this static frontend; tests needing the unavailable `qwen_asr` dependency cannot run in the current environment.
