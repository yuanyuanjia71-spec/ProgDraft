# Development

Thank you for your interest in ProgDraft. This repository contains the implementation of acoustic progress propagation and its Random-K[3,8] training recipe.

## Local checks

Install the [development environment](docs/installation.md), then run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The complete test suite requires the ASR extra, but not a GPU, target weights or audio data. The tests cover teacher forcing, recurrent positions and KV, loss reduction, CE gradients into the predictor, FA mapping, EOS handling and target-verifier branches.

## Reporting an issue

Include the commit, model scale, command, environment and a minimal description of the unexpected behavior. For reproduction questions, include asset checksums and whether you used exported historical caches or newly prepared caches. Share dataset/sample IDs where possible rather than licensed audio or private transcripts.

## Changes

Keep the two configurations and their documented numerical contracts explicit. Changes to model behavior, token indexing, training loss, acoustic coordinates or decoding should include an appropriate contract check. Documentation changes should preserve working relative links and describe the actual command-line interfaces.

Do not commit local weights, caches, audio, run outputs or credentials. These assets are intentionally separate from the source release. The project license remains TBD; see [the README](README.md#acknowledgments-and-license).
