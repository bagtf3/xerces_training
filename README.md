## What this is
Training repo for the Xerces chess engine. This allows training via selfplay, history Xerces games and lczero training data via a vendored adapter.

## Included
- `chunkparser.py` — vendored / adapted from LeelaChessZero (LCZero).

## License
This repo is distributed under the **GNU General Public License v3.0**
(GPL-3.0). See the `LICENSE` file at the repo root for the full text.

## Vendored files & attribution
The following file(s) are copied or adapted from other projects:
`lczero-training` and remain under GPL-3.0:

- `xerces_training/chunkparser.py`  (adapted from LCZero)

## Quick usage
Smoke-test the parser on the included sample chunk (doesnt work yet):

```bash
python run_parser.py sample_data/small.chunk
```

