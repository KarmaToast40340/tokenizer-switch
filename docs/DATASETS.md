# Dataset evaluation

Checked on 2026-10-24 from the Hugging Face dataset cards. Sizes marked *est.* are our
estimates; `scripts/prepare_data.py` and `scripts/encode_corpus.py` print the exact numbers.

## Summary

| | `codeparrot/codeparrot-clean` | `wikimedia/wikipedia` `20231101.en` | `wikimedia/wikipedia` `20231101.zh` |
|---|---|---|---|
| Content | Python files from GitHub | English articles | Chinese articles |
| Rows | 5,361,373 files | 6.41M articles | 1.38M articles |
| Size | ~50 GB of code (12.8 GB of files on the hub) | ~20 GB of text (*est.*) | a few GB of text (*est.*) |
| Splits | `train` (validation is a separate repo) | `train` only | `train` only |
| Fields | `content`, `repo_name`, `path`, `license`, `hash`, `copies`, ... | `id`, `url`, `title`, `text` | same |
| License | per file (`license` column, 15 values: MIT, Apache-2.0, BSD, GPL-2/3, ...) | CC BY-SA 3.0 and GFDL | CC BY-SA 3.0 and GFDL |
| Cleaning by the authors | exact dedup; mean line length < 100; max line length < 1000; alphanumeric fraction > 0.25; keyword-based removal of auto-generated files | `mwparserfromhell`, references and other "unwanted sections" removed | same |

## Fit to the proposal's budgets

What we need per domain: about 1 GB for tokenizer training (up to 100% of one source in the grid),
plus enough text for 200M training tokens without repeating data.

| Domain | 200M tokens with GPT-2 | 200M tokens with a domain tokenizer | Available | Verdict |
|---|---|---|---|---|
| Code | ~0.6 GB (*est.* ~3 bytes/token) | ~0.9 GB (*est.* NSL ~0.7) | ~50 GB | ample: we take 1.5 GB |
| English | (only used in tokenizer mixes) | | ~20 GB | ample: we take 1.2 GB |
| Chinese | ~0.3 GB (*est.* ~1.5 bytes/token) | ~0.8 GB (*est.* ~4 bytes/token) | a few GB | enough: we take 1.5 GB, the script warns if the stream runs out |

Both datasets therefore cover the 50M and 200M token budgets in under one epoch.

## Issues and what the code does about them

### codeparrot-clean

1. **Near-duplicates remain.** Only exact duplicates were removed, so forks, vendored libraries and
   copies of the same file with small edits are still there (the `copies` column counts them).
   *Effect:* a test file can have a near-copy in the training set, which flatters bpb.
   *Mitigation:* `prepare_data.py` keeps every file of a test repository out of `train.jsonl`
   and removes exact duplicates again. Near-dedup (MinHash) is not done; it is optional at our scale.
2. **Not The Stack.** The proposal says "Python files from The Stack". codeparrot-clean comes from the
   GitHub dump on BigQuery, not from The Stack; the proposal text should be updated.
3. **Mixed licenses (including GPL).** Fine for research; we do not redistribute the data, and the repo
   ignores `data/`.
4. **Old code.** The snapshot predates 2022 and includes Python 2 files. Not a problem for compression
   or bpb, but worth mentioning if MBPP (Python 3) is reported.
5. **Possible benchmark contamination.** GitHub code may contain MBPP or HumanEval solutions. Only
   matters if we report MBPP.
6. **Order.** The stream follows the order of the shards (by repository). `prepare_data.py` uses a
   seeded shuffle buffer of 10,000 files.

### Wikipedia English and Chinese

1. **Chinese mixes traditional and simplified characters**, depending on the article's editors. We
   convert everything to simplified with OpenCC `t2s` before any split. This converts characters, not
   regional vocabulary (Taiwan and mainland terms remain), which is acceptable for our purpose.
2. **Short and list articles.** Many stubs and list pages; `--min-chars 200` drops the shortest ones.
3. **Markup artefacts.** Tables and templates are removed by `mwparserfromhell`; empty section titles and
   some leftovers remain. Not harmful for tokenizer training or bpb.
4. **Article order.** The dump is ordered by id; the shuffle buffer avoids training a tokenizer on one
   period of article creation.
5. **License.** CC BY-SA 3.0: attribution if we ever publish data or samples.

## Recommendation

Both datasets fit the project. Use `codeparrot-clean` for code (Python only, already filtered, large
enough for every budget) and `wikimedia/wikipedia` `20231101.en` and `20231101.zh` for English and
Chinese, with the held-out split, repository grouping and OpenCC conversion done in `prepare_data.py`.
