# Provisional dropout-disabled Chinese 50M pilot

These are verified inference measurements of checkpoints trained by release 31dbce7. Training used eval mode: gradients and updates occurred, but dropout was disabled. Reevaluation does not repair that training trajectory. Do not treat this as the intended dropout-0.1 study or pool it with corrected runs.

All three weight hashes, vocabularies, reference-tokenizer behavior, full 1,000-document test coverage, log-to-checkpoint score agreement and speed protocol were checked. The scorer runs in eval mode; tokenizers were fitted on 20 MB, not 1 GB.
