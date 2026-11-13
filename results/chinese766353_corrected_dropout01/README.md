# Chinese 50M checkpoint comparison

Corrected Chinese 50M; dropout 0.1; 20 MB tokenizer pilots

Fresh matched dropout-0.1 training with mode checks on every training phase. Single seed, 20 MB tokenizer fits; not the proposed 1 GB study.

All weight hashes, vocabularies, reference-tokenizer behavior, full 1,000-document test coverage, log-to-checkpoint score agreement and speed protocol were checked. Speed: 20 shared 200-character prefixes, 128 greedy new tokens, synchronized CUDA timing.
