Chinese runs reuse the same YAML files with overrides, e.g.

    python scripts/train.py configs/train/E_progressive.yaml run_name=E_progressive_zh \
        domain=wiki_zh new_tokenizer=artifacts/tokenizers/wiki_zh70_gpt4_32k new_name=wiki_zh70_gpt4_32k
