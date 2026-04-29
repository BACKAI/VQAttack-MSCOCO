from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an ALBEF VQAttack config for one MSCOCO shard.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--answer-list", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--bert-config", default="configs/config_bert.json")
    parser.add_argument("--image-res", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--k-test", type=int, default=128)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = f"""train_file: ['{args.eval_json.as_posix()}']
test_file: ['{args.eval_json.as_posix()}']
answer_list: '{args.answer_list.as_posix()}'

vqa_root: '{args.image_root.as_posix()}'
vg_root: '{args.image_root.as_posix()}'
image_res: {args.image_res}
batch_size_train: {args.batch_size}
batch_size_test: {args.batch_size}
k_test: {args.k_test}

alpha: 0.4
distill: True
warm_up: True

eos: '[SEP]'

bert_config: '{args.bert_config}'

optimizer: {{opt: adamW, lr: 2e-5, weight_decay: 0.02}}
schedular: {{sched: cosine, lr: 2e-5, epochs: 8, min_lr: 1e-6, decay_rate: 1, warmup_lr: 1e-5, warmup_epochs: 4, cooldown_epochs: 0}}
"""
    args.output.write_text(config, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
