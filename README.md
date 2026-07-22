## Readiness Score

```
Low=1, Medium=2, High=3
Engagement: Low=1, Medium=2, High=3
Sentiment: Negative=-1, Neutral=0, Positive=1
Then it scores:

readiness = (Motivation * 2) + Engagement + Sentiment

if readiness <= 4:
    mode = "MI"
elif readiness <= 7:
    mode = "MI-supported CBT"
else:
    mode = "CBT"
```


# LOOKAHEAD DATA GENERATION

The reward-model input is:

```
4 previous Therapist/Client pairs + final candidate Therapist utterance
```

The reward judge context is:

```
4 Therapist/Client pairs ending with the simulated Client reply
```

Use `--resume` to continue an interrupted generation run without regenerating completed patient/mode sessions. Use `--overwrite` only when you want to restart from scratch.

### D1
```
python3 scripts/generate_reward_data.py \
  --lookahead-depth 1 \
  --resume \
  --print
```

This writes:

```
reward_data/reward_training_data.jsonl
reward_data/traces/
```

### Train RM1

Recommended stable regression command:

```
python3 scripts/train_reward_model.py \
  --data reward_data/reward_training_data.jsonl \
  --output-dir reward_models/rm1 \
  --model-name microsoft/deberta-v3-large \
  --max-length 1024 \
  --normalize-target \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```

If you want to train a pairwise/ranker RM1 instead:

```
python3 scripts/train_reward_ranker.py \
  --data reward_data/reward_training_data.jsonl \
  --output-dir reward_rankers/rm1 \
  --model-name microsoft/deberta-v3-large \
  --max-length 1024 \
  --margin 0.4 \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```

Pairwise/ranker training groups candidates by `patient_id`, `mode`, and `turn`, then trains on comparisons where the score gap is at least `--margin`.

### GRPO Therapist Policy

Train a Llama-3.1-8B-Instruct therapist policy with LoRA using the trained reward model/ranker:

```
python3 scripts/train_grpo.py \
  --data reward_data/reward_training_data.jsonl \
  --reward-model reward_rankers/rm1/final \
  --output-dir grpo_models/llama_3_1_8b_lora_rm1 \
  --policy-preset llama \
  --max-prompt-length 1024 \
  --max-completion-length 160 \
  --num-generations 4 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --bf16
```

Train Qwen2.5-7B-Instruct-1M with LoRA instead:

```
python3 scripts/train_grpo.py \
  --data reward_data/reward_training_data.jsonl \
  --reward-model reward_rankers/rm1/final \
  --policy-preset qwen \
  --output-dir grpo_models/qwen_2_5_7b_1m_lora_rm1 \
  --max-prompt-length 1024 \
  --max-completion-length 160 \
  --num-generations 4 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 32 \
  --bf16
```

`--use-lora` trains small adapter weights instead of updating the full model. `--lora-r 8` is the adapter rank, and `--lora-alpha 32` controls the LoRA scaling. The Qwen preset uses `Qwen/Qwen2.5-7B-Instruct-1M`.

GRPO uses the reward model's raw output. If the regression reward model was trained with `--normalize-target`, GRPO receives the normalized score directly instead of scaling it back to `0-10`.

### D2 using RM1

Use the RM1 checkpoint for beam-search scoring:

```
python3 scripts/generate_reward_data.py \
  --lookahead-depth 2 \
  --reward-model reward_models/rm1/final \
  --resume \
  --print
```

If using the pairwise/ranker RM1:

```
python3 scripts/generate_reward_data.py \
  --lookahead-depth 2 \
  --reward-model reward_rankers/rm1/final \
  --resume \
  --print
```

This writes:

```
reward_data/reward_training_data_d2.jsonl
```

### Train RM2

Regression RM2 initialized from RM1:

```
python3 scripts/train_reward_model.py \
  --data reward_data/reward_training_data_d2.jsonl \
  --output-dir reward_models/rm2 \
  --model-name reward_models/rm1/final \
  --max-length 1024 \
  --normalize-target \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```

Pairwise/ranker RM2 initialized from ranker RM1:

```
python3 scripts/train_reward_ranker.py \
  --data reward_data/reward_training_data_d2.jsonl \
  --output-dir reward_rankers/rm2 \
  --model-name reward_rankers/rm1/final \
  --max-length 1024 \
  --margin 0.4 \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```

### D3 using RM2

Use the RM2 checkpoint for deeper beam-search scoring:

```
python3 scripts/generate_reward_data.py \
  --lookahead-depth 3 \
  --reward-model reward_models/rm2/final \
  --resume \
  --print
```

If using the pairwise/ranker RM2:

```
python3 scripts/generate_reward_data.py \
  --lookahead-depth 3 \
  --reward-model reward_rankers/rm2/final \
  --resume \
  --print
```

This writes:

```
reward_data/reward_training_data_d3.jsonl
```

### Train RM3

Regression RM3 initialized from RM2:

```
python3 scripts/train_reward_model.py \
  --data reward_data/reward_training_data_d3.jsonl \
  --output-dir reward_models/rm3 \
  --model-name reward_models/rm2/final \
  --max-length 1024 \
  --normalize-target \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```

Pairwise/ranker RM3 initialized from ranker RM2:

```
python3 scripts/train_reward_ranker.py \
  --data reward_data/reward_training_data_d3.jsonl \
  --output-dir reward_rankers/rm3 \
  --model-name reward_rankers/rm2/final \
  --max-length 1024 \
  --margin 0.4 \
  --learning-rate 1e-6 \
  --train-batch-size 1 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 16 \
  --epochs 3.0
```
