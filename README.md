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

### D1
```
python3 scripts/generate_reward_data.py --lookahead-depth 1 --overwrite
```

### Train RM1
```
python scripts/train_reward_model.py \
  --dataset reward_data/d1.jsonl \
  --output-dir reward_models/rm1 \
  --normalize-target
```

### D2 using RM1
```python
python3 scripts/generate_reward_data.py \
  --lookahead-depth 2 \
  --reward-model reward_models/rm1/final \
  --overwrite
```

### D3 using RM2
```python
python3 scripts/generate_reward_data.py \
  --lookahead-depth 3 \
  --reward-model reward_models/rm2/final \
  --overwrite
```