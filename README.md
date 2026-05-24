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