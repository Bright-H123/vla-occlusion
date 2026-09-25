# Day 2 — scripted expert gate

Seeds 1000..1099, one episode each, produced by `scripts/run_expert.py --n 100`.

**Success: 100/100** (100%, Wilson 95% interval [96%, 100%]). Gate is 90/100. **PASS.**

| outcome | count |
|---|---|
| success | 100 |

Mean episode length 84 control steps (20 Hz). Mean re-stages per episode 0.69. 0.14 s wall clock per episode.

## Null controls on the success criterion

Seeds 5000..5049, disjoint from the gate seeds, produced by `scripts/null_controls.py --n 50`.

| policy | success | expected |
|---|---|---|
| random actions | 0/50 | ~0 |
| zero actions | 0/50 | 0 |
| expert pushing a distractor | 0/50 | 0 |
| expert pushing the instructed cube | 50/50 | ~all |

Criterion **valid**: it is not satisfied by chance, and it is bound to the instructed cube rather than to any cube reaching the goal.
