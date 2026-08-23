# picks.csv

One row per published pick from the MustBeMoose NCAAF model. Rebuilt on every run.

This is a **forecast**, not a bet recommendation. No price is compared anywhere in this
file, so whether a probability beats the odds at your book is entirely your call.

## Reading a pick

This board picks a WINNER, not a spread. `pick` is the team; `pick_claim` is always
`wins outright`; `pick_prob` is that team's chance of winning the game.

| column | meaning |
|---|---|
| `pick_team` | the team being picked |
| `pick_side` | whether that team is `home` or `away` |
| `pick_claim` | always `wins outright` on this board |
| `pick_prob` | P(`pick_team` wins), 0 to 1 |
| `model_spread` | the model's own line on the **home** team, negative meaning home favoured, rounded to a quotable half point. Not a market number; this board quotes moneylines. |
| `model_pick_line` | the same number written the way a book writes it: `-6`, `-12.5`, `PK` |
| `model_total` | the model's projected combined score |
| `home_win_prob` | P(home team wins), the same number as `pick_prob` when `pick_side` is home |
| `pick_result` | `win`, `loss`, `push`, or empty until played |
| `cal` | true inside the calibration window, before Week 4 |

## Grading

`wins outright` rows score on the game result: the pick wins when `pick_team` wins.
`final_margin` is home score minus away score, so flip its sign for an away pick.

## What the probability means

`pick_prob` is calibrated, not a confidence score. A set of rows reading 0.70 should win
about 70% of the time.
