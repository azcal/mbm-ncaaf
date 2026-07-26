# NCAAF rebuild v1

Data: cfbfastR schedule mirrors (CFBD-sourced), 2016 burn-in, 2018-2025 modeled.
Evaluation: FBS-vs-FBS only (n=798 / 808). FCS opponents update states from a low
Elo init but never enter evaluation. Neutral sites zero the home edge. Conference
tiers read per-game, so realignment handles itself.

Winner head (frozen holdouts):
  2024: LL 0.5539 / Brier 0.1890 / hit 68.7%  (own Elo 0.5808, CFBD Elo 0.5780, base 0.6790)
  2025: LL 0.5359 / Brier 0.1816 / hit 70.9%  (own Elo 0.5480, CFBD Elo 0.5470, base 0.6752)
Beats CFBD's own published pregame Elo on both holdouts.

Reference heads (all clear floors, printed on board):
  margin MAE 12.66 / 12.37 vs Elo 13.64 / 13.54 vs constant 15.71 / 15.89
  total  MAE 12.95 / 13.08 vs constant 13.39 / 13.50

Design: margin-of-victory Elo (K=38, HA=45, carry=0.75), decayed opponent-adjusted
scoring (fast/slow), conference-tier gap, rest, neutral/conference flags.
Known v1 gaps, flagged: no injury/portal/QB-change inputs (college reporting is thin
and unvalidatable); early-season portal churn is why the Week 4 calibration gate
matters most on this board. 2026 = live forward test from Week 1 (late August).
