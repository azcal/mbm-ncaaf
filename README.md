# MBM NCAAF daily engine (rebuild v1)

Standalone predictive model, no market inputs. Boards FBS-vs-FBS only. Three heads,
all floor-validated: winner P(home), model margin, model total. Frozen through 2025;
2026 is the forward test. Holdouts: 2024 LL 0.5539 (68.7%), 2025 LL 0.5359 (70.9%),
beating CFBD's own pregame Elo both years.

2026 data comes from ESPN with a verified name-alias map (all 136 FBS programs matched
against the actual 2026 schedule; unmapped opponents are treated as FCS and excluded
from boards). CALIBRATION GATE: predictions print from Week 1, value lines unlock
Week 4 (portal churn makes early college weeks the least trustworthy in the stack).
Runs daily 13:00 UTC plus a Saturday pre-slate refresh. Tripwire: last-200 vs base rate.
