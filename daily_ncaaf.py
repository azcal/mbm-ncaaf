"""
MBM NCAAF daily runner (rebuild v1, frozen spec).
- Historical base: cfbfastR schedule parquets (2016-2025, committed)
- 2026 season: ESPN scoreboard feed, team names mapped to CFBD via committed alias
- Boards FBS-vs-FBS games (7-day horizon): p(home), model margin, model total
- CALIBRATION GATE: value lines suppressed before Week 4
- Grades logged predictions, rolling metrics + last-200 tripwire, Discord alerts
- Top-N divergence filter retired 2026-08-03 against its own holdout; see PUSH_FILTER
Usage: python daily_ncaaf.py [--date YYYY-MM-DD]
"""
import argparse, json, math, os, sys, time, datetime as dt
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tg_delta
import board_format as BF
import freshness as FR
import notify
import engine_ncaaf as E

# Shared files must be at least this new. A minimum rather than an exact match, so a later
# improvement to a shared file does not force an edit here.
#   2026-08-03  board_format.line() field separator. The longest FBS matchup string is
#               "Jacksonville State @ Florida International" at 42 characters, so without
#               it the widest row on every board renders as an unstyled run-on line.
SHARED_MIN = "2026-08-03"
for _mod in (BF, notify, tg_delta, FR):
    _have = getattr(_mod, "VERSION", "")
    if _have < SHARED_MIN:
        raise ImportError(
            f"{_mod.__name__}.py is version {_have or 'unstamped'}, this runner needs "
            f"{SHARED_MIN} or newer. Copy the current shared files into this repo.")

# ESPN 403s ANY custom User-Agent and only serves the requests default. Measured
# 2026-08-05 on the WNBA scoreboard: no header 200, "Mozilla/5.0" 403, an honest
# identifier with contact details 403, a full browser header set 403. mbm-wnba has
# always worked because it never set one. Do not "fix" this by adding a UA.
ESPN_H = {}
H = ESPN_H
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
ALIAS = json.load(open(os.path.join(HERE, "data", "espn_alias.json")))
CONF = json.load(open(os.path.join(HERE, "data", "conf_latest.json")))
TRAIN_LO, TRAIN_HI = 2018, 2025
CUR_SEASON = 2026
CAL_UNLOCK_WEEK = SPEC.get("cal_unlock_week", 4)
# Tracks board_format rather than restating it. This is the console-side copy, and a
# local 0.05 here while board_format ran at 0.00 meant the Actions log and Discord
# quoted different prices for the same game.
MIN_EDGE = BF.MIN_EDGE

# Eight days catches two Saturdays. Measured on 2023-25, the busiest eight-day window
# holds 121 FBS-vs-FBS games against 72 for seven days, so seven guarantees exactly one
# Saturday and halves the board without losing a slate.
HORIZON_DAYS = SPEC.get("horizon_days", 7)

# --- Week 1 early window --------------------------------------------------------------
# A date, not a wider horizon. A horizon big enough to reach the opening Saturday would
# still be that big in November, where an eight-day window already holds 121 games.
# This reaches one fixed date and then stops mattering: past it the board is back on
# HORIZON_DAYS with no edit and nothing to remember to undo.
#
# 2026 opens 29 August with 8 games, then the real Week 1 lands 3-7 September with 91 more.
# A seven-day horizon on 23 August reaches 30 August and catches 8 of 99.
#
# Empty or absent disables it.
EARLY_UNTIL = SPEC.get("early_until")


def board_until(today):
    """Last game date to board. HORIZON_DAYS normally, reaching to EARLY_UNTIL while the
    window is open. max() rather than a swap, so the window can only widen the board."""
    hi = today + pd.Timedelta(days=HORIZON_DAYS)
    if EARLY_UNTIL and today <= pd.Timestamp(EARLY_UNTIL):
        hi = max(hi, pd.Timestamp(EARLY_UNTIL))
    return hi

# The top-N-per-day divergence filter, now OFF by default and kept only as a switch.
#
# Measured on the 2024 and 2025 frozen holdouts, restricted to the 28 dates where the
# filter actually binds: pushed games read log loss 0.5587 and 69.1% hit, dropped games
# 0.5046 and 73.5%. The difference in per-game log loss is +0.054 against the pushed side,
# t=+2.00, p=0.045. The model is LESS accurate on the games the rule selects.
#
# The rule's stated claim still holds, that pushed games beat the Elo floor by more,
# +0.0251 against +0.0088. But that gap is there because Elo is even worse on divergent
# games, not because the model is better on them, and what decides whether a price beats a
# book is how trustworthy the number is. Same rule was tested and dropped on the props
# board for the same reason.
#
# `selected` is still computed and logged either way, so the rule stays under measurement
# by the second tripwire below without deciding what gets pushed. Set push_filter true in
# spec.json to put it back in charge.
PUSH_FILTER = SPEC.get("push_filter", False)
PUSH_PER_DAY = SPEC.get("push_per_day", 20)

# Runaway guard only. A seven-day board tops out near 72 games, so this does not bind in
# normal operation; it exists so a feed glitch cannot fire fifty Discord embeds.
MAX_BLOCKS = SPEC.get("max_blocks", 90)


DATA_DICT = """# picks.csv

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
| `model_pick_line` | the model's line written the way a book writes it, on **`pick_team`**: `-6`, `-12.5`, `PK`. Opposite sign to `model_spread` whenever the pick is the away side, because that column is on the home team. |
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
"""

PICKS_COLS = ["date", "game_id", "season", "week", "away", "home",
              "pick", "pick_team", "pick_side", "pick_claim", "pick_prob",
              "model_spread", "model_pick_line", "model_total", "home_win_prob",
              "cal", "final_margin", "pick_result"]


def write_picks(log):
    """The publishable file: one row per published pick, nothing else.

    Separate from predictions.csv on purpose. That file feeds graded.csv and the dashboard
    repo and must keep its shape; this one is built for a reader who has never seen the
    project. Only rows with a pick, only columns that mean something to an outsider.

    Rebuilt in full from the log every run rather than appended to, because the log already
    knows which rows exist and which have settled. Sorted, and probabilities rounded, so an
    unchanged pick produces a byte-identical line and the git history stays readable.
    """
    if "pick_team" not in log.columns:
        return
    p = log[log.pick_team.notna() & (log.pick_team.astype(str) != "")].copy()
    if not len(p):
        print("picks.csv: no published picks in the log yet")
        return
    p = p.drop_duplicates("game_id", keep="last")
    # NCAAF logs the margin as `result`; the published file calls it final_margin so a
    # reader does not have to guess whether "result" means the score or the pick's outcome.
    p["final_margin"] = pd.to_numeric(p.get("result"), errors="coerce")
    p["home_win_prob"] = pd.to_numeric(p.get("p_home"), errors="coerce")
    p["pick_claim"] = "wins outright"
    p["pick"] = p.pick_team.astype(str)
    mp = np.where(p.pick_side.values == "home", p.final_margin.values, -p.final_margin.values)
    p["pick_result"] = np.where(~np.isfinite(mp), "",
                                np.where(mp > 0, "win", np.where(mp < 0, "loss", "push")))
    for c in ("pick_prob", "home_win_prob"):
        p[c] = pd.to_numeric(p[c], errors="coerce").round(4)
    # Half points, because that is how a spread is quoted. The Ridge returns -6.07 and
    # -1.95, which are not lines anybody posts and imply a precision the model does not
    # have. predictions.csv keeps the raw number for analysis; the published file rounds.
    #
    # Rounding decides which side of a key number the quote lands on, and in college the
    # keys are flatter than the NFL's: measured on 17,423 games 2021-25, margin 3 is 8.1%,
    # 7 is 7.2%, 14 is 4.1%, 10 and 21 both 3.8%. So -3.4 quoting as -3.5 and -2.6 as -2.5
    # is a real difference, and nearest-half-point is the honest way to resolve it rather
    # than nudging toward a key the model did not ask for.
    p["model_spread"] = (pd.to_numeric(p.get("model_spread"), errors="coerce") * 2
                         ).round() / 2 + 0.0
    p["model_total"] = (pd.to_numeric(p.get("model_total"), errors="coerce") * 2).round() / 2
    # On the PICKED team, which is what the column name promises. model_spread stays on
    # the home team, as its own docstring says, so the two disagree in sign on away picks
    # and that is correct rather than a mistake.
    p["model_pick_line"] = [
        BF.handicap(v if sd == "home" else -v) if v == v else ""
        for v, sd in zip(p.model_spread, p.pick_side)]
    for c in PICKS_COLS:
        if c not in p.columns:
            p[c] = ""
    p = p[PICKS_COLS].sort_values(["date", "game_id"]).reset_index(drop=True)
    p.to_csv(os.path.join(LOGS, "picks.csv"), index=False)
    open(os.path.join(LOGS, "picks_README.md"), "w").write(DATA_DICT)
    g = int((p.pick_result.astype(str) != "").sum())
    print(f"picks.csv: {len(p)} pick(s), {g} graded, {len(p) - g} pending")


def min_odds(p):
    p = min(max(float(p), 0.01), 0.99)
    return math.ceil((1 + MIN_EDGE) / p * 100) / 100.0

def load_hist():
    rows = []
    for y in range(2016, 2026):
        s = pd.read_parquet(os.path.join(HERE, "data", f"sched_{y}.parquet"))
        s = s[(s.completed == True) & s.home_points.notna() & s.away_points.notna()]
        s = s[(s.home_division == "fbs") | (s.away_division == "fbs")]
        rows.append(s)
    g = pd.concat(rows, ignore_index=True)
    g["date"] = E.game_day(g.start_date)
    return g

def fetch_2026():
    rows = []
    months = dt.date.today().month
    types = [(2, range(1, 19))] + ([(3, range(1, 6))] if months in (12, 1) else [])
    for st, weeks in types:
        for wk in weeks:
            try:
                r = requests.get(ESPN, params={"dates": CUR_SEASON, "seasontype": st, "week": wk,
                                               "groups": 80, "limit": 400}, headers=H, timeout=60).json()
            except Exception:
                continue
            for e in r.get("events", []):
                c = e["competitions"][0]
                t = {x["homeAway"]: x for x in c["competitors"]}
                if "home" not in t or "away" not in t:
                    continue
                hl, al = t["home"]["team"]["location"], t["away"]["team"]["location"]
                hn, an = ALIAS.get(hl, hl), ALIAS.get(al, al)
                done = bool(e["status"]["type"].get("completed"))
                state = e["status"]["type"].get("state")
                rows.append(dict(game_id=str(e["id"]), season=CUR_SEASON, week=wk,
                    season_type="regular" if st == 2 else "postseason",
                    start_date=e.get("date"), neutral_site=bool(c.get("neutralSite")),
                    conference_game=bool(c.get("conferenceCompetition")),
                    home_team=hn, away_team=an,
                    home_division="fbs" if hl in ALIAS else "fcs",
                    away_division="fbs" if al in ALIAS else "fcs",
                    home_conference=CONF.get(hn), away_conference=CONF.get(an),
                    home_points=float(t["home"].get("score")) if done else np.nan,
                    away_points=float(t["away"].get("score")) if done else np.nan,
                    completed=done, state=state))
            time.sleep(0.15)
    d = pd.DataFrame(rows).drop_duplicates("game_id")
    if len(d):
        d["date"] = E.game_day(d.start_date)
    return d

def run(today_str):
    today = pd.Timestamp(today_str)
    hist = load_hist()
    cur = fetch_2026()
    done26 = cur[cur.completed] if len(cur) else cur
    # `~completed` is not enough: a game being played right now is completed=False and
    # would be boarded at a pre-game price. state is "pre", "in" or "post".
    _pre = cur.state.eq("pre") if "state" in cur.columns else ~cur.completed
    _hi = board_until(today)
    up = cur[_pre & (cur.date >= today) & (cur.date <= _hi)
             & (cur.home_division == "fbs") & (cur.away_division == "fbs")] if len(cur) else pd.DataFrame()
    if EARLY_UNTIL and today <= pd.Timestamp(EARLY_UNTIL):
        print(f"early window open: boarding through {pd.Timestamp(EARLY_UNTIL):%Y-%m-%d} "
              f"({len(up)} FBS game(s)); reverts to a {HORIZON_DAYS}-day horizon by itself")
    g = pd.concat([hist, done26, up], ignore_index=True)
    g["margin"] = g.home_points - g.away_points
    g["total"] = g.home_points + g.away_points
    g["fbs_both"] = (g.home_division == "fbs") & (g.away_division == "fbs")
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    f = E.add_diffs(E.build(g, CFG))
    tr = (f.season >= TRAIN_LO) & (f.season <= TRAIN_HI) & f.fbs_both & f.margin.notna()
    sc = StandardScaler().fit(f.loc[tr, E.W].values)
    mdl = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(f.loc[tr, E.W].values), f.loc[tr, "y"].values)
    trm = f.loc[tr]
    scm = StandardScaler().fit(trm[E.MG].values)
    mm = Ridge(alpha=5.0).fit(scm.transform(trm[E.MG].values), trm.margin.values)
    sct = StandardScaler().fit(trm[E.TT].values)
    mt = Ridge(alpha=5.0).fit(sct.transform(trm[E.TT].values), trm.total.values)
    base = float(f.loc[tr, "y"].mean())

    alerts = []
    os.makedirs(LOGS, exist_ok=True)
    pred_path = os.path.join(LOGS, "predictions.csv")
    log = pd.read_csv(pred_path, dtype={"game_id": str}) if os.path.exists(pred_path) else \
        pd.DataFrame(columns=["date", "game_id", "week", "season", "home", "away",
                              "p_home", "p_elo", "selected", "model"])
    if "selected" not in log.columns:
        log["selected"] = np.nan          # rows logged before the filter was tracked

    board = None
    if len(up):
        rows = f[f.game_id.isin(up.game_id.astype(str)) | f.game_id.isin(up.game_id)].copy()
        p = mdl.predict_proba(sc.transform(rows[E.W].values))[:, 1]
        margins = mm.predict(scm.transform(rows[E.MG].values))
        totals = mt.predict(sct.transform(rows[E.TT].values))
        ha = np.where(rows.neutral == 1, 0, CFG["HA"])
        pe = 1/(1+10**(-((rows.h_elo + ha) - rows.a_elo)/400))
        board = pd.DataFrame({"date": pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d"),
                              "game_id": rows.game_id.astype(str), "week": rows.week,
                              "season": rows.season, "home": rows.home, "away": rows.away,
                              "p_home": p, "p_elo": pe, "model": "v1"})
        board = board.assign(margin_m=margins, total_m=totals, week_v=rows.week.values,
                             div=(board.p_home - board.p_elo).abs())
        disp = board.sort_values("div", ascending=False).groupby("date").head(PUSH_PER_DAY) \
                    .sort_values(["date", "div"], ascending=[True, False])
        # Persist what the divergence rule WOULD pick. Full-slate grading watches the
        # probabilities and is structurally blind to a selection rule, so the rule
        # needs its own record whether or not it is currently in charge.
        board["selected"] = board.game_id.isin(disp.game_id).astype(int)
        # margin_m and total_m were computed on every run and then dropped by this list.
        # They are the model's own spread and total, the two things a downstream reader
        # most wants and the two the Collective's board renders, so they are kept now.
        # model_spread is the margin as a HANDICAP on the home team: a home side favoured
        # by 7 is model_spread -7, so the sign flips.
        board["model_spread"] = -board.margin_m
        board["model_total"] = board.total_m
        board["cal"] = board.week < CAL_UNLOCK_WEEK
        board["pick_side"] = np.where(board.p_home >= .5, "home", "away")
        board["pick_team"] = np.where(board.p_home >= .5, board.home, board.away)
        board["pick_prob"] = np.maximum(board.p_home, 1 - board.p_home)
        keep = ["date", "game_id", "week", "season", "home", "away", "p_home", "p_elo",
                "selected", "model", "model_spread", "model_total", "cal",
                "pick_side", "pick_team", "pick_prob"]
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board[keep]], ignore_index=True)
    else:
        print(f"no FBS games on or before {_hi:%Y-%m-%d}")

    res_map = g[g.margin.notna()].assign(gid=g.loc[g.margin.notna(), "game_id"].astype(str)) \
        .set_index("gid")["margin"]
    log["result"] = log.game_id.map(res_map)
    log.to_csv(pred_path, index=False)
    write_picks(log)
    dec = log[log.result.notna() & (log.result != 0)]
    if len(dec) >= 30:
        y = (dec.result > 0).astype(int).values
        print(f"\nrolling grade, full slate (n={len(y)}):")
        for lbl, pp in (("model v1 ", dec.p_home), ("elo floor", dec.p_elo),
                        ("base rate", pd.Series(np.full(len(y), base)))):
            pc = np.clip(pp.values.astype(float), 1e-6, 1-1e-6)
            print(f"{lbl}: LL={log_loss(y, pc):.4f}  Brier={brier_score_loss(y, pc):.4f}  "
                  f"hit={np.mean((pc > .5) == (y == 1)):.3f}")
        w = dec.tail(200)
        if len(w) >= 200:
            yy = (w.result > 0).astype(int).values
            edge = log_loss(yy, np.full(len(yy), base)) - \
                log_loss(yy, np.clip(w.p_home.values.astype(float), 1e-6, 1-1e-6))
            print(f"last-200 edge vs base: {edge:+.4f}  [{'OK' if edge > 0 else 'TRIPWIRE: bench review'}]")
            if edge <= 0:
                alerts.append("NCAAF winner")

        # ---- second tripwire: the SELECTION RULE, not the model ----
        # The rule no longer decides what gets pushed (see PUSH_FILTER), but it is
        # still marked on every row so it keeps being graded on live data. Retired
        # against a holdout, kept under measurement, same as MLB F5 and the CS2
        # series-length head.
        sel = dec[dec.selected == 1]
        uns = dec[dec.selected == 0]
        if len(sel) >= 50 and len(uns) >= 50:
            def edge_vs_elo(d):
                yy = (d.result > 0).astype(int).values
                pm_ = np.clip(d.p_home.values.astype(float), 1e-6, 1-1e-6)
                pf_ = np.clip(d.p_elo.values.astype(float), 1e-6, 1-1e-6)
                return log_loss(yy, pf_) - log_loss(yy, pm_), log_loss(yy, pm_)
            e_sel, ll_sel = edge_vs_elo(sel)
            e_uns, ll_uns = edge_vs_elo(uns)
            print(f"\nselection rule (top {PUSH_PER_DAY}/day by divergence):")
            print(f"  pushed     n={len(sel):>4}  LL={ll_sel:.4f}  edge vs Elo={e_sel:+.4f}")
            print(f"  not pushed n={len(uns):>4}  LL={ll_uns:.4f}  edge vs Elo={e_uns:+.4f}")
            ws = sel.tail(200)
            if len(ws) >= 100:
                e_w, _ = edge_vs_elo(ws)
                ok = e_w > 0
                print(f"  last-{len(ws)} pushed, edge vs Elo: {e_w:+.4f}  "
                      f"[{'OK' if ok else 'TRIPWIRE: selection rule not beating the floor'}]")
                if not ok:
                    alerts.append("NCAAF selection rule")
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    push, skipped, cal_n = {}, 0, 0
    if board is not None:
        source = disp if PUSH_FILTER else board.sort_values(["date", "div"],
                                                            ascending=[True, False])
        # r.date, not today_str. Every row used to print the run date, so a 15-day early
        # window rendered forty-eight games all dated today and a reader could not tell an
        # opening Saturday fixture from one nine days out.
        # The handicap sits on the HOME team: margin_m is home minus away, so -margin_m is
        # the home side's line. Pairing that with the PICKED team inverts it whenever the
        # away side is picked, which published "Miami +18" for a Miami team the model has
        # winning by 18. Flip it when the pick is on the road.
        rows = []
        for _, r in source.iterrows():
            home_pick = r.p_home >= .5
            hand = (-float(r.margin_m)) if home_pick else float(r.margin_m)
            rows.append((str(r.date)[:10], f"{r.away} @ {r.home}",
                         r.home if home_pick else r.away,
                         max(r.p_home, 1 - r.p_home), r.game_id, r.week,
                         BF.handicap(round(hand * 2) / 2 + 0.0)))
        wg = max([len(x[1]) for x in rows] + [10])
        # Pad the whole "TEAM -6.5" token, not just the handicap. Padding the number alone
        # left the win column ragged, because the team names differ by fifteen characters.
        wp = max([len(f"{x[2]} {x[6]}") for x in rows] + [10])
        for d_, gm, pk, q, gid, wk, spr in rows:
            cal = wk < CAL_UNLOCK_WEEK
            if not cal and BF.min_odds(q) < BF.PRICE_FLOOR:
                skipped += 1
                continue
            if cal:
                # notify.FORECAST, which names the team and carries no price. The previous
                # version matched HEADER and read "forecast 62.2%" with no indication of
                # WHICH side that 62.2% belonged to, which is the one thing a forecast row
                # has to say. The model's own line rides along, so the row states a number
                # as well as a lean.
                ln = (f"{d_}  {gm:<{wg}}  {(pk + ' ' + spr):<{wp}}  win {q:.1%}\n"
                      f"      note: week {int(wk)} calibration, no betting until "
                      f"Week {CAL_UNLOCK_WEEK}")
                cal_n += 1
            else:
                ln = BF.line(d_, gm, pk, q, w_game=wg, w_pick=wp)
                note = BF.live_note("NCAAF", q)
                if note:
                    ln = "\n".join([ln] + note)
            push[gid] = {"line": ln, "min_odds": BF.min_odds(q),
                         # Carries the pick, so a side flipping between the morning and
                         # afternoon runs is reported even when the price barely moves.
                         "match": pk, "cal": bool(cal)}
    print("\n=== MBM NCAAF board ===")
    for _v in push.values():
        print(_v["line"])
    if skipped:
        print(f"({skipped} game{'s' if skipped != 1 else ''} skipped, "
              f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)})")

    mode, lines = tg_delta.gate(LOGS, today_str.replace("-", ""), push)
    if lines or alerts:
        if len(lines) > MAX_BLOCKS:
            dropped = len(lines) - MAX_BLOCKS
            lines = lines[:MAX_BLOCKS] + [f"... and {dropped} more games not shown"]
        body = list(lines) if lines else ["no price changes since the last push"]
        foot = (f"{skipped} game{'s' if skipped != 1 else ''} skipped, "
                f"price floor {BF.PRICE_FLOOR:.2f} / {BF.american(BF.PRICE_FLOOR)}")\
               if skipped else None
        title = "MBM NCAAF" + (" · changes since the last push"
                               if mode == "delta" and lines else "")
        # GREY is the calibration colour. Only when every row on the board is inside the
        # window, so a mixed week reads as a live board with some rows marked.
        colour = notify.GREY if push and cal_n == len(push) else notify.GREEN
        if alerts:
            body = body + ["", "TRIPWIRE: " + ", ".join(alerts)
                           + " below base rate, stop betting this board"]
            colour = notify.RED
        notify.send(title, body, foot, colour)
    elif not push:
        print("no board to send")
    else:
        print("board unchanged since the last push, nothing sent")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
