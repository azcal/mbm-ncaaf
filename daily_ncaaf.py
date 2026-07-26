"""
MBM NCAAF daily runner (rebuild v1, frozen spec).
- Historical base: cfbfastR schedule parquets (2016-2025, committed)
- 2026 season: ESPN scoreboard feed, team names mapped to CFBD via committed alias
- Boards FBS-vs-FBS games (8-day horizon): p(home), model margin, model total
- CALIBRATION GATE: value lines suppressed before Week 4
- Grades logged predictions, rolling metrics + last-200 tripwire, Telegram alerts
Usage: python daily_ncaaf.py [--date YYYY-MM-DD]
"""
import argparse, json, math, os, sys, time, datetime as dt
import numpy as np, pandas as pd, requests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_ncaaf as E

H = {"User-Agent": "Mozilla/5.0"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
LOGS = os.path.join(HERE, "logs")
SPEC = json.load(open(os.path.join(HERE, "spec.json")))
CFG, C = SPEC["cfg"], SPEC["C"]
ALIAS = json.load(open(os.path.join(HERE, "data", "espn_alias.json")))
CONF = json.load(open(os.path.join(HERE, "data", "conf_latest.json")))
TRAIN_LO, TRAIN_HI = 2018, 2025
CUR_SEASON = 2026
CAL_UNLOCK_WEEK = 4
MIN_EDGE = 0.05

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
    g["date"] = pd.to_datetime(g.start_date).dt.tz_localize(None)
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
                    completed=done))
            time.sleep(0.15)
    d = pd.DataFrame(rows).drop_duplicates("game_id")
    if len(d):
        d["date"] = pd.to_datetime(d.start_date).dt.tz_localize(None)
    return d

def run(today_str):
    today = pd.Timestamp(today_str)
    hist = load_hist()
    cur = fetch_2026()
    done26 = cur[cur.completed] if len(cur) else cur
    up = cur[(~cur.completed) & (cur.date >= today) & (cur.date <= today + pd.Timedelta(days=8))
             & (cur.home_division == "fbs") & (cur.away_division == "fbs")] if len(cur) else pd.DataFrame()
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
        pd.DataFrame(columns=["date", "game_id", "week", "season", "home", "away", "p_home", "p_elo", "model"])

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
        log = pd.concat([log[~log.game_id.isin(board.game_id)], board], ignore_index=True)
        board = board.assign(margin_m=margins, total_m=totals, week_v=rows.week.values,
                             div=(board.p_home - board.p_elo).abs())
        disp = board.sort_values("div", ascending=False).groupby("date").head(20) \
                    .sort_values(["date", "div"], ascending=[True, False])
        print(f"\n=== MBM NCAAF board: {len(disp)} of {len(board)} FBS games "
              f"(top 20/day by model-vs-Elo divergence; full slate logged & graded) ===")
        for _, r in disp.iterrows():
            side = f"{r.home} {-r.margin_m:+.1f}" if r.margin_m > 0 else f"{r.away} {r.margin_m:+.1f}"
            vals = (f"value: {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}"
                    if r.week_v >= CAL_UNLOCK_WEEK else "CALIBRATION thru Wk3: predictions only, no betting")
            print(f"{r.date} {r.away} @ {r.home}  p(home)={r.p_home:.3f}  model: {side}, total {r.total_m:.1f}  | {vals}")
    else:
        print("no FBS games in the next 8 days")

    res_map = g[g.margin.notna()].assign(gid=g.loc[g.margin.notna(), "game_id"].astype(str)) \
        .set_index("gid")["margin"]
    log["result"] = log.game_id.map(res_map)
    log.to_csv(pred_path, index=False)
    dec = log[log.result.notna() & (log.result != 0)]
    if len(dec) >= 30:
        y = (dec.result > 0).astype(int).values
        print(f"\nrolling grade (n={len(y)}):")
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
        dec.to_csv(os.path.join(LOGS, "graded.csv"), index=False)

    tg, ch = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT")
    if tg and ch and (board is not None or alerts):
        if board is not None:
            lines = []
            for _, r in disp.iterrows():
                if r.week < CAL_UNLOCK_WEEK:
                    lines.append(f"{r.away} @ {r.home}: {r.p_home:.3f} | CALIBRATION, no betting thru Wk3")
                else:
                    lines.append(f"{r.away} @ {r.home}: {r.p_home:.3f} | {r.home} >= {min_odds(r.p_home):.2f} / {r.away} >= {min_odds(1-r.p_home):.2f}")
            txt = "\n".join(lines[:30])
        else:
            txt = "no FBS games this week"
        if alerts:
            txt += "\n\u26a0 TRIPWIRE: " + ", ".join(alerts) + " below base rate, stop betting this board"
        requests.post(f"https://api.telegram.org/bot{tg}/sendMessage",
                      data={"chat_id": ch, "text": f"MBM NCAAF\n{txt}"}, timeout=20)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    run(ap.parse_args().date)
