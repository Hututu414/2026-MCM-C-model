# NOTE: Q3 enhanced with sustained/survival/rule-sensitivity controversy metrics and special-case reporting.
# This file only changes Q3 logic; Q1/Q2 remain untouched.
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import difflib
import re

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, write_csv
from ..utils.metrics import kendall_tau


PATH_HINT = "# NOTE: Replace with your local absolute path if needed."


SPECIAL_CASES = [
    {"season": 2, "name": "Jerry Rice"},
    {"season": 4, "name": "Billy Ray Cyrus"},
    {"season": 11, "name": "Bristol Palin"},
    {"season": 27, "name": "Bobby Bones"},
]


TAU_PCT = 0.80
TOPK_PER_SEASON = 10
MATCH_THRESHOLD = 0.85


def _select_bayes_pv(q1_vote_point: pd.DataFrame, logger, context: str) -> Tuple[pd.Series, pd.Series]:
    has_mean = "pv_post_mean" in q1_vote_point.columns
    has_median = "pv_post_median" in q1_vote_point.columns
    if not has_mean and not has_median:
        raise ValueError("Q2/Q3/Q4 require Bayesian posterior PV columns from Q1.")

    pv_source = pd.Series(index=q1_vote_point.index, dtype=object)
    fallback = False
    if has_mean:
        pv = pd.to_numeric(q1_vote_point["pv_post_mean"], errors="coerce")
        pv_source[:] = "bayes_post_mean"
        if pv.isna().any():
            if not has_median:
                raise ValueError("Q2/Q3/Q4 require Bayesian posterior PV columns from Q1.")
            mask = pv.isna()
            pv.loc[mask] = pd.to_numeric(q1_vote_point.loc[mask, "pv_post_median"], errors="coerce")
            pv_source.loc[mask] = "bayes_post_median"
            fallback = mask.any()
    else:
        pv = pd.to_numeric(q1_vote_point["pv_post_median"], errors="coerce")
        pv_source[:] = "bayes_post_median"
        fallback = True

    if pv.isna().any():
        raise ValueError("Q2/Q3/Q4 require Bayesian posterior PV columns from Q1.")
    if fallback:
        logger.warning("%s fallback_to_median=1", context)
    return pv, pv_source


def _resolve_pv_source(pv_source: pd.Series) -> str:
    if (pv_source == "bayes_post_mean").any():
        return "bayes_post_mean"
    if (pv_source == "bayes_post_median").any():
        return "bayes_post_median"
    return "bayes_post_mean"


def _parse_elim_list(val: object) -> set:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return set()
    s = str(val).strip()
    if not s:
        return set()
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return set(parts)


def _normalize_name(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _zscore_within_season(df: pd.DataFrame, col: str) -> pd.Series:
    def _z(x: pd.Series) -> pd.Series:
        x = x.fillna(0.0)
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / std

    return df.groupby("season")[col].transform(_z)


def _max_streak(weeks: pd.Series, flags: pd.Series) -> int:
    if weeks.empty:
        return 0
    order = np.argsort(weeks.to_numpy())
    flags_sorted = flags.to_numpy()[order]
    max_run = 0
    cur = 0
    for f in flags_sorted:
        if f:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return int(max_run)


def _load_posterior_samples(q1_dir: Path, logger) -> pd.DataFrame | None:
    parquet_path = q1_dir / "q1_pv_posterior_samples.parquet"
    csv_gz_path = q1_dir / "q1_pv_posterior_samples.csv.gz"
    csv_path = q1_dir / "q1_pv_posterior_samples.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", parquet_path, e)
    if csv_gz_path.exists():
        try:
            return pd.read_csv(csv_gz_path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_gz_path, e)
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            logger.warning("Failed to read %s: %s", csv_path, e)
    return None


def run_q3(
    week_features: pd.DataFrame,
    q1_vote_point: pd.DataFrame,
    q2_weekly_counterfactual: pd.DataFrame,
    out_dir: str,
    logger,
) -> Dict[str, str]:
    ensure_dir(out_dir)

    # Use Bayesian PV (mean/median) for controversy; pv_hat column kept for compatibility
    q1_vote_point = q1_vote_point.copy()
    pv_series, pv_source_series = _select_bayes_pv(q1_vote_point, logger, "Q3")
    q1_vote_point["pv_bayes"] = pv_series
    q1_vote_point["pv_source"] = pv_source_series
    pv_source_global = _resolve_pv_source(pv_source_series)
    logger.info("Q3 pv_source=%s", pv_source_global)

    df = week_features.merge(
        q1_vote_point[["season", "week", "celebrity_name", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    df["pv_hat"] = df["pv_bayes"]

    # Rank of fan votes (Bayesian PV)
    df["rv_hat"] = df.groupby(["season", "week"])["pv_hat"].rank(ascending=False, method="min")
    df["D_ist"] = (df["judge_rank"] - df["rv_hat"]).abs()

    # Week-level Kendall tau
    d_st = []
    for (season, week), grp in df.groupby(["season", "week"]):
        tau = kendall_tau(grp["judge_rank"], grp["rv_hat"])
        d_st.append({"season": season, "week": int(week), "D_st": 1 - tau})
    d_st_df = pd.DataFrame(d_st)
    df = df.merge(d_st_df, on=["season", "week"], how="left")

    # Normalized controversy within week
    df["n_st"] = df.groupby(["season", "week"])["celebrity_name"].transform("count")
    df["D_norm"] = df.apply(lambda r: (r["D_ist"] / (r["n_st"] - 1)) if r["n_st"] > 1 else np.nan, axis=1)
    df["D_pct"] = df.groupby(["season", "week"])["D_norm"].rank(pct=True, method="max")

    # Flags for special patterns
    df["judge_worst_flag"] = (df["judge_rank"] == df["n_st"]).astype(int)
    df["judge_bottom2_flag"] = np.where(
        df["n_st"] >= 2,
        (df["judge_rank"] >= (df["n_st"] - 1)).astype(int),
        df["judge_worst_flag"],
    )
    top_k = (np.ceil(0.30 * df["n_st"]).clip(lower=1)).astype(int)
    df["fan_top_flag"] = (df["rv_hat"] <= top_k).astype(int)
    df["fan_rescue_flag"] = ((df["judge_bottom2_flag"] == 1) & (df["fan_top_flag"] == 1)).astype(int)

    # last_active_week_obs from observed weeks
    last_active_map = df.groupby(["season", "celebrity_name"])["week"].max().to_dict()
    df["last_active_week_obs"] = df.apply(lambda r: last_active_map.get((r["season"], r["celebrity_name"]), np.nan), axis=1)
    df["survived_flag"] = (df["week"] < df["last_active_week_obs"]).astype(int)
    df["survived_after_worst_flag"] = ((df["judge_worst_flag"] == 1) & (df["survived_flag"] == 1)).astype(int)
    df["survived_after_bottom2_flag"] = ((df["judge_bottom2_flag"] == 1) & (df["survived_flag"] == 1)).astype(int)

    # Rule sensitivity from Q2 counterfactual
    q2 = q2_weekly_counterfactual.copy()
    q2_sets = {}
    for _, row in q2.iterrows():
        key = (int(row["season"]), int(row["week"]))
        pct = _parse_elim_list(row.get("eliminated_pct", ""))
        rank = _parse_elim_list(row.get("eliminated_rank", ""))
        save = _parse_elim_list(row.get("eliminated_save", ""))
        true_elim = _parse_elim_list(row.get("eliminated_true", ""))
        rule_sets = [pct, rank]
        if row.get("eliminated_save", "") != "" or int(row["season"]) >= 28:
            rule_sets.append(save)
        disagree = int(len({frozenset(s) for s in rule_sets}) > 1)
        q2_sets[key] = {
            "pct": pct,
            "rank": rank,
            "save": save,
            "true": true_elim,
            "weekly_rule_disagree_flag": disagree,
        }

    def _flag_from_set(r, key_name: str) -> int:
        key = (int(r["season"]), int(r["week"]))
        sets = q2_sets.get(key, {})
        return int(r["celebrity_name"] in sets.get(key_name, set()))

    df["weekly_rule_disagree_flag"] = df.apply(
        lambda r: q2_sets.get((int(r["season"]), int(r["week"])), {}).get("weekly_rule_disagree_flag", 0), axis=1
    )
    df["elim_true_flag"] = df.apply(lambda r: _flag_from_set(r, "true"), axis=1)
    df["elim_pct_flag"] = df.apply(lambda r: _flag_from_set(r, "pct"), axis=1)
    df["elim_rank_flag"] = df.apply(lambda r: _flag_from_set(r, "rank"), axis=1)
    df["elim_save_flag"] = df.apply(lambda r: _flag_from_set(r, "save"), axis=1)

    def _pivotal(r) -> int:
        vals = [r["elim_pct_flag"], r["elim_rank_flag"], r["elim_save_flag"]]
        return int(len(set(vals)) > 1)

    df["contestant_week_rule_pivotal_flag"] = df.apply(_pivotal, axis=1)

    # Optional p_controversy from posterior samples
    df["p_controversy"] = np.nan
    q1_dir = Path(out_dir).parent / "q1"
    posterior_samples_df = _load_posterior_samples(q1_dir, logger)
    if posterior_samples_df is not None:
        for (season, week), grp in df.groupby(["season", "week"]):
            df_sw = posterior_samples_df[(posterior_samples_df["season"] == season) & (posterior_samples_df["week"] == int(week))]
            if df_sw.empty:
                continue
            df_sw = df_sw[df_sw["celebrity_name"].isin(grp["celebrity_name"])]
            if df_sw.empty:
                continue
            pv_mat = df_sw.pivot_table(index="sample_id", columns="celebrity_name", values="pv_sample", aggfunc="mean")
            pv_mat = pv_mat.reindex(columns=grp["celebrity_name"], fill_value=0.0)
            if pv_mat.shape[0] == 0:
                continue
            # sample-based probability of being in top 20% controversy
            counts = dict.fromkeys(grp["celebrity_name"], 0)
            M = pv_mat.shape[0]
            n_st = len(grp)
            if n_st <= 1:
                continue
            judge_rank_vec = grp.set_index("celebrity_name")["judge_rank"].to_dict()
            for _, row in pv_mat.iterrows():
                pv = row.to_numpy(dtype=float)
                rv = pd.Series(pv, index=grp["celebrity_name"]).rank(ascending=False, method="min")
                d_ist = (rv - pd.Series(judge_rank_vec)).abs()
                d_norm = d_ist / (n_st - 1)
                d_pct = d_norm.rank(pct=True, method="max")
                for name in grp["celebrity_name"]:
                    if d_pct.loc[name] >= TAU_PCT:
                        counts[name] += 1
            for name, c in counts.items():
                df.loc[(df["season"] == season) & (df["week"] == int(week)) & (df["celebrity_name"] == name), "p_controversy"] = c / M

    # Popularity vs performance decomposition (approx)
    df["pv_hat_clip"] = df["pv_hat"].clip(1e-6, 1 - 1e-6)
    df["logit_pv"] = np.log(df["pv_hat_clip"] / (1 - df["pv_hat_clip"]))

    beta = 0.0
    if df["judge_percent"].notna().any():
        x = df["judge_percent"].fillna(0).to_numpy()
        y = df["logit_pv"].fillna(0).to_numpy()
        denom = (x ** 2).sum()
        beta = float((x * y).sum() / denom) if denom > 0 else 0.0

    pop_map = df.groupby(["season", "celebrity_name"])["logit_pv"].mean().to_dict()
    df["pop_term"] = df.apply(lambda r: pop_map.get((r["season"], r["celebrity_name"]), 0.0), axis=1)
    df["perf_term"] = beta * df["judge_percent"].fillna(0)
    df["Expl_ist"] = (df["pop_term"].abs() / (df["pop_term"].abs() + df["perf_term"].abs() + 1e-9))

    # bottom2 judge elimination hypothetical count (within bottom2, lower judge_total)
    df["bottom2_judge_elim_hypo_flag"] = 0
    for (season, week), grp in df.groupby(["season", "week"]):
        if len(grp) < 2:
            continue
        bottom2 = grp.sort_values("judge_rank", ascending=False).head(2)
        if bottom2.empty:
            continue
        min_total = bottom2["judge_total"].min()
        mask = (df["season"] == season) & (df["week"] == int(week)) & (df["celebrity_name"].isin(bottom2["celebrity_name"])) & (df["judge_total"] == min_total)
        df.loc[mask, "bottom2_judge_elim_hypo_flag"] = 1

    # season-level aggregation
    season_len_obs = df.groupby("season")["week"].max().to_dict()

    agg_rows = []
    for (season, name), grp in df.groupby(["season", "celebrity_name"]):
        grp = grp.sort_values("week")
        controversy_count = int((grp["D_pct"] >= TAU_PCT).sum())
        controversy_streak_max = _max_streak(grp["week"], grp["D_pct"] >= TAU_PCT)
        controversy_area = float(grp["D_norm"].fillna(0).sum())
        judge_worst_count = int(grp["judge_worst_flag"].sum())
        judge_bottom2_count = int(grp["judge_bottom2_flag"].sum())
        fan_rescue_count = int(grp["fan_rescue_flag"].sum())
        survived_after_worst_count = int(grp["survived_after_worst_flag"].sum())
        survived_after_bottom2_count = int(grp["survived_after_bottom2_flag"].sum())
        pivotal_week_count = int(grp["contestant_week_rule_pivotal_flag"].sum())
        last_active_week_obs = int(grp["last_active_week_obs"].max()) if grp["last_active_week_obs"].notna().any() else np.nan
        season_len = int(season_len_obs.get(season, grp["week"].max()))
        went_far_flag = int(last_active_week_obs >= season_len - 1) if not np.isnan(last_active_week_obs) else 0
        mean_p_controversy = grp["p_controversy"].mean() if "p_controversy" in grp.columns else np.nan

        agg_rows.append(
            {
                "season": season,
                "celebrity_name": name,
                "controversy_count": controversy_count,
                "controversy_streak_max": controversy_streak_max,
                "controversy_area": controversy_area,
                "judge_worst_count": judge_worst_count,
                "judge_bottom2_count": judge_bottom2_count,
                "fan_rescue_count": fan_rescue_count,
                "survived_after_worst_count": survived_after_worst_count,
                "survived_after_bottom2_count": survived_after_bottom2_count,
                "pivotal_week_count": pivotal_week_count,
                "season_len_obs": season_len,
                "last_active_week_obs": last_active_week_obs,
                "went_far_flag": went_far_flag,
                "mean_p_controversy": mean_p_controversy,
            }
        )

    season_profile = pd.DataFrame(agg_rows)

    # Z-scores within season
    for col in [
        "survived_after_worst_count",
        "fan_rescue_count",
        "controversy_area",
        "controversy_streak_max",
        "pivotal_week_count",
        "judge_worst_count",
    ]:
        season_profile[f"z_{col}"] = _zscore_within_season(season_profile, col)

    season_profile["controversy_score"] = (
        0.28 * season_profile["z_survived_after_worst_count"]
        + 0.22 * season_profile["z_fan_rescue_count"]
        + 0.18 * season_profile["z_controversy_area"]
        + 0.12 * season_profile["z_controversy_streak_max"]
        + 0.10 * season_profile["z_pivotal_week_count"]
        + 0.10 * season_profile["z_judge_worst_count"]
    )

    season_profile["rank_in_season"] = season_profile.groupby("season")["controversy_score"].rank(ascending=False, method="min").astype(int)

    def _reason_tags(r) -> str:
        tags = []
        if r.get("fan_rescue_count", 0) >= 2:
            tags.append("FAN_RESCUE_REPEATED")
        if r.get("judge_worst_count", 0) >= 3 and r.get("went_far_flag", 0) == 1:
            tags.append("JUDGE_WORST_BUT_WENT_FAR")
        if r.get("survived_after_worst_count", 0) >= 2:
            tags.append("SURVIVED_AFTER_WORST")
        if r.get("pivotal_week_count", 0) >= 2:
            tags.append("RULE_SENSITIVE")
        if r.get("controversy_streak_max", 0) >= 3:
            tags.append("SUSTAINED_CONTROVERSY")
        return "|".join(tags)

    season_profile["reason_tags"] = season_profile.apply(_reason_tags, axis=1)
    season_profile["pv_source"] = pv_source_global

    # Sanity check special cases: if matched and rank >3 log warning
    for case in SPECIAL_CASES:
        season = case["season"]
        target_name = case["name"]
        if season in season_profile["season"].unique():
            norm_target = _normalize_name(target_name)
            season_names = season_profile[season_profile["season"] == season]["celebrity_name"].tolist()
            season_norm = {_normalize_name(n): n for n in season_names}
            matched_name = season_norm.get(norm_target)
            if matched_name:
                rank = season_profile[(season_profile["season"] == season) & (season_profile["celebrity_name"] == matched_name)]["rank_in_season"].iloc[0]
                if rank > 3:
                    row = season_profile[(season_profile["season"] == season) & (season_profile["celebrity_name"] == matched_name)].iloc[0]
                    logger.warning(
                        "Special case %s (S%s) rank_in_season=%s (>3). z-components: survived=%s, fan=%s, area=%s, streak=%s, pivotal=%s, judge_worst=%s",
                        matched_name,
                        season,
                        rank,
                        row["z_survived_after_worst_count"],
                        row["z_fan_rescue_count"],
                        row["z_controversy_area"],
                        row["z_controversy_streak_max"],
                        row["z_pivotal_week_count"],
                        row["z_judge_worst_count"],
                    )

    # q3_controversy_weekly output (with required legacy columns)
    df["pv_source"] = df.get("pv_source", pv_source_global)
    if isinstance(df["pv_source"], pd.Series):
        df["pv_source"] = df["pv_source"].fillna(pv_source_global)

    weekly_cols = [
        "season",
        "week",
        "celebrity_name",
        "judge_rank",
        "rv_hat",
        "pv_hat",
        "D_ist",
        "D_st",
        "Expl_ist",
        "pv_source",
        "n_st",
        "D_norm",
        "D_pct",
        "judge_worst_flag",
        "judge_bottom2_flag",
        "fan_top_flag",
        "fan_rescue_flag",
        "last_active_week_obs",
        "survived_flag",
        "survived_after_worst_flag",
        "survived_after_bottom2_flag",
        "weekly_rule_disagree_flag",
        "elim_true_flag",
        "elim_pct_flag",
        "elim_rank_flag",
        "elim_save_flag",
        "contestant_week_rule_pivotal_flag",
        "bottom2_judge_elim_hypo_flag",
        "p_controversy",
    ]

    # TopK per season
    topk = season_profile.sort_values(["season", "controversy_score"], ascending=[True, False]).groupby("season").head(TOPK_PER_SEASON)

    # Counterfactual cases: use reason_tags non-empty, fallback to top3 per season
    cases = season_profile[season_profile["reason_tags"] != ""].copy()
    if cases.empty:
        cases = season_profile.sort_values(["season", "controversy_score"], ascending=[True, False]).groupby("season").head(3)

    # Build week lists for eliminations
    def _weeks_str(grp: pd.DataFrame, col: str) -> str:
        weeks = grp.loc[grp[col] == 1, "week"].astype(int).tolist()
        return ";".join(str(w) for w in sorted(weeks)) if weeks else ""

    cf_rows = []
    for _, row in cases.iterrows():
        season = row["season"]
        name = row["celebrity_name"]
        g = df[(df["season"] == season) & (df["celebrity_name"] == name)]
        cf_rows.append(
            {
                "season": season,
                "celebrity_name": name,
                "controversy_score": row["controversy_score"],
                "rank_in_season": row["rank_in_season"],
                "reason_tags": row["reason_tags"],
                "judge_worst_count": row["judge_worst_count"],
                "survived_after_worst_count": row["survived_after_worst_count"],
                "fan_rescue_count": row["fan_rescue_count"],
                "pivotal_week_count": row["pivotal_week_count"],
                "went_far_flag": row["went_far_flag"],
                "weeks_where_pct_would_elim": _weeks_str(g, "elim_pct_flag"),
                "weeks_where_rank_would_elim": _weeks_str(g, "elim_rank_flag"),
                "weeks_where_save_would_elim": _weeks_str(g, "elim_save_flag"),
                "bottom2_judge_elim_hypo_count": int(g["bottom2_judge_elim_hypo_flag"].sum()),
                "pv_source": pv_source_global,
            }
        )

    # Special case matching
    match_rows = []
    report_rows = []

    season_names_map = {
        s: season_profile[season_profile["season"] == s]["celebrity_name"].tolist()
        for s in season_profile["season"].unique()
    }

    for case in SPECIAL_CASES:
        season = case["season"]
        target = case["name"]
        norm_target = _normalize_name(target)
        if season not in season_names_map:
            match_rows.append(
                {
                    "target_season": season,
                    "target_name": target,
                    "matched_name": "",
                    "match_score": 0.0,
                    "matched": 0,
                    "note": "season_not_found",
                }
            )
            report_rows.append(
                {
                    "season": season,
                    "celebrity_name": target,
                    "controversy_score": np.nan,
                    "rank_in_season": np.nan,
                    "reason_tags": "",
                    "judge_worst_count": np.nan,
                    "survived_after_worst_count": np.nan,
                    "fan_rescue_count": np.nan,
                    "controversy_area": np.nan,
                    "controversy_streak_max": np.nan,
                    "pivotal_week_count": np.nan,
                    "last_active_week_obs": np.nan,
                    "season_len_obs": np.nan,
                    "note": "not_found",
                }
            )
            continue

        candidates = season_names_map[season]
        norm_candidates = [_normalize_name(n) for n in candidates]

        # exact / contains match
        matched_name = ""
        match_score = 0.0
        note = ""
        if norm_target in norm_candidates:
            idx = norm_candidates.index(norm_target)
            matched_name = candidates[idx]
            match_score = 1.0
            note = "exact"
        else:
            # contains match
            contains_idx = None
            for i, nc in enumerate(norm_candidates):
                if norm_target in nc or nc in norm_target:
                    contains_idx = i
                    break
            if contains_idx is not None:
                matched_name = candidates[contains_idx]
                match_score = 0.9
                note = "contains"
            else:
                # similarity
                scores = [difflib.SequenceMatcher(None, norm_target, nc).ratio() for nc in norm_candidates]
                top_idx = np.argsort(scores)[::-1][:5]
                best_idx = int(top_idx[0]) if len(top_idx) > 0 else None
                if best_idx is not None:
                    match_score = float(scores[best_idx])
                    matched_name = candidates[best_idx] if match_score >= MATCH_THRESHOLD else ""
                    top_list = [(candidates[i], round(scores[i], 3)) for i in top_idx]
                    note = f"top_candidates={top_list}"

        matched = int(match_score >= MATCH_THRESHOLD and matched_name != "")
        if matched == 0:
            matched_name = ""

        match_rows.append(
            {
                "target_season": season,
                "target_name": target,
                "matched_name": matched_name,
                "match_score": match_score,
                "matched": matched,
                "note": note,
            }
        )

        if matched == 1:
            row = season_profile[(season_profile["season"] == season) & (season_profile["celebrity_name"] == matched_name)].iloc[0]
            report_rows.append(
                {
                    "season": season,
                    "celebrity_name": matched_name,
                    "controversy_score": row["controversy_score"],
                    "rank_in_season": row["rank_in_season"],
                    "reason_tags": row["reason_tags"],
                    "judge_worst_count": row["judge_worst_count"],
                    "survived_after_worst_count": row["survived_after_worst_count"],
                    "fan_rescue_count": row["fan_rescue_count"],
                    "controversy_area": row["controversy_area"],
                    "controversy_streak_max": row["controversy_streak_max"],
                    "pivotal_week_count": row["pivotal_week_count"],
                    "last_active_week_obs": row["last_active_week_obs"],
                    "season_len_obs": row["season_len_obs"],
                    "note": "matched",
                }
            )
        else:
            report_rows.append(
                {
                    "season": season,
                    "celebrity_name": target,
                    "controversy_score": np.nan,
                    "rank_in_season": np.nan,
                    "reason_tags": "",
                    "judge_worst_count": np.nan,
                    "survived_after_worst_count": np.nan,
                    "fan_rescue_count": np.nan,
                    "controversy_area": np.nan,
                    "controversy_streak_max": np.nan,
                    "pivotal_week_count": np.nan,
                    "last_active_week_obs": np.nan,
                    "season_len_obs": np.nan,
                    "note": "not_found",
                }
            )

    out_paths = {
        "q3_controversy_weekly": str(Path(out_dir) / "q3_controversy_weekly.csv"),
        "q3_controversy_topk": str(Path(out_dir) / "q3_controversy_topk.csv"),
        "q3_counterfactual_cases": str(Path(out_dir) / "q3_counterfactual_cases.csv"),
        "q3_special_cases_match": str(Path(out_dir) / "q3_special_cases_match.csv"),
        "q3_special_cases_report": str(Path(out_dir) / "q3_special_cases_report.csv"),
    }

    write_csv(df[weekly_cols], out_paths["q3_controversy_weekly"])
    write_csv(
        topk[[
            "season",
            "celebrity_name",
            "controversy_score",
            "rank_in_season",
            "reason_tags",
            "season_len_obs",
            "last_active_week_obs",
            "went_far_flag",
            "fan_rescue_count",
            "survived_after_worst_count",
            "judge_worst_count",
            "judge_bottom2_count",
            "controversy_area",
            "controversy_streak_max",
            "pivotal_week_count",
            "mean_p_controversy",
            "pv_source",
        ]],
        out_paths["q3_controversy_topk"],
    )
    write_csv(pd.DataFrame(cf_rows), out_paths["q3_counterfactual_cases"])
    write_csv(pd.DataFrame(match_rows), out_paths["q3_special_cases_match"])
    write_csv(pd.DataFrame(report_rows), out_paths["q3_special_cases_report"])

    logger.info("Q3 outputs saved to %s", out_dir)
    return out_paths
