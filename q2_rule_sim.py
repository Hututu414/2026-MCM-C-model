# NOTE: 仅新增 Q1 后验样本明细落盘与 Q2 优先读取后验样本传播；未改变任何模型/规则算法.
# Self-test: python main_solve.py --task q1 --data_dir ./data_processed --out_dir ./outputs/solve --seed 2026
# Self-test: python main_solve.py --task q2 --data_dir ./data_processed --out_dir ./outputs/solve --seed 2026
from __future__ import annotations



from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, write_csv


PATH_HINT = "# NOTE: Replace with your local absolute path if needed."

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


def build_elimination_map(wide: pd.DataFrame) -> Dict[Tuple[int, int], List[str]]:
    elim_map: Dict[Tuple[int, int], List[str]] = {}
    count_map: Dict[Tuple[int, int], int] = {}
    for _, row in wide.iterrows():
        season = int(row["season"])
        last_week = row.get("last_active_week")
        season_length = row.get("season_length")
        if pd.isna(last_week) or pd.isna(season_length):
            continue
        last_week = int(last_week)
        season_length = int(season_length)
        if last_week >= season_length:
            continue
        key = (season, last_week)
        elim_map.setdefault(key, []).append(row["celebrity_name"])
    for key, names in elim_map.items():
        count_map[key] = len(names)
    return elim_map, count_map


def percent_elim(group: pd.DataFrame, pv: np.ndarray, k: int = 1) -> List[str]:
    c_pct = group["judge_percent"].to_numpy(dtype=float) + pv
    order = np.argsort(c_pct)
    return [group.iloc[i]["celebrity_name"] for i in order[:k]]


def rank_elim(group: pd.DataFrame, pv: np.ndarray, k: int = 1) -> List[str]:
    rv = pd.Series(pv).rank(ascending=False, method="min").to_numpy()
    c_rank = group["judge_rank"].to_numpy(dtype=float) + rv
    order = np.argsort(-c_rank)  # higher is worse
    return [group.iloc[i]["celebrity_name"] for i in order[:k]]


def save_elim(group: pd.DataFrame, pv: np.ndarray) -> List[str]:
    rv = pd.Series(pv).rank(ascending=False, method="min").to_numpy()
    c_rank = group["judge_rank"].to_numpy(dtype=float) + rv
    order = np.argsort(-c_rank)
    bottom2 = order[:2]
    if len(bottom2) < 2:
        return [group.iloc[order[0]]["celebrity_name"]]
    judge_totals = group.iloc[bottom2]["judge_total"].to_numpy(dtype=float)
    elim_idx = bottom2[np.argmin(judge_totals)]
    return [group.iloc[elim_idx]["celebrity_name"]]


def simulate_path(
    season: int,
    season_length: int,
    contestants: List[str],
    sim_data: pd.DataFrame,
    elim_count_map: Dict[Tuple[int, int], int],
    rule: str,
) -> Dict[str, int]:
    active = contestants.copy()
    elimination_order: List[str] = []
    for week in range(1, season_length):
        df_week = sim_data[(sim_data["season"] == season) & (sim_data["week"] == week) & (sim_data["celebrity_name"].isin(active))].copy()
        if df_week.empty:
            continue
        # normalize Jpct and P^V within active set
        jtot = df_week["judge_total"].to_numpy(dtype=float)
        jtot_sum = jtot.sum()
        df_week["judge_percent"] = jtot / jtot_sum if jtot_sum > 0 else np.ones(len(df_week)) / len(df_week)
        # ensure judge_rank exists for rank/save rules (path sim data may not carry it)
        if "judge_rank" not in df_week.columns:
            df_week["judge_rank"] = (
                df_week["judge_total"].rank(ascending=False, method="min").astype(float)
            )
        pv = df_week["pv_hat"].to_numpy(dtype=float)
        pv = pv / pv.sum() if pv.sum() > 0 else np.ones(len(df_week)) / len(df_week)

        k = elim_count_map.get((season, week), 1)
        if rule == "percent":
            elim = percent_elim(df_week, pv, k=k)
        elif rule == "rank":
            elim = rank_elim(df_week, pv, k=k)
        elif rule == "save":
            if k == 1:
                elim = save_elim(df_week, pv)
            else:
                elim = rank_elim(df_week, pv, k=k)
        else:
            elim = percent_elim(df_week, pv, k=k)

        for name in elim:
            if name in active:
                active.remove(name)
                elimination_order.append(name)

    # Assign placements
    placements: Dict[str, int] = {}
    remaining = active.copy()
    if remaining:
        # champion
        if len(remaining) == 1:
            placements[remaining[0]] = 1
        else:
            # arbitrary ordering for remaining (tie)
            for idx, name in enumerate(remaining, start=1):
                placements[name] = idx

    place = len(elimination_order) + 1
    for name in elimination_order[::-1]:
        placements[name] = place
        place += 1

    return placements


def run_q2(
    q1_vote_point: pd.DataFrame,
    q1_vote_ci: pd.DataFrame,
    week_features: pd.DataFrame,
    wide: pd.DataFrame,
    q4_pred: pd.DataFrame,
    out_dir: str,
    seed: int,
    logger,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    elim_map, elim_count_map = build_elimination_map(wide)

    week_features = week_features.copy()
    week_features["season"] = week_features["season"].astype(int)
    week_features["week"] = week_features["week"].astype(int)

    q1_vote_point = q1_vote_point.copy()
    q1_vote_point["season"] = q1_vote_point["season"].astype(int)
    q1_vote_point["week"] = q1_vote_point["week"].astype(int)
    pv_series, pv_source_series = _select_bayes_pv(q1_vote_point, logger, "Q2")
    q1_vote_point["pv_bayes"] = pv_series
    q1_vote_point["pv_source"] = pv_source_series
    pv_source_global = _resolve_pv_source(pv_source_series)
    logger.info("Q2 pv_source=%s", pv_source_global)

    merged = week_features.merge(
        q1_vote_point[["season", "week", "celebrity_name", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    merged["pv_hat"] = merged["pv_bayes"]

    rows = []
    for (season, week), grp in merged.groupby(["season", "week"]):
        grp = grp.copy().reset_index(drop=True)
        pv = grp["pv_hat"].to_numpy(dtype=float)
        pv = pv / pv.sum() if pv.sum() > 0 else np.ones(len(grp)) / len(grp)

        elim_true = elim_map.get((season, int(week)), [])
        no_elim_week = 1 if len(elim_true) == 0 else 0
        if elim_true:
            elim_pct = percent_elim(grp, pv, k=len(elim_true))
            elim_rank = rank_elim(grp, pv, k=len(elim_true))
            elim_save = save_elim(grp, pv) if int(season) >= 28 and len(elim_true) == 1 else []
        else:
            elim_pct, elim_rank, elim_save = [], [], []

        c_pct = grp["judge_percent"].to_numpy(dtype=float) + pv
        order = np.argsort(c_pct)
        margin_pct = float(c_pct[order[1]] - c_pct[order[0]]) if len(c_pct) >= 2 else np.nan

        rv = pd.Series(pv).rank(ascending=False, method="min").to_numpy()
        c_rank = grp["judge_rank"].to_numpy(dtype=float) + rv
        order_rank = np.argsort(-c_rank)
        margin_rank = float(c_rank[order_rank[0]] - c_rank[order_rank[1]]) if len(c_rank) >= 2 else np.nan

        rows.append(
            {
                "season": season,
                "week": int(week),
                "n_remaining": len(grp),
                "eliminated_true": ";".join(elim_true),
                "eliminated_pct": ";".join(elim_pct),
                "eliminated_rank": ";".join(elim_rank),
                "eliminated_save": ";".join(elim_save),
                "match_pct": int(set(elim_pct) == set(elim_true)) if elim_true else np.nan,
                "match_rank": int(set(elim_rank) == set(elim_true)) if elim_true else np.nan,
                "match_save": int(set(elim_save) == set(elim_true)) if (elim_true and elim_save) else np.nan,
                "margin_pct": margin_pct,
                "margin_rank": margin_rank,
                "pv_source": pv_source_global,
                "no_elim_week_flag": no_elim_week,
            }
        )

    weekly_cf = pd.DataFrame(rows)

    # Monte Carlo summary using CI
    mc_rows = []
    if not q1_vote_ci.empty:
        q1_vote_ci["season"] = q1_vote_ci["season"].astype(int)
        q1_vote_ci["week"] = q1_vote_ci["week"].astype(int)
        q1_dir = Path(out_dir).parent / "q1"
        posterior_samples_df = _load_posterior_samples(q1_dir, logger)
        for (season, week), grp in merged.groupby(["season", "week"]):
            ci_grp = q1_vote_ci[(q1_vote_ci["season"] == season) & (q1_vote_ci["week"] == week)]
            ci_grp = ci_grp.set_index("celebrity_name").reindex(grp["celebrity_name"]).fillna(0)
            use_posterior = False
            pv_source = "ci_independent_normal_fallback"
            pv_mat = None
            if posterior_samples_df is not None:
                df_sw = posterior_samples_df[
                    (posterior_samples_df["season"] == season) & (posterior_samples_df["week"] == int(week))
                ]
                if not df_sw.empty:
                    df_sw = df_sw[df_sw["celebrity_name"].isin(grp["celebrity_name"])]
                    if not df_sw.empty:
                        pv_mat = df_sw.pivot_table(
                            index="sample_id",
                            columns="celebrity_name",
                            values="pv_sample",
                            aggfunc="mean",
                        )
                        pv_mat = pv_mat.reindex(columns=grp["celebrity_name"], fill_value=0.0)
                        if pv_mat.shape[0] >= 10:
                            row_sums = pv_mat.sum(axis=1).replace(0, np.nan)
                            pv_mat = pv_mat.div(row_sums, axis=0).fillna(1.0 / len(pv_mat.columns))
                            use_posterior = True
                            pv_source = "abc_posterior_samples"
                        else:
                            logger.warning("Q2 MC fallback: season=%s week=%s posterior samples too few (%s)", season, week, pv_mat.shape[0])
                    else:
                        logger.warning("Q2 MC fallback: season=%s week=%s no matching celebrities in posterior samples", season, week)
                else:
                    logger.warning("Q2 MC fallback: season=%s week=%s posterior samples missing", season, week)

            match_pct = 0
            match_rank = 0
            match_save = 0
            elim_true = elim_map.get((season, int(week)), [])
            no_elim_week = 1 if len(elim_true) == 0 else 0
            if use_posterior and pv_mat is not None:
                m = pv_mat.shape[0]
                for _, row in pv_mat.iterrows():
                    pv = row.to_numpy(dtype=float)
                    if elim_true:
                        elim_pct = percent_elim(grp, pv, k=len(elim_true))
                        elim_rank = rank_elim(grp, pv, k=len(elim_true))
                        elim_save = save_elim(grp, pv) if int(season) >= 28 and len(elim_true) == 1 else []
                        match_pct += int(set(elim_pct) == set(elim_true))
                        match_rank += int(set(elim_rank) == set(elim_true))
                        match_save += int(set(elim_save) == set(elim_true)) if elim_save else 0
            else:
                mean = ci_grp["pv_p50"].to_numpy(dtype=float)
                sd = (ci_grp["pv_p97_5"] - ci_grp["pv_p2_5"]).to_numpy(dtype=float) / (2 * 1.96 + 1e-9)
                m = 200
                rng = np.random.default_rng(seed + int(season) * 100 + int(week))
                for _ in range(m):
                    pv = rng.normal(mean, sd)
                    pv = np.clip(pv, 0, None)
                    pv = pv / pv.sum() if pv.sum() > 0 else np.ones(len(pv)) / len(pv)
                    if elim_true:
                        elim_pct = percent_elim(grp, pv, k=len(elim_true))
                        elim_rank = rank_elim(grp, pv, k=len(elim_true))
                        elim_save = save_elim(grp, pv) if int(season) >= 28 and len(elim_true) == 1 else []
                        match_pct += int(set(elim_pct) == set(elim_true))
                        match_rank += int(set(elim_rank) == set(elim_true))
                        match_save += int(set(elim_save) == set(elim_true)) if elim_save else 0

            mc_rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "samples": m,
                    "match_rate_pct": match_pct / m if (m and len(elim_true) > 0) else np.nan,
                    "match_rate_rank": match_rank / m if (m and len(elim_true) > 0) else np.nan,
                    "match_rate_save": match_save / m if (int(season) >= 28 and m and len(elim_true) > 0) else np.nan,
                    "denom": m if len(elim_true) > 0 else 0,
                    "no_elim_week_count": no_elim_week,
                    "pv_source": pv_source,
                }
            )

    mc_summary = pd.DataFrame(mc_rows)

    # Path simulation (optional, uses q4_pred)
    sim_rows = []
    if q4_pred is not None and not q4_pred.empty:
        # Build sim_data with actual + predicted
        sim_base = week_features[["season", "week", "celebrity_name", "judge_total", "n_judges"]].copy()
        sim_base = sim_base.merge(
            q1_vote_point[["season", "week", "celebrity_name", "pv_bayes", "pv_source"]],
            on=["season", "week", "celebrity_name"],
            how="left",
        )
        sim_base = sim_base.rename(columns={"pv_bayes": "pv_hat_obs"})
        pred = q4_pred.copy()
        pred = pred.rename(columns={"p_vote_hat": "pv_hat_pred"})
        sim_base = sim_base.merge(pred, on=["season", "week", "celebrity_name"], how="left")

        # expected n_judges per season-week
        expected = sim_base.groupby(["season", "week"])["n_judges"].max().reset_index()
        sim_base = sim_base.merge(expected, on=["season", "week"], suffixes=("", "_exp"))
        sim_base["n_judges_exp"] = sim_base["n_judges_exp"].fillna(sim_base["n_judges"])

        sim_base["judge_total"] = sim_base["judge_total"].fillna(sim_base["judge_avg_hat"] * sim_base["n_judges_exp"])
        sim_base["pv_hat"] = sim_base["pv_hat_obs"].fillna(sim_base["pv_hat_pred"])

        for season, sdf in wide.groupby("season"):
            contestants = sdf["celebrity_name"].tolist()
            season_length = int(sdf["season_length"].max())
            for rule in ["percent", "rank", "save"]:
                placements = simulate_path(
                    season,
                    season_length,
                    contestants,
                    sim_base,
                    elim_count_map,
                    rule,
                )
                for name, place in placements.items():
                    sim_rows.append(
                        {
                            "season": season,
                            "rule": rule,
                            "celebrity_name": name,
                            "placement_pred": place,
                            "method": "pred_fill",
                            "pv_source": pv_source_global,
                        }
                    )
    else:
        sim_rows.append({"status": "not_run", "reason": "q4_pred_for_counterfactual missing", "pv_source": pv_source_global})

    # Metrics summary
    n_weeks_total = len(weekly_cf)
    n_weeks_no_elim = int(weekly_cf["no_elim_week_flag"].sum()) if "no_elim_week_flag" in weekly_cf.columns else 0
    n_weeks_with_elim = n_weeks_total - n_weeks_no_elim
    no_elim_rate = (n_weeks_no_elim / n_weeks_total) if n_weeks_total else np.nan
    logger.info("Q2 evaluation: excluded no-elimination weeks from accuracy metrics; no_elim_rate=%.4f", no_elim_rate)

    metrics_rows = [
        {
            "model": "Percent",
            "scheme": "A",
            "data_version": "raw",
            "evaluation": "weekly_match_rate",
            "value": weekly_cf["match_pct"].mean(skipna=True) if not weekly_cf.empty else np.nan,
            "weeks": len(weekly_cf),
            "pv_source": pv_source_global,
            "n_weeks_total": n_weeks_total,
            "n_weeks_with_elim": n_weeks_with_elim,
            "n_weeks_no_elim": n_weeks_no_elim,
            "no_elim_rate": no_elim_rate,
        },
        {
            "model": "Rank",
            "scheme": "A",
            "data_version": "raw",
            "evaluation": "weekly_match_rate",
            "value": weekly_cf["match_rank"].mean(skipna=True) if not weekly_cf.empty else np.nan,
            "weeks": len(weekly_cf),
            "pv_source": pv_source_global,
            "n_weeks_total": n_weeks_total,
            "n_weeks_with_elim": n_weeks_with_elim,
            "n_weeks_no_elim": n_weeks_no_elim,
            "no_elim_rate": no_elim_rate,
        },
        {
            "model": "Rank+Save",
            "scheme": "A",
            "data_version": "raw",
            "evaluation": "weekly_match_rate",
            "value": weekly_cf["match_save"].mean(skipna=True) if not weekly_cf.empty else np.nan,
            "weeks": len(weekly_cf),
            "pv_source": pv_source_global,
            "n_weeks_total": n_weeks_total,
            "n_weeks_with_elim": n_weeks_with_elim,
            "n_weeks_no_elim": n_weeks_no_elim,
            "no_elim_rate": no_elim_rate,
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    out_paths = {
        "q2_weekly_counterfactual": str(Path(out_dir) / "q2_weekly_counterfactual.csv"),
        "q2_mc_summary": str(Path(out_dir) / "q2_mc_summary.csv"),
        "q2_metrics": str(Path(out_dir) / "q2_metrics.csv"),
        "q2_path_simulation": str(Path(out_dir) / "q2_path_simulation.csv"),
    }

    write_csv(weekly_cf, out_paths["q2_weekly_counterfactual"])
    write_csv(mc_summary, out_paths["q2_mc_summary"])
    write_csv(metrics_df, out_paths["q2_metrics"])
    write_csv(pd.DataFrame(sim_rows), out_paths["q2_path_simulation"])

    logger.info("Q2 outputs saved to %s", out_dir)
    return out_paths




