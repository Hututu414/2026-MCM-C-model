from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    __package__ = "src.model"

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, write_csv, write_json
from ..utils.logging import get_logger
from ..utils.metrics import kendall_tau


PATH_HINT = "# NOTE: Replace with your local absolute path if needed."


def _load_q1_posterior_samples(out_dir: str, logger) -> Dict[Tuple[int, int], pd.DataFrame] | None:
    q1_dir = Path(out_dir).parent / "q1"
    parquet_path = q1_dir / "q1_pv_posterior_samples.parquet"
    csv_path = q1_dir / "q1_pv_posterior_samples.csv.gz"
    df = None
    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - fallback safety
        logger.warning("Q5 posterior samples load failed: %s", exc)
        df = None

    if df is None or df.empty:
        return None

    required = {"season", "week", "sample_id", "celebrity_name", "pv_sample"}
    if not required.issubset(df.columns):
        logger.warning("Q5 posterior samples missing columns: %s", required - set(df.columns))
        return None

    if "is_accepted" in df.columns:
        df = df[df["is_accepted"] == 1].copy()
    if df.empty:
        return None

    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df["sample_id"] = df["sample_id"].astype(int)

    grouped: Dict[Tuple[int, int], pd.DataFrame] = {}
    for (s, w), g in df.groupby(["season", "week"]):
        grouped[(int(s), int(w))] = g[["sample_id", "celebrity_name", "pv_sample"]].copy()
    return grouped


def _build_ci_cache(q1_vote_ci: pd.DataFrame) -> Dict[Tuple[int, int, str], Tuple[float, float]]:
    ci_cache: Dict[Tuple[int, int, str], Tuple[float, float]] = {}
    if q1_vote_ci is None or q1_vote_ci.empty:
        return ci_cache
    if not {"season", "week", "celebrity_name", "pv_p50", "pv_p2_5", "pv_p97_5"}.issubset(q1_vote_ci.columns):
        return ci_cache
    for _, row in q1_vote_ci.iterrows():
        season = int(row["season"])
        week = int(row["week"])
        name = str(row["celebrity_name"])
        mean = float(row["pv_p50"])
        sd = float((row["pv_p97_5"] - row["pv_p2_5"]) / (2 * 1.96 + 1e-9))
        ci_cache[(season, week, name)] = (mean, sd)
    return ci_cache


def _draw_pv(
    season: int,
    week: int,
    active_names: List[str],
    rng: np.random.Generator,
    posterior: Dict[Tuple[int, int], pd.DataFrame] | None,
    ci_cache: Dict[Tuple[int, int, str], Tuple[float, float]],
) -> Tuple[np.ndarray, str]:
    if posterior is not None and (season, week) in posterior:
        df = posterior[(season, week)]
        if not df.empty:
            sample_ids = df["sample_id"].unique()
            pick_id = int(rng.choice(sample_ids))
            sub = df[df["sample_id"] == pick_id]
            pv_map = dict(zip(sub["celebrity_name"].astype(str), sub["pv_sample"].astype(float)))
            pv = np.array([pv_map.get(name, 0.0) for name in active_names], dtype=float)
            pv = np.clip(pv, 0, None)
            pv = pv / pv.sum() if pv.sum() > 0 else np.ones(len(pv)) / len(pv)
            return pv, "posterior_samples"

    # CI fallback
    means = []
    sds = []
    for name in active_names:
        key = (season, week, name)
        if key in ci_cache:
            mean, sd = ci_cache[key]
        else:
            mean, sd = 1.0 / max(len(active_names), 1), 0.02
        means.append(mean)
        sds.append(sd)
    pv = rng.normal(np.array(means), np.array(sds))
    pv = np.clip(pv, 0, None)
    pv = pv / pv.sum() if pv.sum() > 0 else np.ones(len(pv)) / len(pv)
    return pv, "ci_fallback"


def build_elimination_count(wide: pd.DataFrame) -> Dict[Tuple[int, int], int]:
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
        count_map[key] = count_map.get(key, 0) + 1
    return count_map


def _build_sim_data(
    week_features: pd.DataFrame,
    q1_vote_point: pd.DataFrame,
    q4_pred: pd.DataFrame,
) -> pd.DataFrame:
    sim = week_features[["season", "week", "celebrity_name", "judge_total", "n_judges"]].copy()
    sim = sim.merge(
        q1_vote_point[["season", "week", "celebrity_name", "pv_hat"]],
        on=["season", "week", "celebrity_name"],
        how="left",
    )
    sim = sim.rename(columns={"pv_hat": "pv_hat_obs"})
    if q4_pred is not None and not q4_pred.empty:
        pred = q4_pred.rename(columns={"p_vote_hat": "pv_hat_pred"})
        sim = sim.merge(pred, on=["season", "week", "celebrity_name"], how="left")
    else:
        sim["judge_avg_hat"] = np.nan
        sim["pv_hat_pred"] = np.nan

    expected = sim.groupby(["season", "week"])["n_judges"].max().reset_index()
    sim = sim.merge(expected, on=["season", "week"], suffixes=("", "_exp"))
    sim["n_judges_exp"] = sim["n_judges_exp"].fillna(sim["n_judges"])

    sim["judge_total"] = sim["judge_total"].fillna(sim["judge_avg_hat"] * sim["n_judges_exp"])
    sim["pv_hat"] = sim["pv_hat_obs"].fillna(sim["pv_hat_pred"])
    sim["pv_hat"] = sim["pv_hat"].fillna(0)
    # 改进2：按周兜底 judge_total，避免 NaN/<=0 影响 judge_percent 与 save 逻辑
    def _fix_judge_total(g: pd.DataFrame) -> pd.DataFrame:
        jt = pd.to_numeric(g["judge_total"], errors="coerce")
        jt_sum = jt.sum(skipna=True)
        if jt.isna().all() or not np.isfinite(jt_sum) or jt_sum <= 0:
            g["judge_total"] = 1.0
            return g
        fill_val = jt.median()
        if not np.isfinite(fill_val):
            fill_val = jt.mean()
        if not np.isfinite(fill_val):
            fill_val = 1.0
        g["judge_total"] = jt.fillna(fill_val)
        return g

    sim = sim.groupby(["season", "week"], group_keys=False).apply(_fix_judge_total)
    return sim


def _simulate_rule(
    season: int,
    season_length: int,
    contestants: List[str],
    sim_data: pd.DataFrame,
    elim_count_map: Dict[Tuple[int, int], int],
    wj: float,
    save: int,
    bottom_k: int,
    rng: np.random.Generator | None = None,
    pv_noise: np.ndarray | None = None,
    pv_drawer=None,
    logger=None,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    active = contestants.copy()
    elimination_order: List[str] = []
    upset_count = 0
    violation_count = 0
    violation_support = 0
    violation_inversion = 0
    violation_inversion_pairs = 0
    n_elim_events = 0
    n_elim_weeks = 0
    n_total = len(contestants)
    cumulative_cnew = {name: 0.0 for name in contestants}
    last_week_cnew: Dict[str, float] = {}

    for week in range(1, season_length):
        df_week = sim_data[(sim_data["season"] == season) & (sim_data["week"] == week) & (sim_data["celebrity_name"].isin(active))].copy()
        if df_week.empty:
            continue
        jtot = df_week["judge_total"].to_numpy(dtype=float)
        jtot_sum = jtot.sum()
        df_week["judge_percent"] = jtot / jtot_sum if jtot_sum > 0 else np.ones(len(df_week)) / len(df_week)

        if pv_drawer is not None and rng is not None:
            pv_used = pv_drawer(season, week, df_week["celebrity_name"].tolist(), rng)
        else:
            pv_used = df_week["pv_hat"].to_numpy(dtype=float)
            if pv_noise is not None:
                pv_used = pv_used + pv_noise[: len(pv_used)]
            pv_used = np.clip(pv_used, 0, None)
            pv_used = pv_used / pv_used.sum() if pv_used.sum() > 0 else np.ones(len(pv_used)) / len(pv_used)

        df_week["pv_used"] = pv_used
        c_new = wj * df_week["judge_percent"].to_numpy(dtype=float) + (1 - wj) * pv_used
        df_week["c_new"] = c_new
        for name, val in zip(df_week["celebrity_name"].tolist(), c_new):
            cumulative_cnew[name] = cumulative_cnew.get(name, 0.0) + float(val)
            last_week_cnew[name] = float(val)
        order = np.argsort(c_new)

        k = elim_count_map.get((season, week), 1)
        if k > 0:
            n_elim_events += int(k)
            n_elim_weeks += 1
        if save == 1 and k == 1 and len(df_week) >= bottom_k:
            bottom = order[:bottom_k]
            bottom_df = df_week.iloc[bottom]
            # 改进2：save 分支判定前补齐 judge_total
            jt = pd.to_numeric(bottom_df["judge_total"], errors="coerce")
            med = jt.median()
            if not np.isfinite(med):
                med = jt.mean()
            if not np.isfinite(med):
                med = 1.0
            jt_filled = jt.fillna(med)
            # judge save: eliminate lowest judge_total in bottom
            elim_idx = bottom[np.argmin(jt_filled.to_numpy(dtype=float))]
            elim = [df_week.iloc[elim_idx]["celebrity_name"]]
        else:
            elim = [df_week.iloc[i]["celebrity_name"] for i in order[:k]]

        # excitement: count if eliminated is not in bottom_k of judge_percent
        bottom_k_eff = min(bottom_k, len(df_week))
        jpct_order = np.argsort(df_week["judge_percent"].to_numpy(dtype=float))
        jpct_bottom = set(df_week.iloc[jpct_order[:bottom_k_eff]]["celebrity_name"].tolist())
        for name in elim:
            if name not in jpct_bottom:
                upset_count += 1

        # violations: (1) support-lack (not in bottom_k of judge nor fan)
        #            (2) composite-rank inversion (survivor has lower c_new)
        pv_order = np.argsort(df_week["pv_used"].to_numpy(dtype=float))
        pv_bottom = set(df_week.iloc[pv_order[:bottom_k_eff]]["celebrity_name"].tolist())
        survivors = df_week[~df_week["celebrity_name"].isin(elim)]
        for name in elim:
            elim_row = df_week[df_week["celebrity_name"] == name].iloc[0]
            flag_support = (name not in jpct_bottom) and (name not in pv_bottom)
            # 改进3：inversion 计算为逆序对数量/比例
            inversion_pairs = 0
            if not survivors.empty:
                inversion_pairs = int((survivors["c_new"] < (elim_row["c_new"] - 1e-12)).sum())
            flag_inversion = inversion_pairs > 0
            violation_support += int(flag_support)
            violation_inversion += int(flag_inversion)
            violation_inversion_pairs += inversion_pairs
            # conservative counting: one violation per eliminated contestant if any rule is violated
            if flag_support or flag_inversion:
                violation_count += 1

        for name in elim:
            if name in active:
                active.remove(name)
                elimination_order.append(name)

    placements: Dict[str, int] = {}
    remaining = active.copy()
    try:
        if len(remaining) == 1:
            placements[remaining[0]] = 1
            for k, name in enumerate(reversed(elimination_order), start=2):
                placements[name] = k
        else:
            if remaining:
                if all(name in last_week_cnew for name in remaining):
                    scores = {name: last_week_cnew.get(name, 0.0) for name in remaining}
                else:
                    scores = {name: cumulative_cnew.get(name, 0.0) for name in remaining}
                remaining_sorted = sorted(remaining, key=lambda n: scores.get(n, 0.0), reverse=True)
                for rank, name in enumerate(remaining_sorted, start=1):
                    placements[name] = rank
                start_rank = len(remaining_sorted) + 1
            else:
                start_rank = 1
            for k, name in enumerate(reversed(elimination_order), start=start_rank):
                if name not in placements:
                    placements[name] = k

        assert len(placements) == n_total
        assert set(placements.values()) == set(range(1, n_total + 1))
    except Exception as exc:  # fallback to avoid crash
        if logger is not None:
            logger.warning("Q5 placement mapping fallback (season=%s): %s", season, exc)
        ordered = sorted(contestants, key=lambda n: cumulative_cnew.get(n, 0.0), reverse=True)
        placements = {name: idx + 1 for idx, name in enumerate(ordered)}

    # 改进1：计数 + 率（按淘汰事件数归一化）
    eps = 1e-12
    upset_rate = upset_count / (n_elim_events + eps)
    violation_rate = violation_count / (n_elim_events + eps)
    violation_support_rate = violation_support / (n_elim_events + eps)
    violation_inversion_rate = violation_inversion / (n_elim_events + eps)
    metrics = {
        "upset_count": upset_count,
        "violation_count": violation_count,
        "violation_support": violation_support,
        "violation_inversion": violation_inversion,
        "violation_inversion_pairs": violation_inversion_pairs,
        "n_elim_events": n_elim_events,
        "n_elim_weeks": n_elim_weeks,
        "upset_rate": upset_rate,
        "violation_rate": violation_rate,
        "violation_support_rate": violation_support_rate,
        "violation_inversion_rate": violation_inversion_rate,
    }
    return placements, metrics


def _entropy_weights(metric_matrix: np.ndarray) -> np.ndarray:
    # metric_matrix: rows=candidates, cols=metrics (non-negative, already normalized)
    m = metric_matrix.copy().astype(float)
    n = m.shape[0]
    eps = 1e-12
    col_sum = m.sum(axis=0)
    zero_sum = col_sum <= eps
    col_sum_safe = np.where(zero_sum, 1.0, col_sum)
    p = m / col_sum_safe
    # p*log(p) should be 0 when p==0; avoid log(0) warnings
    log_p = np.where(p > 0, np.log(p), 0.0)
    k = 1.0 / np.log(max(n, 2))
    entropy = -k * (p * log_p).sum(axis=0)
    entropy = np.where(zero_sum, 1.0, entropy)  # 全零列视为无区分度
    d = 1 - entropy
    d_sum = d.sum()
    if d_sum <= eps:
        return np.ones(m.shape[1], dtype=float) / m.shape[1]
    return d / d_sum


def _ahp_weights() -> Tuple[np.ndarray, float]:
    # Fixed 4x4 comparison matrix (fairness, excitement, stability, interpretability)
    A = np.array(
        [
            [1, 3, 3, 5],
            [1/3, 1, 2, 3],
            [1/3, 1/2, 1, 2],
            [1/5, 1/3, 1/2, 1],
        ]
    )
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals.real)
    w = eigvecs[:, max_idx].real
    w = w / w.sum()
    n = A.shape[0]
    lam_max = eigvals.real[max_idx]
    ci = (lam_max - n) / (n - 1)
    ri = 0.9  # for n=4
    cr = ci / ri
    return w, float(cr)


def _pareto_front(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    # maximize metrics columns
    is_dominated = np.zeros(len(df), dtype=bool)
    vals = df[metrics].to_numpy()
    for i in range(len(df)):
        if is_dominated[i]:
            continue
        for j in range(len(df)):
            if i == j:
                continue
            if np.all(vals[j] >= vals[i]) and np.any(vals[j] > vals[i]):
                is_dominated[i] = True
                break
    return df.loc[~is_dominated].copy()


def run_local_refine(
    sim_data: pd.DataFrame,
    wide: pd.DataFrame,
    elim_count_map: Dict[Tuple[int, int], int],
    posterior: Dict[Tuple[int, int], pd.DataFrame] | None,
    ci_cache: Dict[Tuple[int, int, str], Tuple[float, float]],
    metric_cols: List[str],
    weights: np.ndarray,
    col_min: pd.Series,
    col_max: pd.Series,
    out_dir: str,
    seed: int,
    logger,
    anchor_source: str = "score_rank",
    anchor_exclude_edges: int = 1,
    edge_eps: float = 1e-6,
    anchor_wj_override: float | None = None,
    refine_step: float = 0.01,
    refine_half_width: float = 0.1,
    refine_seed: int = 123,
    score_rank: pd.DataFrame | None = None,
    final_rule: Dict[str, object] | None = None,
) -> str | None:
    # 默认开启，仅生成额外文件，不影响任何既有输出内容/随机序列
    anchor_source = (anchor_source or "score_rank").lower()
    anchor_used_mode = "score_rank_raw"
    anchor_wj = None
    anchor_save = None
    anchor_bottom_k = None

    def _select_score_rank_anchor(df: pd.DataFrame | None, exclude_edges: int, eps: float) -> Tuple[pd.Series | None, str]:
        if df is None or df.empty or "wJ" not in df.columns:
            return None, "score_rank_empty"
        sr = df.copy()
        sr["wJ"] = pd.to_numeric(sr["wJ"], errors="coerce")
        sr = sr.dropna(subset=["wJ"])
        if sr.empty:
            return None, "score_rank_empty"
        use_df = sr
        mode = "score_rank_raw"
        if exclude_edges:
            interior = sr[(sr["wJ"] > eps) & (sr["wJ"] < 1 - eps)]
            if not interior.empty:
                use_df = interior
                mode = "score_rank_interior"
        return use_df.iloc[0], mode

    anchor_row = None
    if anchor_source == "final":
        if isinstance(final_rule, dict):
            anchor_wj = final_rule.get("wJ")
            anchor_save = final_rule.get("save")
            anchor_bottom_k = final_rule.get("bottom_k")
            anchor_used_mode = "final"
        if anchor_wj is not None:
            try:
                wj_val = float(anchor_wj)
            except Exception:
                wj_val = None
            if wj_val is not None and anchor_exclude_edges and (wj_val <= edge_eps or wj_val >= 1 - edge_eps):
                anchor_row, _mode = _select_score_rank_anchor(score_rank, 1, edge_eps)
                if anchor_row is not None:
                    anchor_wj = anchor_row.get("wJ", anchor_wj)
                    anchor_save = anchor_row.get("save", anchor_save)
                    anchor_bottom_k = anchor_row.get("bottom_k", anchor_bottom_k)
                    anchor_used_mode = "final_fallback"
        if anchor_wj is None:
            anchor_row, anchor_used_mode = _select_score_rank_anchor(score_rank, anchor_exclude_edges, edge_eps)
            if anchor_row is not None:
                anchor_wj = anchor_row.get("wJ", None)
                anchor_save = anchor_row.get("save", None)
                anchor_bottom_k = anchor_row.get("bottom_k", None)
    else:
        anchor_row, anchor_used_mode = _select_score_rank_anchor(score_rank, anchor_exclude_edges, edge_eps)
        if anchor_row is not None:
            anchor_wj = anchor_row.get("wJ", None)
            anchor_save = anchor_row.get("save", None)
            anchor_bottom_k = anchor_row.get("bottom_k", None)

    if anchor_wj_override is not None and np.isfinite(anchor_wj_override):
        anchor_wj = float(anchor_wj_override)
        anchor_wj = min(max(anchor_wj, 0.0), 1.0)
        anchor_used_mode = "override"

    try:
        anchor_wj = float(anchor_wj)
    except Exception:
        anchor_wj = 0.5
    try:
        anchor_save = int(anchor_save)
    except Exception:
        anchor_save = 0
    try:
        anchor_bottom_k = int(anchor_bottom_k)
    except Exception:
        anchor_bottom_k = 2

    half_width = max(float(refine_half_width), 0.0)
    step = max(float(refine_step), 1e-6)
    start = max(0.0, anchor_wj - half_width)
    end = min(1.0, anchor_wj + half_width)
    grid = np.arange(start, end + 1e-12, step)
    decimals = int(max(0, np.ceil(-np.log10(step + 1e-12))))
    grid = np.round(grid, decimals)
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    if anchor_wj not in grid:
        grid = np.sort(np.unique(np.append(grid, anchor_wj)))

    season_groups = [(int(s), sdf.copy()) for s, sdf in wide.groupby("season")]
    season_meta = [(s, g["celebrity_name"].tolist(), int(g["season_length"].max())) for s, g in season_groups]
    season_sim = {s: sim_data[sim_data["season"] == s] for s, _ in season_groups}

    def pv_drawer(season: int, week: int, active_names: List[str], rng_rep: np.random.Generator) -> np.ndarray:
        pv_vec, _ = _draw_pv(season, week, active_names, rng_rep, posterior, ci_cache)
        return pv_vec

    col_min_vals = np.asarray(col_min.loc[metric_cols], dtype=float)
    col_max_vals = np.asarray(col_max.loc[metric_cols], dtype=float)
    weights_vals = np.asarray(weights, dtype=float)
    eps = 1e-12
    rows = []

    for wj in grid:
        wj = float(wj)
        wv = 1 - wj
        fairness_list = []
        excitement_list = []
        excitement_rate_list = []
        stability_list = []
        interpret_list = []
        violations_rate_list = []

        for season, contestants, season_length in season_meta:
            placements, metrics = _simulate_rule(
                int(season),
                season_length,
                contestants,
                sim_data,
                elim_count_map,
                wj,
                anchor_save,
                anchor_bottom_k,
            )

            season_data = season_sim[season]
            season_data = season_data[season_data["celebrity_name"].isin(contestants)].copy()
            if "judge_percent" in season_data.columns:
                cum_jpct = season_data.groupby("celebrity_name")["judge_percent"].sum()
            else:
                season_data["judge_share"] = season_data["judge_total"] / season_data.groupby(
                    ["season", "week"]
                )["judge_total"].transform("sum")
                season_data["judge_share"] = season_data["judge_share"].fillna(0)
                cum_jpct = season_data.groupby("celebrity_name")["judge_share"].sum()
            place_series = pd.Series(placements)
            placement_score = (len(contestants) + 1) - place_series
            common = cum_jpct.index.intersection(place_series.index)
            fairness = kendall_tau(cum_jpct.loc[common], placement_score.loc[common])

            fairness_list.append(fairness)
            excitement_list.append(metrics["upset_count"])
            excitement_rate_list.append(metrics["upset_rate"])
            interpret_list.append(metrics["violation_count"])
            violations_rate_list.append(metrics["violation_rate"])

        # stability via MC sampling of P^V (independent RNG)
        m = 30
        season_uncertainties = []
        for season, contestants, season_length in season_meta:
            n_contestants = len(contestants)
            places_mat = np.zeros((m, n_contestants), dtype=float)
            for rep in range(m):
                rng_rep = np.random.default_rng(
                    refine_seed + int(wj * 1000) + anchor_save * 100 + anchor_bottom_k * 10 + rep
                )
                placements, _ = _simulate_rule(
                    int(season),
                    season_length,
                    contestants,
                    sim_data,
                    elim_count_map,
                    wj,
                    anchor_save,
                    anchor_bottom_k,
                    rng=rng_rep,
                    pv_drawer=pv_drawer,
                    logger=logger,
                )
                places_mat[rep, :] = [placements.get(name, np.nan) for name in contestants]
            if np.isnan(places_mat).any():
                for j in range(n_contestants):
                    col = places_mat[:, j]
                    if np.isnan(col).any():
                        fill_val = float(np.nanmean(col)) if np.isfinite(np.nanmean(col)) else n_contestants / 2.0
                        col[np.isnan(col)] = fill_val
                        places_mat[:, j] = col
            var_per_contestant = np.var(places_mat, axis=0, ddof=0)
            season_uncertainty_raw = float(np.mean(var_per_contestant))
            max_var = (n_contestants ** 2 - 1) / 12.0
            season_uncertainty = season_uncertainty_raw / (max_var + 1e-12)
            season_uncertainties.append(season_uncertainty)
        overall_uncertainty = float(np.mean(season_uncertainties)) if season_uncertainties else np.nan
        stability = 1 / (1 + overall_uncertainty) if np.isfinite(overall_uncertainty) else np.nan
        stability_list.append(stability)

        fairness = float(np.nanmean(fairness_list))
        excitement = float(np.nanmean(excitement_list))
        excitement_rate = float(np.nanmean(excitement_rate_list))
        interpretability = float(-np.nanmean(interpret_list))
        violations_rate = float(np.nanmean(violations_rate_list))
        interpretability_rate = 1 - violations_rate
        interpretability_pos = interpretability_rate

        metric_lookup = {
            "fairness": fairness,
            "excitement_rate": excitement_rate,
            "stability": stability,
            "interpretability_pos": interpretability_pos,
        }
        metric_vals = np.array([metric_lookup.get(col, np.nan) for col in metric_cols], dtype=float)
        if not np.isfinite(metric_vals).all():
            metric_vals = np.where(np.isfinite(metric_vals), metric_vals, col_min_vals)
        norm = (metric_vals - col_min_vals) / (col_max_vals - col_min_vals + eps)
        score_entropy_local = float(np.dot(norm, weights_vals))
        knee_distance = float(np.linalg.norm(1 - norm))

        rows.append(
            {
                "wJ": wj,
                "wV": wv,
                "save": int(anchor_save),
                "bottom_k": int(anchor_bottom_k),
                "fairness": fairness,
                "excitement": excitement,
                "stability": stability,
                "interpretability": interpretability,
                "excitement_rate": excitement_rate,
                "violations_rate": violations_rate,
                "interpretability_rate": interpretability_rate,
                "score_entropy_local": score_entropy_local,
                "knee_distance": knee_distance,
            }
        )

    refine_df = pd.DataFrame(rows)
    if refine_df.empty:
        logger.warning("Q5 local refine produced empty grid.")
        return None
    base_cols = [
        "wJ",
        "wV",
        "save",
        "bottom_k",
        "fairness",
        "excitement",
        "stability",
        "interpretability",
        "score_entropy_local",
    ]
    extra_cols = [c for c in refine_df.columns if c not in base_cols]
    refine_df = refine_df[base_cols + extra_cols]

    out_path = Path(out_dir) / "q5_local_refine_curve.csv"
    write_csv(refine_df, str(out_path))

    settings = {
        "anchor_source": anchor_source,
        "anchor_used_mode": anchor_used_mode,
        "anchor_exclude_edges": int(anchor_exclude_edges),
        "edge_eps": float(edge_eps),
        "anchor_wj_override": None if anchor_wj_override is None else float(anchor_wj_override),
        "anchor_wJ": anchor_wj,
        "anchor_save": int(anchor_save),
        "anchor_bottom_k": int(anchor_bottom_k),
        "refine_step": float(step),
        "refine_half_width": float(half_width),
        "refine_seed": int(refine_seed),
        "metric_cols": metric_cols,
        "weights_fixed": weights_vals.tolist(),
        "col_min": col_min_vals.tolist(),
        "col_max": col_max_vals.tolist(),
    }
    write_json(settings, str(Path(out_dir) / "q5_local_refine_settings.json"))

    # 生成步长曲线图（PNG），不影响主流程输出
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(refine_df["wJ"], refine_df["score_entropy_local"], marker="o", linewidth=1.2)
        ax.set_xlabel("wJ")
        ax.set_ylabel("score_entropy_local")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / "q5_local_refine_score_entropy.png"), dpi=160)
        plt.close(fig)

        if "knee_distance" in refine_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(refine_df["wJ"], refine_df["knee_distance"], marker="o", linewidth=1.2)
            ax.set_xlabel("wJ")
            ax.set_ylabel("knee_distance")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(str(Path(out_dir) / "q5_local_refine_knee_distance.png"), dpi=160)
            plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional plot
        if logger is not None:
            logger.warning("Q5 local refine plot skipped: %s", exc)

    logger.info("Q5 local refine saved to %s", out_path)
    return str(out_path)


def run_q5(
    week_features: pd.DataFrame,
    wide: pd.DataFrame,
    q1_vote_point: pd.DataFrame,
    q1_vote_ci: pd.DataFrame,
    q4_pred: pd.DataFrame,
    out_dir: str,
    seed: int,
    logger,
    local_refine: int = 1,
    anchor_source: str = "score_rank",
    refine_step: float = 0.01,
    refine_half_width: float = 0.1,
    refine_seed: int = 123,
    anchor_exclude_edges: int = 1,
    edge_eps: float = 1e-6,
    anchor_wj_override: float | None = None,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    elim_count_map = build_elimination_count(wide)
    sim_data = _build_sim_data(week_features, q1_vote_point, q4_pred)

    candidates = []
    rng = np.random.default_rng(seed)

    posterior = _load_q1_posterior_samples(out_dir, logger)
    ci_cache = _build_ci_cache(q1_vote_ci)
    if posterior is not None:
        logger.info("Q5 stability MC uses posterior_samples.")
    else:
        logger.info("Q5 stability MC uses CI_fallback.")
    logger.info("Q5: placements mapping fixed; violation_count uses pv_used (posterior if available).")
    # 改进6：稳定性日志只打印一次
    logger.info("Q5 stability uses mean contestant rank variance across MC replicates.")

    def pv_drawer(season: int, week: int, active_names: List[str], rng_rep: np.random.Generator) -> np.ndarray:
        pv_vec, _ = _draw_pv(season, week, active_names, rng_rep, posterior, ci_cache)
        return pv_vec
    wj_grid = np.round(np.arange(0, 1.01, 0.05), 2)
    # 改进6：缓存赛季分组信息，避免重复 groupby
    season_groups = [(int(s), sdf.copy()) for s, sdf in wide.groupby("season")]
    season_meta = [(s, g["celebrity_name"].tolist(), int(g["season_length"].max())) for s, g in season_groups]
    season_sim = {s: sim_data[sim_data["season"] == s].copy() for s, _ in season_groups}
    for wj in wj_grid:
        for save in [0, 1]:
            for bottom_k in [2, 3]:
                fairness_list = []
                excitement_list = []
                excitement_rate_list = []
                stability_list = []
                interpret_list = []
                violations_rate_list = []
                violation_support_rate_list = []
                violation_inversion_rate_list = []

                for season, contestants, season_length in season_meta:
                    placements, metrics = _simulate_rule(
                        int(season),
                        season_length,
                        contestants,
                        sim_data,
                        elim_count_map,
                        wj,
                        save,
                        bottom_k,
                    )

                    # fairness: Kendall tau between placement_score and cumulative judge_percent (higher is better)
                    season_data = season_sim[season]
                    season_data = season_data[season_data["celebrity_name"].isin(contestants)].copy()
                    if "judge_percent" in season_data.columns:
                        cum_jpct = season_data.groupby("celebrity_name")["judge_percent"].sum()
                    else:
                        # derive weekly judge share then sum across weeks
                        season_data["judge_share"] = season_data["judge_total"] / season_data.groupby(
                            ["season", "week"]
                        )["judge_total"].transform("sum")
                        season_data["judge_share"] = season_data["judge_share"].fillna(0)
                        cum_jpct = season_data.groupby("celebrity_name")["judge_share"].sum()
                    place_series = pd.Series(placements)
                    placement_score = (len(contestants) + 1) - place_series  # higher is better
                    common = cum_jpct.index.intersection(place_series.index)
                    fairness = kendall_tau(cum_jpct.loc[common], placement_score.loc[common])

                    fairness_list.append(fairness)
                    excitement_list.append(metrics["upset_count"])
                    excitement_rate_list.append(metrics["upset_rate"])
                    interpret_list.append(metrics["violation_count"])
                    violations_rate_list.append(metrics["violation_rate"])
                    violation_support_rate_list.append(metrics["violation_support_rate"])
                    violation_inversion_rate_list.append(metrics["violation_inversion_rate"])

                # stability via MC sampling of P^V
                m = 30
                season_uncertainties = []
                for season, contestants, season_length in season_meta:
                    n_contestants = len(contestants)
                    places_mat = np.zeros((m, n_contestants), dtype=float)
                    for rep in range(m):
                        rng_rep = np.random.default_rng(seed + int(wj * 1000) + save * 100 + bottom_k * 10 + rep)
                        placements, _ = _simulate_rule(
                            int(season),
                            season_length,
                            contestants,
                            sim_data,
                            elim_count_map,
                            wj,
                            save,
                            bottom_k,
                            rng=rng_rep,
                            pv_drawer=pv_drawer,
                            logger=logger,
                        )
                        places_mat[rep, :] = [placements.get(name, np.nan) for name in contestants]
                    # fill missing ranks per contestant (mean of available replicates; else N/2)
                    if np.isnan(places_mat).any():
                        missing_count = 0
                        for j in range(n_contestants):
                            col = places_mat[:, j]
                            if np.isnan(col).any():
                                fill_val = float(np.nanmean(col)) if np.isfinite(np.nanmean(col)) else n_contestants / 2.0
                                missing_count += int(np.isnan(col).sum())
                                col[np.isnan(col)] = fill_val
                                places_mat[:, j] = col
                        if missing_count > 0:
                            logger.warning(
                                "Q5 stability: filled %s missing ranks for season %s.",
                                missing_count,
                                season,
                            )
                    var_per_contestant = np.var(places_mat, axis=0, ddof=0)
                    season_uncertainty_raw = float(np.mean(var_per_contestant))
                    max_var = (n_contestants ** 2 - 1) / 12.0
                    season_uncertainty = season_uncertainty_raw / (max_var + 1e-12)  # normalize to [0,1]
                    season_uncertainties.append(season_uncertainty)
                overall_uncertainty = float(np.mean(season_uncertainties)) if season_uncertainties else np.nan
                logger.debug(
                    "Q5 stability MC summary: M=%s, seasons=%s, mean_uncertainty=%.6f",
                    m,
                    len(season_uncertainties),
                    overall_uncertainty if np.isfinite(overall_uncertainty) else float("nan"),
                )
                stability = 1 / (1 + overall_uncertainty) if np.isfinite(overall_uncertainty) else np.nan

                candidates.append(
                    {
                        "wJ": wj,
                        "wV": 1 - wj,
                        "save": save,
                        "bottom_k": bottom_k,
                        "fairness": float(np.nanmean(fairness_list)),
                        "excitement": float(np.nanmean(excitement_list)),
                        "excitement_rate": float(np.nanmean(excitement_rate_list)),
                        "stability": float(stability),
                        "interpretability": float(-np.nanmean(interpret_list)),
                        "violations": float(np.nanmean(interpret_list)),
                        "violations_rate": float(np.nanmean(violations_rate_list)),
                        "violation_support_rate": float(np.nanmean(violation_support_rate_list)),
                        "violation_inversion_rate": float(np.nanmean(violation_inversion_rate_list)),
                    }
                )

    cand_df = pd.DataFrame(candidates)
    # 改进4：正向 interpretability_rate/pos
    if "violations_rate" in cand_df.columns:
        cand_df["interpretability_rate"] = 1 - cand_df["violations_rate"]
        cand_df["interpretability_pos"] = cand_df["interpretability_rate"]

    # Entropy weights
    metric_cols = ["fairness", "excitement_rate", "stability", "interpretability_pos"]
    metric_df = cand_df[metric_cols].copy()
    # 缺失值处理：用列中位数（稳健）填补，避免直接 fillna(0) 引入尺度偏差
    if metric_df.isna().any().any():
        col_median = metric_df.median()
        metric_df = metric_df.fillna(col_median)
        if metric_df.isna().any().any():
            col_mean = metric_df.mean()
            metric_df = metric_df.fillna(col_mean)
        if metric_df.isna().any().any():
            metric_df = metric_df.fillna(0.0)
            if logger is not None:
                logger.warning("Q5 entropy: fallback fillna(0) due to all-NaN columns.")
    # 同量纲化：Min-Max 归一化到 [0,1]，避免计数指标主导
    eps = 1e-12
    col_min = metric_df.min(axis=0)
    col_max = metric_df.max(axis=0)
    metric_norm = (metric_df - col_min) / (col_max - col_min + eps)
    weights = _entropy_weights(metric_norm.to_numpy())
    entropy_df = pd.DataFrame({"metric": metric_cols, "weight": weights})

    # AHP weights
    ahp_w, cr = _ahp_weights()
    ahp_df = pd.DataFrame({"metric": metric_cols, "weight": ahp_w})
    ahp_df["consistency_ratio"] = cr

    # Score rank (entropy weights)
    cand_df["score_entropy"] = (metric_norm * weights).sum(axis=1)
    score_rank = cand_df.sort_values("score_entropy", ascending=False).copy()

    # Pareto front
    pareto = _pareto_front(cand_df, metric_cols)

    # Final rule selection
    # 改进5：基准公平性改为分位数阈值
    BASELINE_FAIRNESS_Q = 0.6
    baseline_fairness = float(cand_df["fairness"].quantile(BASELINE_FAIRNESS_Q))
    pareto_ok = pareto[pareto["fairness"] >= baseline_fairness]
    if pareto_ok.empty:
        pareto_ok = pareto
    final = pareto_ok.sort_values(["violations", "fairness"], ascending=[True, False]).iloc[0]
    final_rule = {
        "wJ": float(final["wJ"]),
        "wV": float(final["wV"]),
        "save": int(final["save"]),
        "bottom_k": int(final["bottom_k"]),
        "metrics": {k: float(final[k]) for k in metric_cols + ["violations", "violations_rate", "interpretability_rate"]},
        "baseline_fairness": float(baseline_fairness),
        "baseline_quantile": float(BASELINE_FAIRNESS_Q),
    }

    out_paths = {
        "q5_rule_candidates": str(Path(out_dir) / "q5_rule_candidates.csv"),
        "q5_entropy_weights": str(Path(out_dir) / "q5_entropy_weights.csv"),
        "q5_ahp_weights": str(Path(out_dir) / "q5_ahp_weights.csv"),
        "q5_score_rank": str(Path(out_dir) / "q5_score_rank.csv"),
        "q5_pareto_front": str(Path(out_dir) / "q5_pareto_front.csv"),
        "q5_final_rule": str(Path(out_dir) / "q5_final_rule.json"),
    }

    write_csv(cand_df, out_paths["q5_rule_candidates"])
    write_csv(entropy_df, out_paths["q5_entropy_weights"])
    write_csv(ahp_df, out_paths["q5_ahp_weights"])
    write_csv(score_rank, out_paths["q5_score_rank"])
    write_csv(pareto, out_paths["q5_pareto_front"])
    write_json(final_rule, out_paths["q5_final_rule"])

    # 改进1/4：运行结束时的轻量 sanity check（只打印一次）
    if not cand_df.empty and "violations_rate" in cand_df.columns:
        v_rate = cand_df["violations_rate"]
        e_rate = cand_df["excitement_rate"]
        logger.info(
            "Q5 rates summary: violations_rate nonzero=%.4f min=%.6f median=%.6f max=%.6f",
            float((v_rate > 0).mean()),
            float(v_rate.min()),
            float(v_rate.median()),
            float(v_rate.max()),
        )
        logger.info(
            "Q5 rates summary: excitement_rate min=%.6f median=%.6f max=%.6f",
            float(e_rate.min()),
            float(e_rate.median()),
            float(e_rate.max()),
        )

    # 本地加密网格（默认开启），仅额外输出，不影响既有结果
    if local_refine:
        run_local_refine(
            sim_data=sim_data,
            wide=wide,
            elim_count_map=elim_count_map,
            posterior=posterior,
            ci_cache=ci_cache,
            metric_cols=list(metric_cols),
            weights=np.array(weights, dtype=float),
            col_min=col_min.copy(),
            col_max=col_max.copy(),
            out_dir=out_dir,
            seed=seed,
            logger=logger,
            anchor_source=anchor_source,
            anchor_exclude_edges=anchor_exclude_edges,
            edge_eps=edge_eps,
            anchor_wj_override=anchor_wj_override,
            refine_step=refine_step,
            refine_half_width=refine_half_width,
            refine_seed=refine_seed,
            score_rank=score_rank.copy(),
            final_rule=final_rule.copy(),
        )

    logger.info("Q5 outputs saved to %s", out_dir)
    return out_paths


def run_q5_refine_only(
    data_dir: str,
    out_dir: str,
    seed: int,
    logger,
    anchor_source: str = "score_rank",
    anchor_exclude_edges: int = 1,
    edge_eps: float = 1e-6,
    anchor_wj_override: float | None = None,
    refine_step: float = 0.01,
    refine_half_width: float = 0.1,
    refine_seed: int = 123,
) -> str | None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    out_root = Path(out_dir)
    if not out_root.is_absolute():
        out_root = project_root / out_root

    q5_dir = out_root / "q5"
    ensure_dir(str(q5_dir))
    required = ["q5_rule_candidates.csv", "q5_score_rank.csv", "q5_entropy_weights.csv"]
    if anchor_source == "final":
        required.append("q5_final_rule.json")
    missing = [name for name in required if not (q5_dir / name).exists()]
    if missing:
        raise SystemExit(f"Q5 refine-only missing files: {missing}")

    week_features = pd.read_csv(data_dir / "dwts_week_features_active.csv")
    wide = pd.read_csv(data_dir / "dwts_clean_wide.csv")
    q1_vote_point = pd.read_csv(out_root / "q1" / "q1_vote_point.csv")
    q1_vote_ci = pd.read_csv(out_root / "q1" / "q1_vote_ci.csv")
    q4_pred_path = out_root / "q4" / "q4_pred_for_counterfactual.csv"
    q4_pred = pd.read_csv(q4_pred_path) if q4_pred_path.exists() else pd.DataFrame()

    cand_df = pd.read_csv(q5_dir / "q5_rule_candidates.csv")
    score_rank = pd.read_csv(q5_dir / "q5_score_rank.csv")
    entropy_df = pd.read_csv(q5_dir / "q5_entropy_weights.csv")
    final_rule = None
    if (q5_dir / "q5_final_rule.json").exists():
        final_rule = pd.read_json(q5_dir / "q5_final_rule.json", typ="series").to_dict()

    metric_cols = ["fairness", "excitement_rate", "stability", "interpretability_pos"]
    if "interpretability_pos" not in cand_df.columns:
        if "violations_rate" in cand_df.columns:
            cand_df["interpretability_pos"] = 1 - cand_df["violations_rate"]
        elif "interpretability_rate" in cand_df.columns:
            cand_df["interpretability_pos"] = cand_df["interpretability_rate"]
    for col in metric_cols:
        if col not in cand_df.columns:
            raise SystemExit(f"Q5 refine-only missing metric column: {col}")

    metric_df = cand_df[metric_cols].copy()
    if metric_df.isna().any().any():
        col_median = metric_df.median()
        metric_df = metric_df.fillna(col_median)
        if metric_df.isna().any().any():
            col_mean = metric_df.mean()
            metric_df = metric_df.fillna(col_mean)
        if metric_df.isna().any().any():
            metric_df = metric_df.fillna(0.0)
            if logger is not None:
                logger.warning("Q5 refine-only: fallback fillna(0) due to all-NaN columns.")
    col_min = metric_df.min(axis=0)
    col_max = metric_df.max(axis=0)

    weight_map = dict(zip(entropy_df["metric"], entropy_df["weight"]))
    if any(col not in weight_map for col in metric_cols):
        raise SystemExit("Q5 refine-only entropy weights missing required metrics.")
    weights = np.array([weight_map[col] for col in metric_cols], dtype=float)

    elim_count_map = build_elimination_count(wide)
    sim_data = _build_sim_data(week_features, q1_vote_point, q4_pred)
    posterior = _load_q1_posterior_samples(str(q5_dir), logger)
    ci_cache = _build_ci_cache(q1_vote_ci)

    return run_local_refine(
        sim_data=sim_data,
        wide=wide,
        elim_count_map=elim_count_map,
        posterior=posterior,
        ci_cache=ci_cache,
        metric_cols=list(metric_cols),
        weights=weights,
        col_min=col_min,
        col_max=col_max,
        out_dir=str(q5_dir),
        seed=seed,
        logger=logger,
        anchor_source=anchor_source,
        anchor_exclude_edges=anchor_exclude_edges,
        edge_eps=edge_eps,
        anchor_wj_override=anchor_wj_override,
        refine_step=refine_step,
        refine_half_width=refine_half_width,
        refine_seed=refine_seed,
        score_rank=score_rank,
        final_rule=final_rule,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q5 local refine runner (default: local_refine=1)")
    parser.add_argument("--data_dir", default="./data_processed", help="Processed data directory")
    parser.add_argument("--out_dir", default="./outputs/solve", help="Output root directory")
    parser.add_argument("--seed", default=2026, type=int, help="Random seed")
    parser.add_argument("--local_refine", default=1, type=int, help="Enable local refine (1=on, 0=off)")
    parser.add_argument("--anchor_source", default="score_rank", choices=["score_rank", "final"])
    parser.add_argument("--anchor_exclude_edges", default=1, type=int)
    parser.add_argument("--edge_eps", default=1e-6, type=float)
    parser.add_argument("--anchor_wj_override", default=None, type=float)
    parser.add_argument("--refine_step", default=0.01, type=float)
    parser.add_argument("--refine_half_width", default=0.1, type=float)
    parser.add_argument("--refine_seed", default=123, type=int)
    return parser.parse_args()


def _read_csv_or_exit(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {name} -> {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = project_root / out_root

    log_dir = out_root / "logs"
    logger = get_logger("q5_local_refine", str(log_dir / "q5_local_refine.log"))

    week_features = _read_csv_or_exit(data_dir / "dwts_week_features_active.csv", "week_features")
    wide = _read_csv_or_exit(data_dir / "dwts_clean_wide.csv", "wide")
    q1_dir = out_root / "q1"
    q4_dir = out_root / "q4"
    q5_dir = out_root / "q5"
    ensure_dir(str(q5_dir))

    q1_vote_point = _read_csv_or_exit(q1_dir / "q1_vote_point.csv", "q1_vote_point")
    q1_vote_ci = _read_csv_or_exit(q1_dir / "q1_vote_ci.csv", "q1_vote_ci")
    q4_pred_path = q4_dir / "q4_pred_for_counterfactual.csv"
    q4_pred = pd.read_csv(q4_pred_path) if q4_pred_path.exists() else pd.DataFrame()

    run_q5(
        week_features,
        wide,
        q1_vote_point,
        q1_vote_ci,
        q4_pred,
        str(q5_dir),
        args.seed,
        logger,
        local_refine=args.local_refine,
        anchor_source=args.anchor_source,
        anchor_exclude_edges=args.anchor_exclude_edges,
        edge_eps=args.edge_eps,
        anchor_wj_override=args.anchor_wj_override,
        refine_step=args.refine_step,
        refine_half_width=args.refine_half_width,
        refine_seed=args.refine_seed,
    )

# 如何关闭 local refine（默认开启）:
# python q5_new_rule.py --local_refine 0
