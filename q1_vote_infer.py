# NOTE: 仅新增 Q1 后验样本明细落盘与 Q2 优先读取后验样本传播；未改变任何模型/规则算法.
# Self-test: python main_solve.py --task q1 --data_dir ./data_processed --out_dir ./outputs/solve --seed 2026
# Self-test: python main_solve.py --task q2 --data_dir ./data_processed --out_dir ./outputs/solve --seed 2026
from __future__ import annotations



from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir, write_csv


PATH_HINT = "# NOTE: Replace with your local absolute path if needed."
PHI_PRIOR_A0 = 2.0
PHI_PRIOR_B0 = 2.0
PHI_MIN_ACCEPT = 30
PHI_SUMMARY_SAMPLES = 2000
STATE_SIGMA = 0.3


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def _summarize_phi(
    phi_samples: List[float],
    prior_a: float,
    prior_b: float,
    rng: np.random.Generator,
    fallback_n: int = PHI_SUMMARY_SAMPLES,
) -> Tuple[float, float, float, float]:
    if phi_samples:
        arr = np.asarray(phi_samples, dtype=float)
    else:
        arr = rng.beta(prior_a, prior_b, size=fallback_n)
    return (
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


def _beta_moment_match(
    phi_samples: List[float],
    prior_a: float,
    prior_b: float,
    logger=None,
) -> Tuple[float, float, bool]:
    if len(phi_samples) < 2:
        if logger is not None:
            logger.warning("phi posterior degenerate, keep prior (too few samples)")
        return max(1e-3, prior_a), max(1e-3, prior_b), False

    arr = np.asarray(phi_samples, dtype=float)
    m = float(np.mean(arr))
    v = float(np.var(arr, ddof=1))
    v = max(v, 1e-12)
    k = m * (1 - m) / v - 1.0

    if (not np.isfinite(k)) or k <= 0:
        if logger is not None:
            logger.warning("phi posterior degenerate, keep prior (k<=0 or non-finite)")
        return max(1e-3, prior_a), max(1e-3, prior_b), False

    a_new = max(1e-3, m * k)
    b_new = max(1e-3, (1 - m) * k)
    if not (np.isfinite(a_new) and np.isfinite(b_new)):
        if logger is not None:
            logger.warning("phi posterior degenerate, keep prior (non-finite params)")
        return max(1e-3, prior_a), max(1e-3, prior_b), False

    return a_new, b_new, True


def get_rule_type(season: int) -> str:
    if season <= 2:
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    return "save"


def predict_elimination(rule_type: str, group_df: pd.DataFrame, pv_vector: np.ndarray, elim_k: int) -> List[str]:
    if elim_k <= 0:
        return []

    if rule_type == "percent":
        c_pct = group_df["judge_percent"].to_numpy(dtype=float) + pv_vector
        order = np.argsort(c_pct)
        return [group_df.iloc[i]["celebrity_name"] for i in order[:elim_k]]

    rv = pd.Series(pv_vector).rank(ascending=False, method="min").to_numpy()
    c_rank = group_df["judge_rank"].to_numpy(dtype=float) + rv
    order = np.argsort(-c_rank)

    if rule_type == "rank":
        return [group_df.iloc[i]["celebrity_name"] for i in order[:elim_k]]

    # save
    if elim_k == 1:
        bottom_two = order[:2]
        if len(bottom_two) < 2:
            return [group_df.iloc[order[0]]["celebrity_name"]]
        judge_totals = group_df.iloc[bottom_two]["judge_total"].to_numpy(dtype=float)
        elim_idx = bottom_two[np.argmin(judge_totals)]
        return [group_df.iloc[elim_idx]["celebrity_name"]]

    # For multiple eliminations, approximate with rank
    return [group_df.iloc[i]["celebrity_name"] for i in order[:elim_k]]


def _margin_by_rule(
    rule_type: str,
    group_df: pd.DataFrame,
    pv_vector: np.ndarray,
    elim_k: int,
) -> Tuple[float, str, int]:
    if elim_k <= 0 or len(group_df) < 2:
        return np.nan, "na", 0

    if rule_type == "percent":
        c_pct = group_df["judge_percent"].to_numpy(dtype=float) + pv_vector
        order = np.argsort(c_pct)
        margin = float(c_pct[order[1]] - c_pct[order[0]]) if len(c_pct) >= 2 else np.nan
        return margin, "percent_margin", 0

    rv = pd.Series(pv_vector).rank(ascending=False, method="min").to_numpy()
    c_rank = group_df["judge_rank"].to_numpy(dtype=float) + rv
    order = np.argsort(-c_rank)

    if rule_type == "rank" or elim_k > 1:
        margin = float(c_rank[order[0]] - c_rank[order[1]]) if len(c_rank) >= 2 else np.nan
        approx = 1 if (rule_type == "save" and elim_k > 1) else 0
        return margin, "rank_margin", approx

    # save with single elimination: margin is judge_total gap in bottom-two
    bottom_two = order[:2]
    if len(bottom_two) < 2:
        return np.nan, "save_margin", 0
    bottom_df = group_df.iloc[bottom_two]
    judge_totals = bottom_df["judge_total"].to_numpy(dtype=float)
    margin = float(np.abs(judge_totals[0] - judge_totals[1]))
    return margin, "save_judge_gap", 0


def build_elimination_map(wide: pd.DataFrame) -> Dict[Tuple[int, int], List[str]]:
    elim_map: Dict[Tuple[int, int], List[str]] = {}
    for _, row in wide.iterrows():
        season = int(row["season"])
        last_week = row.get("last_active_week")
        season_length = row.get("season_length")
        if pd.isna(last_week) or pd.isna(season_length):
            continue
        last_week = int(last_week)
        season_length = int(season_length)
        if last_week >= season_length:
            continue  # finale week
        key = (season, last_week)
        elim_map.setdefault(key, []).append(row["celebrity_name"])
    return elim_map


def solve_week(
    group: pd.DataFrame,
    prev_pv: np.ndarray,
    elim_list: List[str],
    lambda_grid: List[Tuple[float, float]],
    eps: float = 1e-4,
    penalty_mu: float = 100.0,
    rule_type: str = "percent",
    logger=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    Jpct = group["judge_percent"].to_numpy(dtype=float)
    n = len(Jpct)

    if prev_pv is None or len(prev_pv) != n:
        prev_pv = Jpct.copy()
        prev_pv = prev_pv / prev_pv.sum() if prev_pv.sum() > 0 else np.ones(n) / n

    installed = set(cp.installed_solvers())
    solver_pref = ["ECOS", "OSQP", "SCS"]
    solver_list = [s for s in solver_pref if s in installed]
    if not solver_list:
        solver_list = [None]

    best_obj = np.inf
    best_p = None
    best_info = {
        "lambda1": np.nan,
        "lambda2": np.nan,
        "status": "failed",
        "objective": np.nan,
        "solver_name": "",
        "total_slack": np.nan,
        "max_slack": np.nan,
        "infeasibility_flag": 1,
        "rule_type": rule_type,
    }

    elim_idx = [i for i, name in enumerate(group["celebrity_name"].tolist()) if name in elim_list]
    non_elim_idx = [i for i in range(n) if i not in elim_idx]

    for lambda1, lambda2 in lambda_grid:
        p = cp.Variable(n)
        constraints = [p >= 0, cp.sum(p) == 1]

        slack = None
        if elim_idx and non_elim_idx:
            m = len(elim_idx) * len(non_elim_idx)
            slack = cp.Variable(m, nonneg=True)
            s_ptr = 0
            for e in elim_idx:
                for i in non_elim_idx:
                    constraints.append(Jpct[e] + p[e] <= Jpct[i] + p[i] - eps + slack[s_ptr])
                    s_ptr += 1

        obj = lambda1 * cp.sum_squares(p - prev_pv) + lambda2 * cp.sum_squares(p - Jpct)
        if slack is not None:
            obj = obj + penalty_mu * cp.sum(slack)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        for solver_name in solver_list:
            try:
                if solver_name:
                    prob.solve(solver=solver_name, warm_start=True)
                else:
                    prob.solve(warm_start=True)
            except Exception as e:
                if logger is not None:
                    logger.warning(
                        "Q1 solve_week failed: season=%s week=%s solver=%s lambda=(%s,%s) err=%s",
                        group["season"].iloc[0] if "season" in group.columns else "NA",
                        group["week"].iloc[0] if "week" in group.columns else "NA",
                        solver_name,
                        lambda1,
                        lambda2,
                        str(e),
                    )
                continue

            if p.value is None:
                continue
            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                continue

            total_slack = float(np.sum(slack.value)) if slack is not None and slack.value is not None else 0.0
            max_slack = float(np.max(slack.value)) if slack is not None and slack.value is not None else 0.0

            if prob.value < best_obj:
                best_obj = prob.value
                best_p = np.clip(p.value, 0, None)
                best_p = best_p / best_p.sum() if best_p.sum() > 0 else np.ones(n) / n
                best_info = {
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "status": prob.status,
                    "objective": float(prob.value),
                    "solver_name": solver_name or "auto",
                    "total_slack": total_slack,
                    "max_slack": max_slack,
                    "infeasibility_flag": int(total_slack > 0),
                    "rule_type": rule_type,
                }
            break

    if best_p is None:
        fallback = Jpct.copy()
        fallback = fallback / fallback.sum() if fallback.sum() > 0 else np.ones(n) / n
        best_p = fallback
        best_info = {
            "lambda1": np.nan,
            "lambda2": np.nan,
            "status": "fallback",
            "objective": np.nan,
            "solver_name": "fallback",
            "total_slack": np.nan,
            "max_slack": np.nan,
            "infeasibility_flag": 1,
            "rule_type": rule_type,
        }

    return best_p, best_info


def abc_sample_week(
    group: pd.DataFrame,
    elim_list: List[str],
    feature_matrix: np.ndarray,
    rule_type: str,
    elim_k: int,
    delta_s: float,
    eta_t: float,
    prev_dev: Optional[np.ndarray] = None,
    phi_prior_a: float = PHI_PRIOR_A0,
    phi_prior_b: float = PHI_PRIOR_B0,
    min_accept: int = PHI_MIN_ACCEPT,
    state_sigma: float = STATE_SIGMA,
    target_accept: int = 200,
    max_tries: int = 3000,
    seed: int = 2026,
    logger=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    Jpct = group["judge_percent"].to_numpy(dtype=float)
    n = len(Jpct)

    accepted: List[np.ndarray] = []
    accepted_phi: List[float] = []
    u_center_sum = np.zeros(n, dtype=float)
    mu_center_sum = np.zeros(n, dtype=float)
    tries = 0
    dist_sum = 0.0
    dist_min = np.inf
    dev_prev = np.zeros(n, dtype=float)
    if prev_dev is not None and len(prev_dev) == n:
        dev_prev = np.asarray(prev_dev, dtype=float)
        dev_prev = np.where(np.isfinite(dev_prev), dev_prev, 0.0)

    def draw_proposal(alpha_sd: float) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        beta = rng.normal(1.0, 0.3)
        gamma = rng.normal(0, 0.2, size=feature_matrix.shape[1]) if feature_matrix.size else np.array([])
        alpha = rng.normal(0, alpha_sd, size=n)
        phi = float(rng.beta(phi_prior_a, phi_prior_b))
        mu = alpha + beta * Jpct + delta_s + eta_t
        if feature_matrix.size:
            mu = mu + feature_matrix @ gamma
        mu_c = mu - np.mean(mu)
        eps = rng.normal(0, state_sigma, size=n)
        u_c = mu_c + phi * dev_prev + eps
        u_c = u_c - np.mean(u_c)
        pv = _softmax(u_c)
        return pv, phi, mu_c, u_c

    if not elim_list:
        for _ in range(target_accept):
            pv, phi, mu_c, u_c = draw_proposal(alpha_sd=0.5)
            accepted.append(pv)
            accepted_phi.append(phi)
            u_center_sum += u_c
            mu_center_sum += mu_c
            dist_sum += 0.0
            dist_min = 0.0
        tries = target_accept
        threshold_used = 0
    else:
        max_threshold = min(len(elim_list) + 1, 3)
        thresholds = list(range(0, max_threshold + 1))
        threshold_used = thresholds[-1]
        for threshold in thresholds:
            while tries < max_tries and len(accepted) < target_accept:
                tries += 1
                pv, phi, mu_c, u_c = draw_proposal(alpha_sd=0.6)

                predicted = predict_elimination(rule_type, group, pv, elim_k)
                dist = len(set(predicted) ^ set(elim_list))
                dist_sum += dist
                dist_min = min(dist_min, dist)

                if dist <= threshold:
                    accepted.append(pv)
                    accepted_phi.append(phi)
                    u_center_sum += u_c
                    mu_center_sum += mu_c

            threshold_used = threshold
            if len(accepted) >= target_accept or tries >= max_tries:
                break

    accept_rate = len(accepted) / tries if tries > 0 else 0
    mean_dist = dist_sum / tries if tries > 0 else np.nan
    min_dist = dist_min if np.isfinite(dist_min) else np.nan

    accepted_count = len(accepted)
    p_correct_exact = 0.0
    p_correct_within = 0.0
    pred_mode_elim = ""
    pred_mode_prob = 0.0
    elim_posterior = {name: 0.0 for name in group["celebrity_name"].tolist()}

    if accepted_count > 0:
        pred_counts: Dict[Tuple[str, ...], int] = {}
        correct_exact = 0
        correct_within = 0
        elim_counts = {name: 0 for name in group["celebrity_name"].tolist()}

        for pv in accepted:
            pred = predict_elimination(rule_type, group, pv, elim_k)
            pred_set = tuple(sorted(pred))
            dist = len(set(pred) ^ set(elim_list))
            if dist == 0:
                correct_exact += 1
            if dist <= threshold_used:
                correct_within += 1
            pred_counts[pred_set] = pred_counts.get(pred_set, 0) + 1
            for name in pred_set:
                elim_counts[name] += 1

        p_correct_exact = correct_exact / accepted_count
        p_correct_within = correct_within / accepted_count

        mode_set = max(pred_counts.items(), key=lambda x: x[1])[0] if pred_counts else tuple()
        pred_mode_elim = "|".join(mode_set)
        pred_mode_prob = (pred_counts[mode_set] / accepted_count) if pred_counts else 0.0

        elim_posterior = {name: elim_counts[name] / accepted_count for name in elim_counts}

    info = {
        "accepted": len(accepted),
        "tries": tries,
        "accept_rate": accept_rate,
        "ess": len(accepted),
        "rule_type": rule_type,
        "dist_threshold": threshold_used,
        "mean_dist": mean_dist,
        "min_dist": min_dist,
        "p_correct_exact": p_correct_exact,
        "p_correct_within_threshold": p_correct_within,
        "pred_mode_elim": pred_mode_elim,
        "pred_mode_prob": pred_mode_prob,
        "elim_posterior": elim_posterior,
    }

    phi_post_mean, phi_post_median, phi_post_q025, phi_post_q975 = _summarize_phi(
        accepted_phi, phi_prior_a, phi_prior_b, rng
    )
    phi_next_a = phi_prior_a
    phi_next_b = phi_prior_b
    phi_update_flag = False
    if accepted_count >= min_accept:
        phi_next_a, phi_next_b, phi_update_flag = _beta_moment_match(accepted_phi, phi_prior_a, phi_prior_b, logger)
    else:
        if logger is not None:
            logger.warning(
                "Q1 ABC phi: accepted_n < MIN_ACCEPT; keep prior (accepted=%s min_accept=%s)",
                accepted_count,
                min_accept,
            )

    u_center_post = None
    mu_center_post = None
    if accepted_count > 0:
        u_center_post = u_center_sum / accepted_count
        mu_center_post = mu_center_sum / accepted_count

    info.update(
        {
            "phi_prior_a": phi_prior_a,
            "phi_prior_b": phi_prior_b,
            "phi_post_mean": phi_post_mean,
            "phi_post_median": phi_post_median,
            "phi_post_q025": phi_post_q025,
            "phi_post_q975": phi_post_q975,
            "phi_next_a": phi_next_a,
            "phi_next_b": phi_next_b,
            "phi_update_flag": int(phi_update_flag),
            "phi_accepted_n": accepted_count,
            "u_center_post": u_center_post,
            "mu_center_post": mu_center_post,
        }
    )

    if not accepted:
        accepted.append(Jpct / Jpct.sum() if Jpct.sum() > 0 else np.ones(n) / n)

    return np.vstack(accepted), info


def run_q1(
    week_features: pd.DataFrame,
    wide: pd.DataFrame,
    out_dir: str,
    seed: int,
    logger,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    elim_map = build_elimination_map(wide)
    lambda_grid = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0)]

    outputs = []
    consistency_rows = []
    snapshot_rows = []
    abc_rows = []
    bayes_consistency_rows = []
    ci_rows = []
    elim_post_rows = []
    phi_rows = []
    state_rows = []
    pv_samples_frames: List[pd.DataFrame] = []
    excluded_weeks = 0
    total_weeks = 0

    week_features = week_features.copy()
    week_features["season"] = week_features["season"].astype(int)
    week_features["week"] = week_features["week"].astype(int)

    # Build features for ABC (include ballroom_partner)
    feature_cols = [
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry_region",
        "ballroom_partner",
    ]
    feature_df = wide[["season", "celebrity_name", "celebrity_age_during_season"] + feature_cols].copy()
    feature_df["celebrity_age_during_season"] = pd.to_numeric(feature_df["celebrity_age_during_season"], errors="coerce")
    dummies = pd.get_dummies(feature_df[feature_cols], dummy_na=True, prefix=feature_cols)
    feature_df = pd.concat([feature_df[["season", "celebrity_name", "celebrity_age_during_season"]], dummies], axis=1)

    # Random effects draws reused by season/week
    rng = np.random.default_rng(seed + 12345)
    season_effects = {int(s): float(rng.normal(0, 0.2)) for s in week_features["season"].unique()}
    week_effects = {int(w): float(rng.normal(0, 0.2)) for w in week_features["week"].unique()}

    for season, season_df in week_features.groupby("season"):
        season_df = season_df.sort_values("week")
        prev_pv_map: Dict[str, float] = {}
        prev_u_center_map: Dict[str, float] = {}
        prev_mu_center_map: Dict[str, float] = {}
        phi_prior_a = PHI_PRIOR_A0
        phi_prior_b = PHI_PRIOR_B0
        rule_type = get_rule_type(int(season))

        for week, group in season_df.groupby("week"):
            group = group.copy().reset_index(drop=True)
            elim_list = elim_map.get((season, int(week)), [])
            elim_k = len(elim_list)
            no_elim_week_flag = int(elim_k == 0)
            total_weeks += 1

            prev_pv = np.array([prev_pv_map.get(name, np.nan) for name in group["celebrity_name"].tolist()])
            if np.isnan(prev_pv).any():
                prev_pv = group["judge_percent"].to_numpy(dtype=float)
                prev_pv = prev_pv / prev_pv.sum() if prev_pv.sum() > 0 else np.ones(len(group)) / len(group)

            pv_hat, info = solve_week(
                group,
                prev_pv,
                elim_list,
                lambda_grid,
                rule_type=rule_type,
                logger=logger,
            )

            # Update prev map
            for idx, name in enumerate(group["celebrity_name"].tolist()):
                prev_pv_map[name] = pv_hat[idx]

            predicted_elim = predict_elimination(rule_type, group, pv_hat, elim_k)
            margin, margin_rule, approx_save = _margin_by_rule(rule_type, group, pv_hat, elim_k)

            constraint_rule_used = "percent_exact" if rule_type == "percent" else "percent_approx"
            convex_valid_flag = 1 if rule_type == "percent" else 0
            if convex_valid_flag == 0:
                excluded_weeks += 1
            convex_note = "Convex PV is percent-approx; invalid for rank/save seasons."

            # Elimination-based consistency only for elim_k > 0
            if elim_k > 0:
                elim_match = int(set(predicted_elim) == set(elim_list))
            else:
                elim_match = np.nan

            consistency_rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "n_remaining": len(group),
                    "rule_type": rule_type,
                    "constraint_rule_used": constraint_rule_used,
                    "approx_save": approx_save,
                    "convex_valid_flag": convex_valid_flag,
                    "convex_note": convex_note if convex_valid_flag == 0 else "percent-approx",
                    "eliminated_true": ";".join(elim_list) if elim_list else "",
                    "eliminated_pred": ";".join(predicted_elim),
                    "match": elim_match,
                    "margin": margin,
                    "margin_rule": margin_rule,
                    "lambda1": info.get("lambda1"),
                    "lambda2": info.get("lambda2"),
                    "solver_status": info.get("status"),
                    "solver_name": info.get("solver_name"),
                    "objective": info.get("objective"),
                    "total_slack": info.get("total_slack"),
                    "max_slack": info.get("max_slack"),
                    "infeasibility_flag": info.get("infeasibility_flag"),
                    "elim_k": elim_k,
                    "no_elim_week_flag": no_elim_week_flag,
                }
            )

            # ABC sampling
            feat_rows = feature_df[(feature_df["season"] == season) & (feature_df["celebrity_name"].isin(group["celebrity_name"]))]
            feat_rows = feat_rows.set_index("celebrity_name").reindex(group["celebrity_name"]).fillna(0)
            feat_matrix = feat_rows.drop(columns=["season"]).to_numpy(dtype=float)

            delta_s = season_effects.get(int(season), 0.0)
            eta_t = week_effects.get(int(week), 0.0)

            prev_u_center = np.array([prev_u_center_map.get(name, np.nan) for name in group["celebrity_name"].tolist()])
            prev_mu_center = np.array([prev_mu_center_map.get(name, np.nan) for name in group["celebrity_name"].tolist()])
            dev_prev = np.zeros(len(group), dtype=float)
            if len(prev_u_center) == len(group) and len(prev_mu_center) == len(group):
                missing = np.isnan(prev_u_center) | np.isnan(prev_mu_center)
                dev_prev = np.where(missing, 0.0, prev_u_center - prev_mu_center)

            samples, abc_info = abc_sample_week(
                group,
                elim_list,
                feat_matrix,
                rule_type,
                elim_k,
                delta_s,
                eta_t,
                prev_dev=dev_prev,
                phi_prior_a=phi_prior_a,
                phi_prior_b=phi_prior_b,
                min_accept=PHI_MIN_ACCEPT,
                seed=seed + int(season) * 100 + int(week),
                logger=logger,
            )
            phi_post_mean = abc_info.get("phi_post_mean")
            phi_post_median = abc_info.get("phi_post_median")
            phi_post_q025 = abc_info.get("phi_post_q025")
            phi_post_q975 = abc_info.get("phi_post_q975")
            phi_next_a = float(abc_info.get("phi_next_a", phi_prior_a))
            phi_next_b = float(abc_info.get("phi_next_b", phi_prior_b))
            phi_update_flag = int(abc_info.get("phi_update_flag", 0))

            phi_rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "phi_prior_a": phi_prior_a,
                    "phi_prior_b": phi_prior_b,
                    "phi_post_mean": phi_post_mean,
                    "phi_post_median": phi_post_median,
                    "phi_post_q025": phi_post_q025,
                    "phi_post_q975": phi_post_q975,
                    "accepted_n": int(abc_info.get("accepted", 0)),
                    "tried_n": int(abc_info.get("tries", 0)),
                }
            )

            if logger is not None:
                logger.info(
                    "Q1 ABC phi: season=%s week=%s prior=(%.4f,%.4f) post_mean=%.4f q025=%.4f q975=%.4f accepted=%s tries=%s update=%s next=(%.4f,%.4f)",
                    season,
                    int(week),
                    phi_prior_a,
                    phi_prior_b,
                    float(phi_post_mean) if phi_post_mean is not None else np.nan,
                    float(phi_post_q025) if phi_post_q025 is not None else np.nan,
                    float(phi_post_q975) if phi_post_q975 is not None else np.nan,
                    int(abc_info.get("accepted", 0)),
                    int(abc_info.get("tries", 0)),
                    phi_update_flag,
                    phi_next_a,
                    phi_next_b,
                )
                if int(abc_info.get("accepted", 0)) < PHI_MIN_ACCEPT:
                    logger.warning(
                        "Q1 ABC phi: insufficient accepted; keep prior (season=%s week=%s accepted=%s min_accept=%s)",
                        season,
                        int(week),
                        int(abc_info.get("accepted", 0)),
                        PHI_MIN_ACCEPT,
                    )
                elif not phi_update_flag:
                    logger.warning(
                        "Q1 ABC phi: posterior degenerate; keep prior (season=%s week=%s)",
                        season,
                        int(week),
                    )

            if phi_update_flag:
                phi_prior_a = phi_next_a
                phi_prior_b = phi_next_b
            else:
                phi_prior_a = phi_prior_a
                phi_prior_b = phi_prior_b

            u_center_post = abc_info.get("u_center_post")
            mu_center_post = abc_info.get("mu_center_post")
            if isinstance(u_center_post, np.ndarray) and isinstance(mu_center_post, np.ndarray):
                dev_center_post = u_center_post - mu_center_post
                for idx, name in enumerate(group["celebrity_name"].tolist()):
                    prev_u_center_map[name] = float(u_center_post[idx])
                    prev_mu_center_map[name] = float(mu_center_post[idx])
                    state_rows.append(
                        {
                            "season": season,
                            "week": int(week),
                            "celebrity_name": name,
                            "u_center_post": float(u_center_post[idx]),
                            "mu_center_post": float(mu_center_post[idx]),
                            "dev_center_post": float(dev_center_post[idx]),
                        }
                    )
            else:
                if logger is not None:
                    logger.warning(
                        "Q1 ABC state: missing u/mu center; keep previous state (season=%s week=%s)",
                        season,
                        int(week),
                    )

            # Vectorized expansion of posterior samples to long format
            M = samples.shape[0]
            n = len(group)
            sample_ids = np.repeat(np.arange(M, dtype=int), n)
            celeb_names = np.tile(group["celebrity_name"].to_numpy(), M)
            pv_flat = samples.reshape(-1)
            accepted_total = int(abc_info.get("accepted", 0))
            is_accepted = (sample_ids < accepted_total).astype(int)
            pv_samples_frames.append(
                pd.DataFrame(
                    {
                        "season": int(season),
                        "week": int(week),
                        "rule_type": rule_type,
                        "sample_id": sample_ids,
                        "celebrity_name": celeb_names,
                        "pv_sample": pv_flat,
                        "is_accepted": is_accepted,
                        "accepted_total": accepted_total,
                        "tries": int(abc_info.get("tries", 0)),
                        "dist_threshold": int(abc_info.get("dist_threshold", 0)),
                    }
                )
            )

            abc_rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "rule_type": abc_info.get("rule_type"),
                    "dist_threshold": abc_info.get("dist_threshold"),
                    "mean_dist": abc_info.get("mean_dist"),
                    "min_dist": abc_info.get("min_dist"),
                    "accepted": abc_info["accepted"],
                    "tries": abc_info["tries"],
                    "accept_rate": abc_info["accept_rate"],
                    "ess": abc_info["ess"],
                    "p_correct_exact": abc_info.get("p_correct_exact"),
                    "p_correct_within_threshold": abc_info.get("p_correct_within_threshold"),
                    "pred_mode_elim": abc_info.get("pred_mode_elim"),
                    "pred_mode_prob": abc_info.get("pred_mode_prob"),
                    "elim_k": elim_k,
                    "no_elim_week_flag": no_elim_week_flag,
                    "bayes_check_defined_flag": int(elim_k > 0),
                }
            )
            bayes_consistency_rows.append(
                {
                    "season": season,
                    "week": int(week),
                    "rule_type": rule_type,
                    "elim_k": elim_k,
                    "no_elim_week_flag": no_elim_week_flag,
                    "bayes_check_defined_flag": int(elim_k > 0),
                    "bayes_match_exact": (abc_info.get("p_correct_exact") if elim_k > 0 else np.nan),
                    "bayes_match_within": (abc_info.get("p_correct_within_threshold") if elim_k > 0 else np.nan),
                    "bayes_mode_prob": (abc_info.get("pred_mode_prob") if elim_k > 0 else np.nan),
                    "bayes_mode_elim": (abc_info.get("pred_mode_elim", "") if elim_k > 0 else ""),
                    "bayes_mean_dist": abc_info.get("mean_dist", np.nan),
                    "bayes_min_dist": abc_info.get("min_dist", np.nan),
                    "accepted": abc_info.get("accepted", np.nan),
                    "attempts": abc_info.get("tries", np.nan),
                }
            )

            elim_posterior = abc_info.get("elim_posterior", {})
            for name in group["celebrity_name"].tolist():
                elim_post_rows.append(
                    {
                        "season": season,
                        "week": int(week),
                        "celebrity_name": name,
                        "rule_type": rule_type,
                        "elim_k": elim_k,
                        "p_elim": float(elim_posterior.get(name, 0.0)),
                    }
                )

            # CI rows
            p2_5 = np.percentile(samples, 2.5, axis=0)
            p50 = np.percentile(samples, 50, axis=0)
            p97_5 = np.percentile(samples, 97.5, axis=0)
            pmean = np.mean(samples, axis=0)
            pmed = np.median(samples, axis=0)
            for idx, name in enumerate(group["celebrity_name"]):
                ci_rows.append(
                    {
                        "season": season,
                        "week": int(week),
                        "celebrity_name": name,
                        "pv_p2_5": p2_5[idx],
                        "pv_p50": p50[idx],
                        "pv_p97_5": p97_5[idx],
                        "accepted": abc_info["accepted"],
                        "rule_type": rule_type,
                    }
                )

            for idx, row in group.iterrows():
                pv_hat_out = pv_hat[idx] if convex_valid_flag == 1 else np.nan
                outputs.append(
                    {
                        "season": season,
                        "week": int(week),
                        "celebrity_name": row["celebrity_name"],
                        "rule_type": rule_type,
                        "judge_percent": row["judge_percent"],
                        "judge_rank": row["judge_rank"],
                        "judge_total": row["judge_total"],
                        "n_judges": row.get("n_judges"),
                        "n_remaining": row.get("n_remaining"),
                        "pv_hat": pv_hat_out,
                        "pv_post_mean": pmean[idx],
                        "pv_post_median": pmed[idx],
                        "convex_valid_flag": convex_valid_flag,
                        "convex_note": convex_note if convex_valid_flag == 0 else "percent-approx",
                        "elim_k": elim_k,
                        "no_elim_week_flag": no_elim_week_flag,
                    }
                )
                snapshot_rows.append(
                    {
                        "season": season,
                        "week": int(week),
                        "celebrity_name": row["celebrity_name"],
                        "rule_type": rule_type,
                        "judge_percent": row["judge_percent"],
                        "judge_rank": row["judge_rank"],
                        "judge_total": row["judge_total"],
                        "n_remaining": row.get("n_remaining"),
                        "pv_hat": pv_hat_out,
                        "pv_post_mean": pmean[idx],
                        "pv_post_median": pmed[idx],
                        "convex_valid_flag": convex_valid_flag,
                        "convex_note": convex_note if convex_valid_flag == 0 else "percent-approx",
                        "elim_k": elim_k,
                        "no_elim_week_flag": no_elim_week_flag,
                    }
                )

    outputs_df = pd.DataFrame(outputs)
    outputs_df["rv_hat"] = outputs_df.groupby(["season", "week"])["pv_hat"].rank(ascending=False, method="min")

    out_paths = {
        "q1_vote_point": str(Path(out_dir) / "q1_vote_point.csv"),
        "q1_vote_ci": str(Path(out_dir) / "q1_vote_ci.csv"),
        "q1_consistency_report": str(Path(out_dir) / "q1_consistency_report.csv"),
        "q1_abc_diagnostics": str(Path(out_dir) / "q1_abc_diagnostics.csv"),
        "q1_inputs_snapshot": str(Path(out_dir) / "q1_inputs_snapshot.csv"),
        "q1_elim_posterior": str(Path(out_dir) / "q1_elim_posterior.csv"),
        "q1_phi_posterior": str(Path(out_dir) / "q1_phi_posterior.csv"),
        "q1_state_evolution": str(Path(out_dir) / "q1_state_evolution.csv"),
        "q1_pv_posterior_samples": str(Path(out_dir) / "q1_pv_posterior_samples.parquet"),
    }

    write_csv(outputs_df, out_paths["q1_vote_point"])
    write_csv(pd.DataFrame(ci_rows), out_paths["q1_vote_ci"])
    consistency_df = pd.DataFrame(consistency_rows)
    excluded_no_elim_weeks_total = int(consistency_df["no_elim_week_flag"].sum()) if not consistency_df.empty else 0
    no_elim_week_rate = (excluded_no_elim_weeks_total / total_weeks) if total_weeks else np.nan
    consistency_eval_weeks = total_weeks - excluded_no_elim_weeks_total
    consistency_df["convex_excluded_weeks_total"] = excluded_weeks
    consistency_df["excluded_no_elim_weeks_total"] = excluded_no_elim_weeks_total
    consistency_df["no_elim_week_rate"] = no_elim_week_rate
    consistency_df["consistency_eval_weeks"] = consistency_eval_weeks
    write_csv(consistency_df, out_paths["q1_consistency_report"])
    abc_df = pd.DataFrame(abc_rows)
    write_csv(abc_df, out_paths["q1_abc_diagnostics"])
    write_csv(pd.DataFrame(snapshot_rows), out_paths["q1_inputs_snapshot"])
    write_csv(pd.DataFrame(elim_post_rows), out_paths["q1_elim_posterior"])
    write_csv(pd.DataFrame(phi_rows), out_paths["q1_phi_posterior"])
    if state_rows:
        write_csv(pd.DataFrame(state_rows), out_paths["q1_state_evolution"])

    # Bayes consistency report + summary (exclude no-elimination weeks)
    bayes_consistency_df = pd.DataFrame(bayes_consistency_rows)
    if not bayes_consistency_df.empty:
        write_csv(bayes_consistency_df, str(Path(out_dir) / "q1_bayes_consistency_report.csv"))

        total = bayes_consistency_df.groupby("season")["week"].count().rename("total_weeks")
        eval_weeks = bayes_consistency_df[bayes_consistency_df["elim_k"] > 0].groupby("season")["week"].count().rename("eval_weeks")
        excluded = bayes_consistency_df[bayes_consistency_df["elim_k"] == 0].groupby("season")["week"].count().rename("excluded_no_elim_weeks")
        summary = pd.concat([total, eval_weeks, excluded], axis=1).fillna(0)
        summary["no_elim_week_rate"] = summary["excluded_no_elim_weeks"] / summary["total_weeks"]

        eval_df = bayes_consistency_df[bayes_consistency_df["elim_k"] > 0]
        rates = eval_df.groupby("season")[["bayes_match_exact", "bayes_match_within", "bayes_mode_prob", "bayes_mean_dist"]].mean()
        rates = rates.rename(
            columns={
                "bayes_match_exact": "bayes_match_rate_exact",
                "bayes_match_within": "bayes_match_rate_within",
                "bayes_mode_prob": "bayes_mode_agreement",
                "bayes_mean_dist": "bayes_mean_dist_avg",
            }
        )
        summary = summary.merge(rates, left_index=True, right_index=True, how="left").reset_index()

        if "accepted" in eval_df.columns and "attempts" in eval_df.columns:
            eval_df = eval_df.copy()
            eval_df["bayes_accept_rate"] = np.where(eval_df["attempts"] > 0, eval_df["accepted"] / eval_df["attempts"], np.nan)
            acc_rates = eval_df.groupby("season")["bayes_accept_rate"].mean()
            summary = summary.merge(acc_rates, on="season", how="left")

        overall = {
            "season": "overall",
            "total_weeks": int(bayes_consistency_df["week"].count()),
            "eval_weeks": int(eval_df["week"].count()),
            "excluded_no_elim_weeks": int((bayes_consistency_df["elim_k"] == 0).sum()),
        }
        overall["no_elim_week_rate"] = (overall["excluded_no_elim_weeks"] / overall["total_weeks"]) if overall["total_weeks"] else np.nan
        overall["bayes_match_rate_exact"] = eval_df["bayes_match_exact"].mean() if not eval_df.empty else np.nan
        overall["bayes_match_rate_within"] = eval_df["bayes_match_within"].mean() if not eval_df.empty else np.nan
        overall["bayes_mode_agreement"] = eval_df["bayes_mode_prob"].mean() if not eval_df.empty else np.nan
        overall["bayes_mean_dist_avg"] = eval_df["bayes_mean_dist"].mean() if not eval_df.empty else np.nan
        overall["bayes_accept_rate"] = eval_df["bayes_accept_rate"].mean() if ("bayes_accept_rate" in eval_df.columns and not eval_df.empty) else np.nan
        summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

        write_csv(summary, str(Path(out_dir) / "q1_bayes_consistency_summary.csv"))
        logger.info(
            "Q1 Bayes consistency: excluded no-elimination weeks from Bayes consistency aggregation; no_elim_week_rate=%.3f",
            overall["no_elim_week_rate"],
        )

    # Convex scope note
    note_path = Path(out_dir) / "q1_convex_scope_note.txt"
    note_path.write_text(
        "Convex PV is a percent-approx inversion.\\n"
        "For rank/save seasons, convex pv_hat is set to NaN and convex_valid_flag=0.\\n"
        "Downstream analyses should use Bayesian PV (pv_post_mean/median).\\n",
        encoding="utf-8",
    )

    # No-elimination handling note
    no_elim_note_path = Path(out_dir) / "q1_no_elim_handling_note.txt"
    no_elim_note_path.write_text(
        "Bayesian PV computed for all weeks (including no-elimination weeks).\\n"
        "Elimination-based consistency checks computed only for weeks with elim_k>0.\\n"
        "no_elim_week_flag marks weeks without eliminations; such weeks excluded from elimination-based metrics.\\n",
        encoding="utf-8",
    )

    # Save posterior samples (parquet preferred, fallback to csv.gz)
    if pv_samples_frames:
        pv_samples_df = pd.concat(pv_samples_frames, ignore_index=True)
        parquet_path = Path(out_dir) / "q1_pv_posterior_samples.parquet"
        csv_path = Path(out_dir) / "q1_pv_posterior_samples.csv.gz"
        try:
            pv_samples_df.to_parquet(parquet_path, index=False)
            out_paths["q1_pv_posterior_samples"] = str(parquet_path)
        except Exception as e:
            if logger is not None:
                logger.warning("Parquet write failed (%s). Falling back to csv.gz.", e)
            pv_samples_df.to_csv(csv_path, index=False, compression="gzip")
            out_paths["q1_pv_posterior_samples"] = str(csv_path)
    else:
        if logger is not None:
            logger.warning("No posterior samples collected; skip q1_pv_posterior_samples output.")

    logger.info("Q1 outputs saved to %s", out_dir)
    return out_paths




