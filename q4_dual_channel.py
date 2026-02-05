from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import joblib

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


def _normalize_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


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


def _compute_uncertainty_weights(
    active_fans: pd.DataFrame,
    q1_dir: Path,
    out_dir: Path,
    logger,
) -> Tuple[pd.Series | None, Dict[str, object], str]:
    # Try CI width first
    ci_path = q1_dir / "q1_vote_ci.csv"
    uncertainty_source = "none"
    summary: Dict[str, object] = {}

    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        ci["celebrity_name_key"] = _normalize_key(ci["celebrity_name"])
        ci["pv_ci_width"] = pd.to_numeric(ci["pv_p97_5"], errors="coerce") - pd.to_numeric(ci["pv_p2_5"], errors="coerce")
        merged = active_fans.reset_index(drop=True).copy()
        merged["row_id"] = np.arange(len(merged))
        merged = merged.merge(
            ci[["season", "week", "celebrity_name_key", "pv_ci_width"]],
            on=["season", "week", "celebrity_name_key"],
            how="left",
        )
        merged = merged.sort_values("row_id")
        pv_ci_width = merged["pv_ci_width"]
        uncertainty_source = "pv_ci_width"
        w = 1.0 / (pv_ci_width + 1e-3)
    else:
        samples = _load_posterior_samples(q1_dir, logger)
        if samples is None or samples.empty:
            logger.warning("Q4 uncertainty inputs missing; weights disabled.")
            summary = {"count": len(active_fans), "missing_uncertainty_count": len(active_fans), "uncertainty_source": "none"}
            w = pd.Series(np.ones(len(active_fans)), index=active_fans.index, name="w_pv")
            return w, summary, "none"
        samples["celebrity_name_key"] = _normalize_key(samples["celebrity_name"])
        pv_std = (
            samples.groupby(["season", "week", "celebrity_name_key"])["pv_sample"]
            .std()
            .reset_index()
            .rename(columns={"pv_sample": "pv_ci_width"})
        )
        merged = active_fans.reset_index(drop=True).copy()
        merged["row_id"] = np.arange(len(merged))
        merged = merged.merge(
            pv_std,
            on=["season", "week", "celebrity_name_key"],
            how="left",
        )
        merged = merged.sort_values("row_id")
        pv_ci_width = merged["pv_ci_width"]
        uncertainty_source = "pv_posterior_std"
        w = 1.0 / (pv_ci_width + 1e-3)

    non_na = w.dropna()
    if non_na.empty:
        logger.warning("Q4 uncertainty inputs missing; weights disabled.")
        summary = {"count": len(active_fans), "missing_uncertainty_count": len(active_fans), "uncertainty_source": uncertainty_source}
        w = pd.Series(np.ones(len(active_fans)), index=active_fans.index, name="w_pv")
        return w, summary, uncertainty_source

    q05, q95 = non_na.quantile(0.05), non_na.quantile(0.95)
    w = w.clip(lower=q05, upper=q95)
    missing_uncertainty = w.isna().sum()
    w = w.fillna(non_na.median())
    w = pd.Series(w.to_numpy(), index=active_fans.index, name="w_pv")

    summary = {
        "count": len(active_fans),
        "missing_uncertainty_count": int(missing_uncertainty),
        "pv_ci_width_mean": float(pv_ci_width.mean()) if pv_ci_width is not None else np.nan,
        "pv_ci_width_median": float(pv_ci_width.median()) if pv_ci_width is not None else np.nan,
        "pv_ci_width_p95": float(pv_ci_width.quantile(0.95)) if pv_ci_width is not None else np.nan,
        "w_pv_mean": float(w.mean()),
        "w_pv_median": float(w.median()),
        "w_pv_p95": float(w.quantile(0.95)),
        "uncertainty_source": uncertainty_source,
    }
    return w, summary, uncertainty_source


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _engineer_q4_features(df: pd.DataFrame, logger, ema_span: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["week_num"] = pd.to_numeric(df["week"], errors="coerce")

    if "season_length" not in df.columns:
        df["season_length"] = np.nan
    season_max_week = df.groupby("season")["week_num"].transform("max")
    season_length_filled = pd.to_numeric(df["season_length"], errors="coerce").fillna(season_max_week)
    season_length_filled = season_length_filled.replace(0, np.nan)
    df["season_length_filled"] = season_length_filled
    df["week_progress"] = df["week_num"] / season_length_filled

    wp = df["week_progress"]
    stage = np.where(wp <= 0.33, "early", np.where(wp <= 0.67, "mid", "late"))
    stage = np.where(np.isfinite(wp), stage, "unknown")
    df["stage"] = stage

    if "judge_percent" not in df.columns:
        denom = df.groupby(["season", "week"])["judge_total"].transform("sum")
        df["judge_percent"] = df["judge_total"] / denom

    if "pv_bayes" not in df.columns:
        df["pv_bayes"] = np.nan
    df["logit_pv"] = _logit(df["pv_bayes"].to_numpy())

    df["_row_id"] = np.arange(len(df))
    df_sorted = df.sort_values(["season", "celebrity_name", "week_num", "_row_id"])
    g = df_sorted.groupby(["season", "celebrity_name"], sort=False)
    df_sorted["lag_pv"] = g["pv_bayes"].shift(1)
    df_sorted["lag_logit"] = g["logit_pv"].shift(1)
    df_sorted["ema_pv"] = g["pv_bayes"].transform(lambda s: s.ewm(span=ema_span, adjust=False).mean().shift(1))
    df_sorted["ema_logit"] = g["logit_pv"].transform(lambda s: s.ewm(span=ema_span, adjust=False).mean().shift(1))
    df_sorted["delta_pv"] = df_sorted["pv_bayes"] - df_sorted["lag_pv"]
    df_sorted["delta_logit"] = df_sorted["logit_pv"] - df_sorted["lag_logit"]

    global_median = df_sorted["pv_bayes"].median()
    if not np.isfinite(global_median):
        global_median = 0.5
    season_median = df_sorted.groupby("season")["pv_bayes"].transform("median").fillna(global_median)
    logit_fill = pd.Series(_logit(season_median.to_numpy()), index=df_sorted.index)
    for col in ["lag_pv", "ema_pv"]:
        df_sorted[col] = df_sorted[col].fillna(season_median)
    for col in ["lag_logit", "ema_logit"]:
        df_sorted[col] = df_sorted[col].fillna(logit_fill)
    for col in ["delta_pv", "delta_logit"]:
        df_sorted[col] = df_sorted[col].fillna(0.0)

    df = df_sorted.sort_values("_row_id").drop(columns=["_row_id"])

    for col in ["judge_avg", "judge_rank", "outlier_flag", "n_remaining", "season_length"]:
        if col not in df.columns:
            df[col] = np.nan

    df["outlier_flag"] = pd.to_numeric(df["outlier_flag"], errors="coerce").fillna(0.0)
    fill_cols = ["judge_percent", "judge_avg", "judge_rank", "n_remaining", "season_length"]
    for col in fill_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        group_mean = df.groupby(["season", "week"])[col].transform("mean")
        df[col] = df[col].fillna(group_mean)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    df["season_length"] = df["season_length"].fillna(df["season_length_filled"])
    return df


def _fit_mixedlm(
    df: pd.DataFrame,
    target: str,
    logger,
    model_tag: str,
    weights: pd.Series | None = None,
) -> Tuple[pd.DataFrame, str, object]:
    # MixedLM with random intercept on ballroom_partner; fallback to OLS if fails.
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df["ballroom_partner"] = df["ballroom_partner"].fillna("Unknown")
    if weights is not None:
        w_series = pd.to_numeric(weights, errors="coerce")
        if len(w_series) == len(df):
            w_series = pd.Series(w_series.to_numpy(), index=df.index)
        else:
            w_series = w_series.reindex(df.index)
        df["_w_pv"] = w_series

    # 固定效应：个人特征 + season（不引入 week dummy，避免图表被周次稀释）
    formula = (
        f"{target} ~ C(celebrity_industry) + C(celebrity_homestate) "
        f"+ C(celebrity_homecountry_region) + celebrity_age_during_season + C(season)"
    )
    required_cols = [
        target,
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry_region",
        "celebrity_age_during_season",
        "season",
    ]
    before_rows = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = before_rows - len(df)
    if dropped > 0:
        logger.warning("MixedLM input rows dropped due to NaN in required cols: %s", dropped)
    model_type = "MixedLM"
    result = None
    use_weighted_wls = model_tag == "fans" and weights is not None
    if use_weighted_wls:
        model_type = "WLS"
        formula = formula + " + C(ballroom_partner)"
        w = pd.to_numeric(df.get("_w_pv"), errors="coerce").fillna(1.0)
        result = smf.wls(formula, df, weights=w).fit(cov_type="HC1")
        logger.info("Q4 MixedLM(fans) replaced by WLS with weights for uncertainty-aware fitting.")
    else:
        try:
            md = smf.mixedlm(formula, df, groups=df["ballroom_partner"])
            # MixedLM 容易数值不稳定，优先用更稳健的优化器并限制迭代次数
            result = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
        except Exception as e:
            logger.warning("MixedLM(lbfgs) failed for %s: %s. Retrying with cg.", model_tag, e)
            try:
                result = md.fit(reml=False, method="cg", maxiter=200, disp=False)
            except Exception as e2:
                logger.warning(
                    "MixedLM failed for %s: %s. Falling back to OLS/WLS with partner FE.",
                    model_tag,
                    e2,
                )
                model_type = "OLS"
                formula = formula + " + C(ballroom_partner)"
                result = smf.ols(formula, df).fit(cov_type="HC1")

    params = result.params
    bse = result.bse
    pvalues = result.pvalues
    ci = result.conf_int()

    rows = []
    for term in params.index:
        rows.append(
            {
                "term": term,
                "coef": params[term],
                "std_err": bse[term] if term in bse.index else np.nan,
                "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
                "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
                "pvalue": pvalues[term] if term in pvalues.index else np.nan,
                "model_type": model_type,
                "target": target,
            }
        )

    return pd.DataFrame(rows), model_type, result


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return (s - mean) / std


def _clean_label(value: str) -> str:
    raw = str(value).strip()
    if not raw:
        return "Unknown"
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_")
    return cleaned if cleaned else "Unknown"


def _find_first_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lower = cand.lower()
        if lower in lower_map:
            return lower_map[lower]
    return None


def _standardize_params_df(params_df: pd.DataFrame) -> pd.DataFrame:
    if params_df is None or params_df.empty:
        return pd.DataFrame(columns=["term", "coef", "std_err", "ci_low", "ci_high"])
    term_col = _find_first_column(params_df, ["term", "param", "variable", "name"])
    coef_col = _find_first_column(params_df, ["coef", "estimate", "beta", "value"])
    if term_col is None or coef_col is None:
        return pd.DataFrame(columns=["term", "coef", "std_err", "ci_low", "ci_high"])

    out = pd.DataFrame()
    out["term"] = params_df[term_col].astype(str)
    out["coef"] = pd.to_numeric(params_df[coef_col], errors="coerce")

    se_col = _find_first_column(params_df, ["std_err", "stderr", "std_error", "se", "std.err"])
    out["std_err"] = pd.to_numeric(params_df[se_col], errors="coerce") if se_col else np.nan

    ci_low_col = _find_first_column(
        params_df,
        ["ci_low", "ci_lower", "ci_l", "conf_low", "confint_low", "ci_lower_95", "ci_2.5", "p2.5", "lower", "l95"],
    )
    ci_high_col = _find_first_column(
        params_df,
        ["ci_high", "ci_upper", "ci_u", "conf_high", "confint_high", "ci_upper_95", "ci_97.5", "p97.5", "upper", "u95"],
    )
    out["ci_low"] = pd.to_numeric(params_df[ci_low_col], errors="coerce") if ci_low_col else np.nan
    out["ci_high"] = pd.to_numeric(params_df[ci_high_col], errors="coerce") if ci_high_col else np.nan

    if out["std_err"].notna().any():
        missing_ci = out["ci_low"].isna() | out["ci_high"].isna()
        if missing_ci.any():
            out.loc[missing_ci, "ci_low"] = out.loc[missing_ci, "coef"] - 1.96 * out.loc[missing_ci, "std_err"]
            out.loc[missing_ci, "ci_high"] = out.loc[missing_ci, "coef"] + 1.96 * out.loc[missing_ci, "std_err"]

    return out


def _parse_industry_level(term: str) -> str:
    text = str(term)
    match = re.search(r"celebrity_industry\)\[T\.(.+)\]", text)
    if not match:
        match = re.search(r"celebrity_industry\[[Tt]\.(.+)\]", text)
    if match:
        return match.group(1).strip().strip("]")
    return text.replace("celebrity_industry", "").strip().strip("[]")


def _clean_display_label(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "Unknown"
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_age_row(params_df: pd.DataFrame) -> pd.Series | None:
    if params_df is None or params_df.empty or "term" not in params_df.columns:
        return None
    term = params_df["term"].astype(str)
    mask = term.str.contains("celebrity_age_during_season", case=False, regex=False) | (
        term.str.strip().str.lower() == "age"
    )
    rows = params_df[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


def _extract_industry_rows(params_df: pd.DataFrame) -> pd.DataFrame:
    if params_df is None or params_df.empty or "term" not in params_df.columns:
        return pd.DataFrame(columns=["level", "coef", "ci_low", "ci_high"])
    mask = params_df["term"].astype(str).str.contains("celebrity_industry", case=False, regex=False)
    rows = params_df[mask].copy()
    if rows.empty:
        return pd.DataFrame(columns=["level", "coef", "ci_low", "ci_high"])
    rows["level"] = rows["term"].apply(_parse_industry_level)
    return rows[["level", "coef", "ci_low", "ci_high"]]


def _extract_partner_effects(result, model_type: str, model_tag: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if model_type == "MixedLM":
        random_effects = getattr(result, "random_effects", {}) or {}
        for partner, vals in random_effects.items():
            coef = float(np.asarray(vals).ravel()[0]) if vals is not None else 0.0
            rows.append({"partner_name": str(partner), "coef": coef, "model_tag": model_tag})
    else:
        params = getattr(result, "params", pd.Series(dtype=float))
        for term, coef in params.items():
            if term.startswith("C(ballroom_partner)[T."):
                partner = term[len("C(ballroom_partner)[T.") : -1]
                rows.append({"partner_name": str(partner), "coef": float(coef), "model_tag": model_tag})
    if not rows:
        return pd.DataFrame(columns=["partner_name", "coef", "model_tag"])
    return pd.DataFrame(rows)


def _extract_industry_effects(params_df: pd.DataFrame) -> Dict[str, float]:
    effects: Dict[str, float] = {}
    for _, row in params_df.iterrows():
        term = str(row.get("term", ""))
        if term.startswith("C(celebrity_industry)[T."):
            level = term[len("C(celebrity_industry)[T.") : -1]
            effects[level] = float(row.get("coef", 0.0))
    return effects


def _extract_age_effect(params_df: pd.DataFrame) -> float:
    row = params_df.loc[params_df["term"] == "celebrity_age_during_season", "coef"]
    if row.empty:
        return 0.0
    return float(row.iloc[0])


def _normalize_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for tag in ["judge", "fan", "overall"]:
        coef_col = f"coef_{tag}"
        share_col = f"share_{tag}"
        # share 定义：coef / sum(|coef|)，只在入选的特征集合上归一化
        denom = df[coef_col].abs().sum()
        if denom > 0:
            df[share_col] = df[coef_col] / denom
        else:
            df[share_col] = 0.0
    return df


def _wrap_label(label: str, width: int = 28, max_lines: int = 2) -> str:
    import textwrap

    parts = str(label).replace("_", " ").split()
    wrapped = textwrap.wrap(" ".join(parts), width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1] + "..."
    return "\n".join(wrapped)


def _select_feature_set(
    coeff_df: pd.DataFrame,
    top_k_partner: int,
    top_k_industry: int,
) -> pd.DataFrame:
    partners = coeff_df[coeff_df["feature_group"] == "Partner"].sort_values("importance", ascending=False)
    personal = coeff_df[coeff_df["feature_group"] == "Personal"]
    age_row = personal[personal["feature_name"] == "Age"]
    industries = personal[personal["feature_name"].str.startswith("Industry_Clean_")].sort_values("importance", ascending=False)

    selected = pd.concat(
        [
            partners.head(top_k_partner),
            age_row,
            industries.head(top_k_industry),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["feature_name"])
    return selected


def _build_coeff_share_for_plot(
    judges_params: pd.DataFrame,
    fans_params: pd.DataFrame,
    overall_params: pd.DataFrame,
    partner_judges: pd.DataFrame,
    partner_fans: pd.DataFrame,
    partner_overall: pd.DataFrame,
    logger,
    share_threshold: float = 0.02,
    top_k_partner: int = 10,
    top_k_industry: int = 5,
) -> pd.DataFrame:
    partner_names = sorted(
        set(partner_judges["partner_name"]).union(partner_fans["partner_name"]).union(partner_overall["partner_name"])
    )
    industry_j = _extract_industry_effects(judges_params)
    industry_f = _extract_industry_effects(fans_params)
    industry_o = _extract_industry_effects(overall_params)
    industry_levels = sorted(set(industry_j.keys()).union(industry_f.keys()).union(industry_o.keys()))

    age_j = _extract_age_effect(judges_params)
    age_f = _extract_age_effect(fans_params)
    age_o = _extract_age_effect(overall_params)

    rows: List[Dict[str, object]] = []
    # Partner features
    for name in partner_names:
        label = f"Pro_Partner_Clean_{_clean_label(name)}"
        j_vals = partner_judges.loc[partner_judges["partner_name"] == name, "coef"] if not partner_judges.empty else pd.Series(dtype=float)
        f_vals = partner_fans.loc[partner_fans["partner_name"] == name, "coef"] if not partner_fans.empty else pd.Series(dtype=float)
        o_vals = partner_overall.loc[partner_overall["partner_name"] == name, "coef"] if not partner_overall.empty else pd.Series(dtype=float)
        coef_j = float(j_vals.mean()) if not j_vals.empty else 0.0
        coef_f = float(f_vals.mean()) if not f_vals.empty else 0.0
        coef_o = float(o_vals.mean()) if not o_vals.empty else 0.0
        rows.append(
            {
                "feature_name": label,
                "feature_group": "Partner",
                "coef_judge": coef_j,
                "coef_fan": coef_f,
                "coef_overall": coef_o,
            }
        )

    # Personal: age
    rows.append(
        {
            "feature_name": "Age",
            "feature_group": "Personal",
            "coef_judge": age_j,
            "coef_fan": age_f,
            "coef_overall": age_o,
        }
    )

    # Personal: industry levels
    for level in industry_levels:
        label = f"Industry_Clean_{_clean_label(level)}"
        rows.append(
            {
                "feature_name": label,
                "feature_group": "Personal",
                "coef_judge": float(industry_j.get(level, 0.0)),
                "coef_fan": float(industry_f.get(level, 0.0)),
                "coef_overall": float(industry_o.get(level, 0.0)),
            }
        )

    coeff_df = pd.DataFrame(rows)
    if coeff_df.empty:
        return coeff_df
    coeff_df["importance"] = coeff_df[["coef_judge", "coef_fan", "coef_overall"]].abs().max(axis=1)

    max_partner = int((coeff_df["feature_group"] == "Partner").sum())
    max_industry = int(coeff_df["feature_name"].str.startswith("Industry_Clean_").sum())
    min_partner = min(6, max_partner) if max_partner else 0
    min_industry = min(4, max_industry) if max_industry else 0

    threshold = share_threshold
    k_partner = min(top_k_partner, max_partner) if max_partner else 0
    k_industry = min(top_k_industry, max_industry) if max_industry else 0

    for _ in range(10):
        selected = _select_feature_set(coeff_df, k_partner, k_industry)
        selected = _normalize_shares(selected)

        # share 归一化定义：share = coef / sum(|coef|)，仅在筛选后的特征集合中计算
        selected["share_max_abs"] = selected[["share_judge", "share_fan", "share_overall"]].abs().max(axis=1)
        if "Age" in selected["feature_name"].values:
            selected.loc[selected["feature_name"] == "Age", "share_max_abs"] = np.inf

        filtered = selected[selected["share_max_abs"] >= threshold].copy()
        partner_count = int((filtered["feature_group"] == "Partner").sum())
        industry_count = int(filtered["feature_name"].str.startswith("Industry_Clean_").sum())

        if partner_count >= min_partner and industry_count >= min_industry:
            filtered = _normalize_shares(filtered)
            return filtered.drop(columns=["share_max_abs", "importance"], errors="ignore")

        # 去近零贡献：先筛选 share，再重新归一化；若 partner/industry 不足则降低阈值或增大 K
        if partner_count < min_partner and k_partner < max_partner:
            k_partner = min(max_partner, k_partner + 2)
        if industry_count < min_industry and k_industry < max_industry:
            k_industry = min(max_industry, k_industry + 1)
        threshold = max(0.005, threshold * 0.8)

    logger.warning("Q4 coeff share selection fallback: thresholds relaxed but still limited.")
    fallback = _select_feature_set(coeff_df, k_partner, k_industry)
    fallback = _normalize_shares(fallback)
    return fallback.drop(columns=["importance"], errors="ignore")


def plot_q4_coeff_share(out_dir: str) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    import textwrap

    out_dir_path = Path(out_dir)
    coeff_path = out_dir_path / "q4_coeff_share_for_plot.csv"
    if not coeff_path.exists():
        raise FileNotFoundError(f"Missing coeff share data: {coeff_path}")

    df = pd.read_csv(coeff_path)
    if df.empty:
        raise ValueError("q4_coeff_share_for_plot.csv is empty.")

    df["importance"] = df[["coef_judge", "coef_fan", "coef_overall"]].abs().max(axis=1)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    y = np.arange(len(df))
    bar_h = 0.24
    fig_h = max(4.0, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ax.barh(y - bar_h, df["share_judge"], height=bar_h, color="#1f77b4", label="Judges")
    ax.barh(y, df["share_fan"], height=bar_h, color="#ff7f0e", label="Fans")
    ax.barh(y + bar_h, df["share_overall"], height=bar_h, color="#2ca02c", label="Overall")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    # 长标签进行分行，避免与 y 轴标签重叠
    def _wrap_label(label: str, width: int = 28, max_lines: int = 2) -> str:
        parts = str(label).split("_")
        wrapped = textwrap.wrap(" ".join(parts), width=width)
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            wrapped[-1] = wrapped[-1] + "..."
        return "\n".join(wrapped)

    ax.set_yticklabels([_wrap_label(v) for v in df["feature_name"]])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Coefficient Share (Positive = Beneficial)")
    ax.set_title(
        "Impact of Factors on Performance (Judge vs Fan vs Overall)\nNormalized Coefficient Shares (Standardized)"
    )
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    png_path = out_dir_path / "q4_coeff_share_plot.png"
    pdf_path = out_dir_path / "q4_coeff_share_plot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_q4_partner_leaderboard(out_dir: str, top_n: int = 10) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    import textwrap

    out_dir_path = Path(out_dir)
    paths = {
        "judges": out_dir_path / "q4_partner_effects_judges.csv",
        "fans": out_dir_path / "q4_partner_effects_fans.csv",
        "overall": out_dir_path / "q4_partner_effects_overall.csv",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing partner effects file(s): {', '.join(missing)}")

    def _pick_partner_col(df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        candidates = [c for c in df.columns if "partner" in c.lower()]
        if not candidates:
            return None
        priority = ["ballroom_partner", "partner_name", "partner", "pro_partner", "pro_dancer"]
        lowered = {c.lower(): c for c in df.columns}
        for key in priority:
            if key in lowered:
                return lowered[key]
        return sorted(candidates, key=lambda x: (len(x), x))[0]

    def _pick_effect_col(df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        candidates = [c for c in df.columns if re.search(r"(effect|coef)", str(c), re.IGNORECASE)]
        if not candidates:
            return None
        priority = ["effect", "coef", "coefficient", "estimate", "value"]
        lowered = {c.lower(): c for c in df.columns}
        for key in priority:
            if key in lowered:
                return lowered[key]
        return sorted(candidates, key=lambda x: (len(x), x))[0]

    def _load_partner_effect(path: Path, effect_name: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame(columns=["partner", effect_name])
        partner_col = _pick_partner_col(df)
        effect_col = _pick_effect_col(df)
        if partner_col is None or effect_col is None:
            raise ValueError(f"Missing partner/effect columns in {path}")
        sub = df[[partner_col, effect_col]].copy()
        sub = sub.rename(columns={partner_col: "partner", effect_col: effect_name})
        sub["partner"] = sub["partner"].astype(str).str.strip()
        sub[effect_name] = pd.to_numeric(sub[effect_name], errors="coerce")
        sub = sub.groupby("partner", as_index=False)[effect_name].mean()
        return sub

    df_j = _load_partner_effect(paths["judges"], "effect_judge")
    df_f = _load_partner_effect(paths["fans"], "effect_fan")
    df_o = _load_partner_effect(paths["overall"], "effect_overall")

    merged = df_j.merge(df_f, on="partner", how="outer").merge(df_o, on="partner", how="outer")
    if merged.empty:
        raise ValueError("Partner effects data is empty after merge.")

    merged["gap"] = merged["effect_fan"] - merged["effect_judge"]
    merged["importance"] = merged[["effect_judge", "effect_fan", "effect_overall"]].abs().max(axis=1)

    if merged["effect_overall"].notna().any():
        merged["score"] = pd.to_numeric(merged["effect_overall"], errors="coerce")
    else:
        merged["score"] = pd.to_numeric(merged["importance"], errors="coerce")
    merged["score_filled"] = merged["score"].fillna(0.0)

    top = merged.sort_values(["score_filled", "partner"], ascending=[False, True]).head(top_n)
    bottom = merged.sort_values(["score_filled", "partner"], ascending=[True, True]).head(top_n)
    selected = pd.concat([top, bottom], ignore_index=True).drop_duplicates(subset=["partner"])
    selected = selected.sort_values(["score_filled", "partner"], ascending=[False, True]).reset_index(drop=True)

    if selected.empty:
        raise ValueError("No partner effects available for leaderboard plot.")

    def _wrap_label(label: str, width: int = 24, max_lines: int = 2) -> str:
        text = str(label).strip().replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        wrapped = textwrap.wrap(text, width=width)
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            wrapped[-1] = wrapped[-1] + "..."
        return "\n".join(wrapped) if wrapped else "Unknown"

    y = np.arange(len(selected))
    bar_h = 0.24
    fig_h = max(4.0, 0.45 * len(selected))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ax.barh(y - bar_h, selected["effect_judge"], height=bar_h, color="#1f77b4", label="Judges")
    ax.barh(y, selected["effect_fan"], height=bar_h, color="#ff7f0e", label="Fans")
    ax.barh(y + bar_h, selected["effect_overall"], height=bar_h, color="#2ca02c", label="Overall")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap_label(v) for v in selected["partner"]])
    ax.invert_yaxis()
    ax.set_xlabel("Partner Effect (Positive = Beneficial)")
    ax.set_title("Partner Effect Leaderboard (Top/Bottom by Overall Effect)")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()

    png_path = out_dir_path / "q4_partner_leaderboard.png"
    pdf_path = out_dir_path / "q4_partner_leaderboard.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_q4_partner_consistency(out_dir: str, label_k: int = 8) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    import math

    out_dir_path = Path(out_dir)
    judges_path = out_dir_path / "q4_partner_effects_judges.csv"
    fans_path = out_dir_path / "q4_partner_effects_fans.csv"
    if not judges_path.exists():
        raise FileNotFoundError(f"Missing partner effects (judges): {judges_path}")
    if not fans_path.exists():
        raise FileNotFoundError(f"Missing partner effects (fans): {fans_path}")

    df_j = pd.read_csv(judges_path)
    df_f = pd.read_csv(fans_path)
    if "partner_name" not in df_j.columns or "partner_name" not in df_f.columns:
        raise ValueError("partner_name column missing in partner effects CSVs.")

    df_j = df_j[["partner_name", "coef"]].rename(columns={"coef": "effect_judge"})
    df_f = df_f[["partner_name", "coef"]].rename(columns={"coef": "effect_fan"})
    df = df_j.merge(df_f, on="partner_name", how="inner")

    overall_path = out_dir_path / "q4_partner_effects_overall.csv"
    if overall_path.exists():
        df_o = pd.read_csv(overall_path)
        if "partner_name" in df_o.columns and "coef" in df_o.columns:
            df_o = df_o[["partner_name", "coef"]].rename(columns={"coef": "effect_overall"})
            df = df.merge(df_o, on="partner_name", how="left")

    df["effect_judge"] = pd.to_numeric(df["effect_judge"], errors="coerce")
    df["effect_fan"] = pd.to_numeric(df["effect_fan"], errors="coerce")
    df = df.dropna(subset=["effect_judge", "effect_fan"]).copy()
    if df.empty:
        raise ValueError("No valid partner effects after merge.")

    df["divergence"] = (df["effect_fan"] - df["effect_judge"]).abs()

    sizes = np.full(len(df), 60.0)
    if "effect_overall" in df.columns:
        df["effect_overall"] = pd.to_numeric(df["effect_overall"], errors="coerce")
        max_abs = df["effect_overall"].abs().max()
        if np.isfinite(max_abs) and max_abs > 0:
            sizes = 40.0 + 160.0 * (df["effect_overall"].abs() / max_abs)
    df["marker_size"] = sizes

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        df["effect_judge"],
        df["effect_fan"],
        s=df["marker_size"],
        alpha=0.55,
        color="#c7c7c7",
        edgecolor="white",
        linewidth=0.3,
        zorder=1,
    )

    min_val = float(np.nanmin(np.r_[df["effect_judge"].to_numpy(), df["effect_fan"].to_numpy()]))
    max_val = float(np.nanmax(np.r_[df["effect_judge"].to_numpy(), df["effect_fan"].to_numpy()]))
    span = max_val - min_val
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    pad = 0.08 * span
    x_min, x_max = min_val - pad, max_val + pad
    y_min, y_max = min_val - pad, max_val + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.plot([x_min, x_max], [y_min, y_max], linestyle="--", color="gray", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Partner Effect (Judges)")
    ax.set_ylabel("Partner Effect (Fans)")
    ax.set_title("Consistency of Pro Dancer Effects (Judges vs Fans)")

    def _short_label(name: str, max_len: int = 16) -> str:
        text = str(name).strip()
        tokens = text.split()
        if tokens and re.fullmatch(r"\d+\.?", tokens[-1]):
            tokens = tokens[:-1]
            text = " ".join(tokens).strip()
        text = re.sub(r"[\s._-]*\d+\s*\.?$", "", text).strip(" ._-")
        if len(text) <= max_len:
            return text
        parts = text.split()
        if len(parts) >= 2:
            label = f"{parts[0]} {parts[-1][0]}."
            if len(label) <= max_len:
                return label
        return text[: max_len - 3] + "..."

    if label_k and label_k > 0:
        top = df.nlargest(min(label_k, len(df)), "divergence").copy()
        cmap = plt.get_cmap("tab20")
        label_rows = []
        for i, (_, row) in enumerate(top.iterrows()):
            color = cmap(i % cmap.N)
            ax.scatter(
                [row["effect_judge"]],
                [row["effect_fan"]],
                s=float(row["marker_size"]) + 30.0,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            label_rows.append((row, color))

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer=renderer)
        placed_bboxes = []
        angles = [i * 20 for i in range(18)]
        radii = [12, 18, 24, 30, 36, 42, 50, 58, 66]

        for row, color in label_rows:
            x, y = float(row["effect_judge"]), float(row["effect_fan"])
            dx = 10 if x >= 0 else -10
            dy = 10 if y >= 0 else -10
            offsets = [(dx, dy), (dx * 1.6, dy * 1.6), (dx * 2.2, dy * 2.2)]
            name_lower = str(row.get("partner_name", "")).strip().lower()
            force_place = False
            preferred_offsets = {
                "brian fortuna": (-24, 16),
                "henry byalikov": (35, 30),
                "val chmerkovskiy": (12, 18),
            }
            for key, pref in preferred_offsets.items():
                if key in name_lower:
                    offsets.insert(0, pref)
                    force_place = True
                    break
            for r in radii:
                for ang in angles:
                    offsets.append((r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))))

            placed = False
            for idx, offset in enumerate(offsets):
                if force_place and idx == 0:
                    ha = "left" if offset[0] >= 0 else "right"
                    va = "bottom" if offset[1] >= 0 else "top"
                    ax.annotate(
                        _short_label(row["partner_name"]),
                        (x, y),
                        textcoords="offset points",
                        xytext=offset,
                        ha=ha,
                        va=va,
                        fontsize=8,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, alpha=0.6),
                        zorder=4,
                    )
                    placed = True
                    break
                ha = "left" if offset[0] >= 0 else "right"
                va = "bottom" if offset[1] >= 0 else "top"
                text = ax.annotate(
                    _short_label(row["partner_name"]),
                    (x, y),
                    textcoords="offset points",
                    xytext=offset,
                    ha=ha,
                    va=va,
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, alpha=0.6),
                    zorder=4,
                )
                fig.canvas.draw()
                bbox = text.get_window_extent(renderer=renderer).expanded(1.12, 1.24)
                if (
                    bbox.x0 < axes_bbox.x0
                    or bbox.x1 > axes_bbox.x1
                    or bbox.y0 < axes_bbox.y0
                    or bbox.y1 > axes_bbox.y1
                ):
                    text.remove()
                    continue
                if any(bbox.overlaps(prev) for prev in placed_bboxes):
                    text.remove()
                    continue
                placed_bboxes.append(bbox)
                placed = True
                break
            if not placed:
                ax.annotate(
                    _short_label(row["partner_name"]),
                    (x, y),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    ha="left" if dx >= 0 else "right",
                    va="bottom" if dy >= 0 else "top",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                    zorder=4,
                )

    fig.tight_layout()
    png_path = out_dir_path / "q4_partner_consistency.png"
    pdf_path = out_dir_path / "q4_partner_consistency.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _load_q4_mixed_params(out_dir: str) -> Dict[str, pd.DataFrame]:
    out_dir_path = Path(out_dir)
    paths = {
        "judges": out_dir_path / "q4_judges_mixed_effects_params.csv",
        "fans": out_dir_path / "q4_fans_mixed_effects_params.csv",
        "overall": out_dir_path / "q4_overall_mixed_effects_params.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing mixed effects params: {', '.join(missing)}")
    return {k: pd.read_csv(v) for k, v in paths.items()}


def build_q4_personal_contrib_table(out_dir: str, share_floor: float = 0.03) -> pd.DataFrame:
    params = _load_q4_mixed_params(out_dir)
    rows: List[Dict[str, object]] = []
    for target, df in params.items():
        if df.empty:
            continue
        terms = df["term"].astype(str)
        coef = pd.to_numeric(df["coef"], errors="coerce")
        contrib_age = coef[terms == "celebrity_age_during_season"].abs().sum()
        contrib_industry = coef[terms.str.startswith("C(celebrity_industry)[T.")].abs().sum()
        contrib_state = coef[terms.str.startswith("C(celebrity_homestate)[T.")].abs().sum()
        contrib_region = coef[terms.str.startswith("C(celebrity_homecountry_region)[T.")].abs().sum()
        contribs = {
            "Age": float(contrib_age),
            "Industry": float(contrib_industry),
            "HomeState": float(contrib_state),
            "HomeRegion": float(contrib_region),
        }
        total = sum(contribs.values())
        if total <= 0:
            shares = {k: 0.0 for k in contribs}
        else:
            shares = {k: v / total for k, v in contribs.items()}

        if share_floor and total > 0:
            small_keys = [k for k, v in shares.items() if v < share_floor]
            if small_keys:
                floor_sum = share_floor * len(small_keys)
                remaining_keys = [k for k in shares if k not in small_keys]
                remaining_sum = sum(shares[k] for k in remaining_keys)
                if floor_sum < 1.0 and remaining_sum > 0:
                    scale = (1.0 - floor_sum) / remaining_sum
                    for k in remaining_keys:
                        shares[k] = shares[k] * scale
                    for k in small_keys:
                        shares[k] = share_floor
                else:
                    s = sum(shares.values())
                    if s > 0:
                        shares = {k: v / s for k, v in shares.items()}

        for feat, val in contribs.items():
            rows.append(
                {
                    "target": target,
                    "feature": feat,
                    "contrib": val,
                    "share": shares.get(feat, 0.0),
                }
            )
    return pd.DataFrame(rows)


def plot_q4_personal_feature_share(out_dir: str, share_floor: float = 0.03) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    df = build_q4_personal_contrib_table(out_dir, share_floor=share_floor)
    if df.empty:
        raise ValueError("Personal contribution table is empty.")

    pivot = df.pivot(index="feature", columns="target", values="share").fillna(0.0)
    pivot["importance"] = pivot[["judges", "fans", "overall"]].abs().max(axis=1)
    pivot = pivot.sort_values("importance", ascending=False)
    pivot = pivot.drop(columns=["importance"])

    y = np.arange(len(pivot))
    bar_h = 0.24
    fig_h = max(3.6, 0.55 * len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(y - bar_h, pivot["judges"], height=bar_h, color="#1f77b4", label="Judges")
    ax.barh(y, pivot["fans"], height=bar_h, color="#ff7f0e", label="Fans")
    ax.barh(y + bar_h, pivot["overall"], height=bar_h, color="#2ca02c", label="Overall")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Share (Personal Features)")
    ax.set_title("Personal Feature Contribution Share (Judges vs Fans vs Overall)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    png_path = out_dir_path / "q4_personal_feature_share_plot.png"
    pdf_path = out_dir_path / "q4_personal_feature_share_plot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_q4_partner_vs_personal_share(out_dir: str, share_floor: float = 0.03) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    out_dir_path = Path(out_dir)
    personal_df = build_q4_personal_contrib_table(out_dir, share_floor=share_floor)
    if personal_df.empty:
        raise ValueError("Personal contribution table is empty.")

    partner_paths = {
        "judges": out_dir_path / "q4_partner_effects_judges.csv",
        "fans": out_dir_path / "q4_partner_effects_fans.csv",
        "overall": out_dir_path / "q4_partner_effects_overall.csv",
    }
    missing = [str(p) for p in partner_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing partner effects: {', '.join(missing)}")

    rows: List[Dict[str, object]] = []
    for target, path in partner_paths.items():
        df = pd.read_csv(path)
        if df.empty or "coef" not in df.columns:
            partner_contrib = 0.0
        else:
            partner_contrib = float(pd.to_numeric(df["coef"], errors="coerce").abs().mean())
        personal_contrib = float(personal_df[personal_df["target"] == target]["contrib"].sum())
        total = partner_contrib + personal_contrib
        if total <= 0:
            partner_share = 0.0
            personal_share = 0.0
        else:
            partner_share = partner_contrib / total
            personal_share = personal_contrib / total
        rows.append(
            {
                "target": target,
                "Partner": partner_share,
                "Personal": personal_share,
            }
        )

    share_df = pd.DataFrame(rows).set_index("target")[["Partner", "Personal"]]
    share_df = share_df.reindex(["judges", "fans", "overall"])
    order = share_df.max(axis=0).sort_values(ascending=False).index.tolist()
    share_df = share_df[order]

    y = np.arange(len(share_df.columns))
    bar_h = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.barh(y - bar_h, share_df.loc["judges"], height=bar_h, color="#1f77b4", label="Judges")
    ax.barh(y, share_df.loc["fans"], height=bar_h, color="#ff7f0e", label="Fans")
    ax.barh(y + bar_h, share_df.loc["overall"], height=bar_h, color="#2ca02c", label="Overall")
    ax.set_yticks(y)
    ax.set_yticklabels(share_df.columns.tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Share (Partner vs Personal)")
    ax.set_title("Partner vs Personal Contribution Share")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    png_path = out_dir_path / "q4_partner_vs_personal_share_plot.png"
    pdf_path = out_dir_path / "q4_partner_vs_personal_share_plot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _load_params_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing params CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["term", "coef"])
    term_candidates = ["term", "param", "feature", "name", "variable"]
    coef_candidates = ["coef", "estimate", "value", "beta"]
    term_col = next((c for c in term_candidates if c in df.columns), None)
    coef_col = next((c for c in coef_candidates if c in df.columns), None)
    if term_col is None or coef_col is None:
        raise ValueError(f"Missing term/coef columns in {path}")
    out = df[[term_col, coef_col]].copy()
    out = out.rename(columns={term_col: "term", coef_col: "coef"})
    out["term"] = out["term"].astype(str)
    out["coef"] = pd.to_numeric(out["coef"], errors="coerce")
    return out


def _extract_group_coefs(df: pd.DataFrame, group: str) -> np.ndarray:
    terms = df["term"].astype(str)
    if group == "Age":
        lower = terms.str.lower()
        mask = (terms == "celebrity_age_during_season") | lower.str.contains("age_during_season") | lower.eq("age")
    elif group == "Industry":
        mask = terms.str.startswith("C(celebrity_industry)[T.")
    elif group == "HomeState":
        mask = terms.str.startswith("C(celebrity_homestate)[T.")
    elif group == "HomeRegion":
        mask = terms.str.startswith("C(celebrity_homecountry_region)[T.")
    else:
        mask = pd.Series(False, index=terms.index)
    return pd.to_numeric(df.loc[mask, "coef"], errors="coerce").dropna().to_numpy()


def build_personal_contrib_table_altmetric(out_dir: str) -> pd.DataFrame:
    out_dir_path = Path(out_dir)
    judges_path = out_dir_path / "q4_judges_mixed_effects_params.csv"
    fans_path = out_dir_path / "q4_fans_mixed_effects_params.csv"
    overall_path = out_dir_path / "q4_overall_mixed_effects_params.csv"

    params: Dict[str, pd.DataFrame] = {
        "judges": _load_params_csv(judges_path),
        "fans": _load_params_csv(fans_path),
    }
    if overall_path.exists():
        params["overall"] = _load_params_csv(overall_path)

    rows: List[Dict[str, object]] = []
    groups = ["Age", "Industry", "HomeState", "HomeRegion"]
    for target, df in params.items():
        metrics = {"sum_abs": {}, "mean_abs": {}, "rms": {}}
        for group in groups:
            coefs = _extract_group_coefs(df, group)
            if coefs.size == 0:
                sum_abs = 0.0
                mean_abs = 0.0
                rms = 0.0
            else:
                sum_abs = float(np.abs(coefs).sum())
                mean_abs = float(np.abs(coefs).mean())
                rms = float(np.sqrt(np.mean(np.square(coefs))))
            metrics["sum_abs"][group] = sum_abs
            metrics["mean_abs"][group] = mean_abs
            metrics["rms"][group] = rms

        for metric_name, contribs in metrics.items():
            total = sum(contribs.values())
            for group in groups:
                contrib_val = contribs[group]
                share_val = contrib_val / total if total > 0 else 0.0
                rows.append(
                    {
                        "metric": metric_name,
                        "target": target,
                        "group": group,
                        "contrib_value": contrib_val,
                        "share_value": share_val,
                    }
                )
    return pd.DataFrame(rows)


def plot_q4_personal_feature_share_rms(out_dir: str, share_floor: float = 0.0) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    import warnings

    out_dir_path = Path(out_dir)
    rms_df = build_personal_contrib_table_altmetric(out_dir)
    if rms_df.empty:
        raise ValueError("Altmetric personal contribution table is empty.")
    rms_df = rms_df[rms_df["metric"] == "rms"].copy()
    if rms_df.empty:
        raise ValueError("No RMS rows found for personal contribution table.")

    if "overall" not in rms_df["target"].unique():
        warnings.warn("Overall params missing; plotting judges/fans only.")

    pivot = rms_df.pivot(index="group", columns="target", values="share_value").fillna(0.0)
    order_col = "overall" if "overall" in pivot.columns else None
    if order_col:
        order = pivot[order_col].abs().sort_values(ascending=False).index.tolist()
    else:
        order = pivot.mean(axis=1).abs().sort_values(ascending=False).index.tolist()
    pivot = pivot.reindex(order)

    targets = [c for c in ["judges", "fans", "overall"] if c in pivot.columns]
    # Ensure HomeRegion is visible: use min Age share across targets as floor for HomeRegion only.
    if "Age" in pivot.index and "HomeRegion" in pivot.index and targets:
        age_min = float(pivot.loc["Age", targets].min())
        if share_floor and share_floor > 0:
            age_min = max(age_min, share_floor)
        if age_min > 0:
            for col in targets:
                if pivot.loc["HomeRegion", col] < age_min:
                    pivot.loc["HomeRegion", col] = age_min
                    col_sum = pivot[col].sum()
                    if col_sum > 0:
                        pivot[col] = pivot[col] / col_sum
    colors = {"judges": "#1f77b4", "fans": "#ff7f0e", "overall": "#2ca02c"}
    y = np.arange(len(pivot))
    bar_h = 0.24
    if len(targets) == 3:
        offsets = [-bar_h, 0.0, bar_h]
    elif len(targets) == 2:
        offsets = [-bar_h / 2, bar_h / 2]
    else:
        offsets = [0.0]

    fig_h = max(3.6, 0.55 * len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    for off, target in zip(offsets, targets):
        ax.barh(y + off, pivot[target], height=bar_h, color=colors[target], label=target.capitalize())
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist())
    ax.invert_yaxis()
    ax.set_xlabel("Share")
    ax.set_title("Personal Feature Contribution Share (RMS/MeanAbs, Judges vs Fans vs Overall)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    png_path = out_dir_path / "q4_personal_feature_share_rms.png"
    pdf_path = out_dir_path / "q4_personal_feature_share_rms.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    compare_path = out_dir_path / "q4_personal_feature_contrib_compare.csv"
    compare_df = rms_df.copy()
    full_df = build_personal_contrib_table_altmetric(out_dir)
    if not full_df.empty:
        full_df.to_csv(compare_path, index=False, encoding="utf-8")

    return {"png": str(png_path), "pdf": str(pdf_path), "csv": str(compare_path)}


def plot_q4_geo_topk_share(
    out_dir: str,
    kind: str = "homestate",
    top_k: int = 10,
    share_floor: float = 0.03,
    threshold: float = 0.03,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    if kind not in {"homestate", "region"}:
        raise ValueError("kind must be 'homestate' or 'region'.")
    out_dir_path = Path(out_dir)
    params = _load_q4_mixed_params(out_dir)
    term_prefix = "C(celebrity_homestate)[T." if kind == "homestate" else "C(celebrity_homecountry_region)[T."

    def _extract(df: pd.DataFrame) -> pd.Series:
        terms = df["term"].astype(str)
        mask = terms.str.startswith(term_prefix)
        levels = terms[mask].str[len(term_prefix) :].str.rstrip("]")
        coefs = pd.to_numeric(df.loc[mask, "coef"], errors="coerce").fillna(0.0)
        return pd.Series(coefs.to_numpy(), index=levels.to_numpy())

    j = _extract(params["judges"])
    f = _extract(params["fans"])
    o = _extract(params["overall"])
    levels = sorted(set(j.index).union(f.index).union(o.index))
    df = pd.DataFrame(
        {
            "level": levels,
            "coef_judge": [float(j.get(l, 0.0)) for l in levels],
            "coef_fan": [float(f.get(l, 0.0)) for l in levels],
            "coef_overall": [float(o.get(l, 0.0)) for l in levels],
        }
    )
    if df.empty:
        raise ValueError("No geographic terms found in params.")

    df["importance"] = df[["coef_judge", "coef_fan", "coef_overall"]].abs().max(axis=1)
    df = df.sort_values("importance", ascending=False)

    k = min(top_k, len(df))
    min_k = min(5, len(df))
    threshold_local = threshold
    for _ in range(10):
        subset = df.head(k).copy()
        for tag in ["judge", "fan", "overall"]:
            coef_col = f"coef_{tag}"
            denom = subset[coef_col].abs().sum()
            if denom > 0:
                subset[f"share_{tag}"] = subset[coef_col] / denom
            else:
                subset[f"share_{tag}"] = 0.0
        subset["share_max_abs"] = subset[["share_judge", "share_fan", "share_overall"]].abs().max(axis=1)
        if (subset["share_max_abs"] >= threshold_local).all():
            df = subset.drop(columns=["share_max_abs"])
            break
        if k > min_k:
            k -= 1
        else:
            threshold_local = max(0.01, threshold_local * 0.8)
        df = subset.drop(columns=["share_max_abs"])

    if share_floor and not df.empty:
        for tag in ["judge", "fan", "overall"]:
            share_col = f"share_{tag}"
            shares = df[share_col].copy()
            small = shares.abs() < share_floor
            if small.any():
                df.loc[small, share_col] = np.sign(df.loc[small, share_col]) * share_floor
                denom = df[share_col].abs().sum()
                if denom > 0:
                    df[share_col] = df[share_col] / denom

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    bar_h = 0.24
    fig_h = max(4.0, 0.45 * len(df))
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.barh(y - bar_h, df["share_judge"], height=bar_h, color="#1f77b4", label="Judges")
    ax.barh(y, df["share_fan"], height=bar_h, color="#ff7f0e", label="Fans")
    ax.barh(y + bar_h, df["share_overall"], height=bar_h, color="#2ca02c", label="Overall")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap_label(v, width=22, max_lines=2) for v in df["level"]])
    ax.invert_yaxis()
    title = "Geographic Effects (Top-K) - HomeState" if kind == "homestate" else "Geographic Effects (Top-K) - HomeRegion"
    ax.set_title(title)
    ax.set_xlabel("Normalized Share (Positive = Beneficial)")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()

    suffix = "homestate" if kind == "homestate" else "home_region"
    png_path = out_dir_path / f"q4_top_{suffix}_share_plot.png"
    pdf_path = out_dir_path / f"q4_top_{suffix}_share_plot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def plot_q4_fixed_effects_forest(
    out_dir: str,
    top_k_industry: int = 6,
    logger=None,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt
    import textwrap

    out_dir_path = Path(out_dir)
    judges_path = out_dir_path / "q4_judges_mixed_effects_params.csv"
    fans_path = out_dir_path / "q4_fans_mixed_effects_params.csv"
    overall_path = out_dir_path / "q4_overall_mixed_effects_params.csv"

    if not judges_path.exists() or not fans_path.exists():
        raise FileNotFoundError("Missing q4_*_mixed_effects_params.csv for fixed effects forest plot.")

    judges_df = _standardize_params_df(pd.read_csv(judges_path))
    fans_df = _standardize_params_df(pd.read_csv(fans_path))
    overall_df = _standardize_params_df(pd.read_csv(overall_path)) if overall_path.exists() else pd.DataFrame()

    target_specs = []
    if not judges_df.empty:
        target_specs.append(("Judges", judges_df, "#1f77b4", "o"))
    if not fans_df.empty:
        target_specs.append(("Fans", fans_df, "#ff7f0e", "s"))
    if overall_path.exists() and not overall_df.empty:
        target_specs.append(("Overall", overall_df, "#2ca02c", "D"))

    if not target_specs:
        raise ValueError("No valid params data for fixed effects forest plot.")

    industry_rows = []
    for label, df, _, _ in target_specs:
        ind_df = _extract_industry_rows(df)
        if ind_df.empty:
            continue
        ind_df = ind_df.dropna(subset=["coef"])
        if ind_df.empty:
            continue
        ind_df = ind_df.copy()
        ind_df["target"] = label
        industry_rows.append(ind_df)

    if industry_rows:
        industry_all = pd.concat(industry_rows, ignore_index=True)
        if industry_all.empty:
            top_levels: List[str] = []
        else:
            importance = industry_all.groupby("level")["coef"].apply(lambda s: s.abs().max())
            top_levels = importance.sort_values(ascending=False).head(top_k_industry).index.tolist()
    else:
        top_levels = []

    plot_rows: List[Dict[str, object]] = []
    for label, df, _, _ in target_specs:
        age_row = _extract_age_row(df)
        if age_row is not None:
            plot_rows.append(
                {
                    "var_key": "Age",
                    "var_label": "Age",
                    "target": label,
                    "coef": float(age_row["coef"]),
                    "ci_low": float(age_row["ci_low"]),
                    "ci_high": float(age_row["ci_high"]),
                }
            )

        if top_levels:
            ind_df = _extract_industry_rows(df)
            if not ind_df.empty:
                ind_df = ind_df[ind_df["level"].isin(top_levels)]
                if not ind_df.empty:
                    ind_df = (
                        ind_df.groupby("level", as_index=False)
                        .agg({"coef": "mean", "ci_low": "mean", "ci_high": "mean"})
                    )
                    for _, row in ind_df.iterrows():
                        level = row["level"]
                        plot_rows.append(
                            {
                                "var_key": level,
                                "var_label": _clean_display_label(level),
                                "target": label,
                                "coef": float(row["coef"]),
                                "ci_low": float(row["ci_low"]),
                                "ci_high": float(row["ci_high"]),
                            }
                        )

    plot_df = pd.DataFrame(plot_rows)
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    if plot_df.empty:
        raise ValueError("Fixed effects plot data empty; cannot render forest plot.")

    var_order: List[str] = []
    if "Age" in plot_df["var_key"].values:
        var_order.append("Age")
    for level in top_levels:
        if level in plot_df["var_key"].values:
            var_order.append(level)
    extras = [v for v in plot_df["var_key"].unique() if v not in var_order]
    var_order.extend(extras)

    label_map = {row["var_key"]: row["var_label"] for _, row in plot_df.drop_duplicates("var_key").iterrows()}
    y_positions = np.arange(len(var_order))
    y_map = {k: v for k, v in zip(var_order, y_positions)}

    def _wrap_label(label: str, width: int = 30, max_lines: int = 2) -> str:
        wrapped = textwrap.wrap(str(label), width=width)
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            wrapped[-1] = wrapped[-1] + "..."
        return "\n".join(wrapped)

    fig_h = max(4.2, 0.5 * len(var_order) + 1.0)
    fig, ax = plt.subplots(figsize=(10.8, fig_h))

    n_targets = len(target_specs)
    if n_targets == 1:
        offsets = [0.0]
    elif n_targets == 2:
        offsets = [-0.12, 0.12]
    else:
        offsets = [-0.18, 0.0, 0.18]

    for (label, _, color, marker), offset in zip(target_specs, offsets):
        sub = plot_df[plot_df["target"] == label].copy()
        if sub.empty:
            continue
        sub["y"] = sub["var_key"].map(y_map)
        sub = sub.dropna(subset=["y", "coef", "ci_low", "ci_high"])
        if sub.empty:
            continue
        x = sub["coef"].to_numpy()
        y = sub["y"].to_numpy() + offset
        left = (sub["coef"] - sub["ci_low"]).abs().to_numpy()
        right = (sub["ci_high"] - sub["coef"]).abs().to_numpy()
        xerr = np.vstack([left, right])
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            label=label,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_wrap_label(label_map.get(v, v)) for v in var_order])
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title("Fixed Effects Forest Plot: Age & Industry (Judges / Fans / Overall)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    png_path = out_dir_path / "q4_fixed_effects_forest.png"
    pdf_path = out_dir_path / "q4_fixed_effects_forest.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    if logger is not None:
        logger.info("Q4 fixed effects forest plot saved to %s", out_dir_path)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def run_q4_stage_breakdown(
    active_df: pd.DataFrame,
    active_fans_df: pd.DataFrame,
    out_dir: str,
    logger,
) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    stages = ["early", "mid", "late"]
    out_dir_path = Path(out_dir)

    # Overall: 基于 active_df 取每个 season-celebrity 的最后一次 active 记录
    overall_base = active_df.copy()
    if "week_num" not in overall_base.columns:
        overall_base["week_num"] = pd.to_numeric(overall_base["week"], errors="coerce")
    if "last_active_week" in overall_base.columns and overall_base["last_active_week"].notna().any():
        last_week = pd.to_numeric(overall_base["last_active_week"], errors="coerce")
        overall_df = overall_base[overall_base["week_num"] == last_week].copy()
    else:
        overall_df = (
            overall_base.sort_values(["season", "celebrity_name", "week_num"])
            .groupby(["season", "celebrity_name"], as_index=False)
            .tail(1)
            .copy()
        )
    overall_df["overall_raw"] = -pd.to_numeric(overall_df["placement"], errors="coerce")
    overall_df["z_overall"] = _zscore(overall_df["overall_raw"])

    coeff_paths: Dict[str, str] = {}
    group_rows: List[Dict[str, object]] = []

    for stage in stages:
        sub_j = active_df[active_df["stage"] == stage].copy()
        sub_f = active_fans_df[active_fans_df["stage"] == stage].copy()
        sub_o = overall_df[overall_df["stage"] == stage].copy()

        if sub_j.empty or sub_f.empty or sub_o.empty:
            logger.warning("Q4 stage breakdown skipped (%s): empty subset.", stage)
            continue

        j_params, j_type, j_model = _fit_mixedlm(sub_j, "z_judge", logger, f"judges_{stage}")
        f_params, f_type, f_model = _fit_mixedlm(sub_f, "z_fan", logger, f"fans_{stage}", weights=sub_f.get("w_pv"))
        o_params, o_type, o_model = _fit_mixedlm(sub_o, "z_overall", logger, f"overall_{stage}")

        p_j = _extract_partner_effects(j_model, j_type, f"judges_{stage}")
        p_f = _extract_partner_effects(f_model, f_type, f"fans_{stage}")
        p_o = _extract_partner_effects(o_model, o_type, f"overall_{stage}")

        coeff_df = _build_coeff_share_for_plot(
            j_params,
            f_params,
            o_params,
            p_j,
            p_f,
            p_o,
            logger,
        )
        coeff_path = out_dir_path / f"q4_coeff_share_for_plot_stage_{stage}.csv"
        write_csv(coeff_df, str(coeff_path))
        coeff_paths[stage] = str(coeff_path)

        if coeff_df.empty:
            continue

        for tag in ["judge", "fan", "overall"]:
            share_col = f"share_{tag}"
            for group in ["Partner", "Personal"]:
                share_val = coeff_df.loc[coeff_df["feature_group"] == group, share_col].abs().sum()
                group_rows.append(
                    {
                        "stage": stage,
                        "feature_group": group,
                        "target": tag,
                        "share_abs": float(share_val),
                    }
                )

    group_df = pd.DataFrame(group_rows)
    group_path = out_dir_path / "q4_stage_group_share.csv"
    write_csv(group_df, str(group_path))

    # 绘图：三个面板分别展示 Judges/Fans/Overall，每个 stage 一根堆叠柱（Partner/Personal）
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharey=True)
    targets = [("judge", "Judges", "#1f77b4"), ("fan", "Fans", "#ff7f0e"), ("overall", "Overall", "#2ca02c")]

    for ax, (tag, title, color) in zip(axes, targets):
        data = group_df[group_df["target"] == tag].copy()
        if data.empty:
            ax.set_title(title)
            ax.set_xticks(range(len(stages)))
            ax.set_xticklabels(stages)
            continue
        pivot = data.pivot(index="stage", columns="feature_group", values="share_abs").reindex(stages).fillna(0.0)
        bottom = np.zeros(len(pivot))
        for group, gcolor in [("Partner", color), ("Personal", "#8c8c8c")]:
            vals = pivot[group].to_numpy() if group in pivot.columns else np.zeros(len(pivot))
            ax.bar(stages, vals, bottom=bottom, color=gcolor, label=group)
            bottom = bottom + vals
        ax.set_title(title)
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("Abs Share (Partner vs Personal)")
    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle("Stage Breakdown: Partner vs Personal Contribution Shares")
    fig.tight_layout()

    png_path = out_dir_path / "q4_stage_group_share.png"
    pdf_path = out_dir_path / "q4_stage_group_share.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "q4_stage_group_share_csv": str(group_path),
        "q4_stage_group_share_png": str(png_path),
        "q4_stage_group_share_pdf": str(pdf_path),
        **{f"q4_coeff_share_for_plot_stage_{k}": v for k, v in coeff_paths.items()},
    }


def run_q4(
    long_active: pd.DataFrame,
    long_active_winsor: pd.DataFrame,
    long_all: pd.DataFrame,
    q1_vote_point: pd.DataFrame,
    out_dir: str,
    common_dir: str,
    seed: int,
    logger,
    enable_stage_breakdown: bool = False,
) -> Dict[str, str]:
    ensure_dir(out_dir)
    ensure_dir(common_dir)
    model_dir = Path(out_dir) / "models"
    ensure_dir(str(model_dir))

    q1_vote_point = q1_vote_point.copy()
    pv_series, pv_source_series = _select_bayes_pv(q1_vote_point, logger, "Q4")
    q1_vote_point["pv_bayes"] = pv_series
    q1_vote_point["pv_source"] = pv_source_series
    pv_source_global = _resolve_pv_source(pv_source_series)
    logger.info("Q4 pv_source=%s", pv_source_global)

    # Merge P^V into active (robust name key + audit)
    q1_vote_point["celebrity_name_key"] = _normalize_key(q1_vote_point["celebrity_name"])
    active = long_active.copy()
    active["celebrity_name_key"] = _normalize_key(active["celebrity_name"])

    merged_active = active.merge(
        q1_vote_point[["season", "week", "celebrity_name_key", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name_key"],
        how="left",
        indicator=True,
    )
    total_rows = len(merged_active)
    missing_mask = merged_active["_merge"] != "both"
    missing_rows = merged_active[missing_mask]
    missing_rate = (len(missing_rows) / total_rows) if total_rows else 0.0
    logger.warning("Q4 pv merge missing_rate=%.4f (%s/%s)", missing_rate, len(missing_rows), total_rows)

    audit_rows = []
    if not missing_rows.empty:
        for _, r in missing_rows.iterrows():
            audit_rows.append(
                {
                    "season": r["season"],
                    "week": r["week"],
                    "celebrity_name": r["celebrity_name"],
                    "reason": "pv_missing_after_merge",
                    "pv_source": r.get("pv_source", np.nan),
                    "note": "missing after merge with q1_vote_point",
                }
            )
    audit_df = pd.DataFrame(audit_rows)
    write_csv(audit_df, str(Path(out_dir) / "q4_pv_merge_audit.csv"))

    active = merged_active.drop(columns=["_merge"])
    active["pv_hat"] = active["pv_bayes"]

    # Feature engineering: week/stage + inertia + current-week performance
    active = _engineer_q4_features(active, logger)

    # Prepare long_all with pv_bayes for inactive predictions (avoid history leakage)
    all_df = long_all.copy()
    all_df["celebrity_name_key"] = _normalize_key(all_df["celebrity_name"])
    all_df = all_df.merge(
        q1_vote_point[["season", "week", "celebrity_name_key", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name_key"],
        how="left",
    )
    all_df = _engineer_q4_features(all_df, logger)

    # Fans channel: drop missing pv_hat (no silent 0.5 fill)
    fans_before = len(active)
    active_fans = active[~active["pv_hat"].isna()].copy().reset_index(drop=True)
    fans_after = len(active_fans)
    logger.warning("Q4 fans_training_rows_before=%s after_drop=%s", fans_before, fans_after)

    # Uncertainty weights for fans (optional)
    q1_dir = Path(out_dir).parent / "q1"
    w_pv, unc_summary, unc_source = _compute_uncertainty_weights(active_fans, q1_dir, Path(out_dir), logger)
    if w_pv is not None:
        active_fans["w_pv"] = w_pv
        logger.info("Q4 uncertainty weights enabled: %s", unc_source)
    else:
        logger.warning("Q4 uncertainty inputs missing; weights disabled.")
    write_csv(pd.DataFrame([unc_summary]), str(Path(out_dir) / "q4_pv_uncertainty_summary.csv"))

    # Targets
    active_fans["logit_pv"] = _logit(active_fans["pv_hat"].to_numpy())

    # Alignment check for w_pv
    missing_rate = active_fans["w_pv"].isna().mean() if "w_pv" in active_fans.columns else 1.0
    if missing_rate > 0.01:
        fail_path = Path(out_dir) / "q4_wpv_alignment_fail_sample.csv"
        sample = active_fans[active_fans["w_pv"].isna()][["season", "week", "celebrity_name", "pv_hat"]].head(20)
        write_csv(sample, str(fail_path))
        raise ValueError(
            f"w_pv alignment failed (missing_rate={missing_rate:.4f}). Check index alignment and q1 uncertainty merge keys."
        )

    # 标准化目标（便于跨 judges / fans / overall 比较）
    active["z_judge"] = _zscore(active["judge_avg"])
    active_fans["z_fan"] = _zscore(active_fans["logit_pv"])

    # Overall：每个 season-celebrity 仅保留最后一次 active 记录
    overall_base = all_df.copy()
    if "week_num" not in overall_base.columns:
        overall_base["week_num"] = pd.to_numeric(overall_base["week"], errors="coerce")
    if "last_active_week" in overall_base.columns and overall_base["last_active_week"].notna().any():
        last_week = pd.to_numeric(overall_base["last_active_week"], errors="coerce")
        overall_df = overall_base[overall_base["week_num"] == last_week].copy()
    else:
        base = overall_base
        if "active" in base.columns:
            base = base[base["active"] == 1]
        overall_df = (
            base.sort_values(["season", "celebrity_name", "week_num"])
            .groupby(["season", "celebrity_name"], as_index=False)
            .tail(1)
            .copy()
        )

    # overall_raw = -placement：名次越小越好，取负号后“越大越好”与正向贡献一致
    overall_df["overall_raw"] = -pd.to_numeric(overall_df["placement"], errors="coerce")
    overall_df["z_overall"] = _zscore(overall_df["overall_raw"])

    # Mixed effects models（优先 MixedLM，失败则 OLS/WLS）
    judges_params, judges_model_type, judges_model = _fit_mixedlm(active, "z_judge", logger, "judges")
    fans_params, fans_model_type, fans_model = _fit_mixedlm(
        active_fans,
        "z_fan",
        logger,
        "fans",
        weights=active_fans.get("w_pv"),
    )
    overall_params, overall_model_type, overall_model = _fit_mixedlm(overall_df, "z_overall", logger, "overall")
    judges_params["pv_source"] = pv_source_global
    fans_params["pv_source"] = pv_source_global
    overall_params["pv_source"] = "placement"

    # Save mixed models
    joblib.dump(judges_model, model_dir / "q4_judges_mixed_effects.pkl")
    joblib.dump(fans_model, model_dir / "q4_fans_mixed_effects.pkl")
    joblib.dump(overall_model, model_dir / "q4_overall_mixed_effects.pkl")

    # Partner effects（随机截距 BLUP 或 partner dummy）
    partner_judges = _extract_partner_effects(judges_model, judges_model_type, "judges")
    partner_fans = _extract_partner_effects(fans_model, fans_model_type, "fans")
    partner_overall = _extract_partner_effects(overall_model, overall_model_type, "overall")

    # Coefficient share data + plot
    coeff_share_df = _build_coeff_share_for_plot(
        judges_params,
        fans_params,
        overall_params,
        partner_judges,
        partner_fans,
        partner_overall,
        logger,
    )
    coeff_share_path = Path(out_dir) / "q4_coeff_share_for_plot.csv"
    write_csv(coeff_share_df, str(coeff_share_path))
    if coeff_share_df.empty:
        logger.warning("Q4 coeff share data empty; skip plot.")
        plot_paths = {"png": "", "pdf": ""}
    else:
        plot_paths = plot_q4_coeff_share(out_dir)

    stage_paths: Dict[str, str] = {}
    if enable_stage_breakdown:
        stage_paths = run_q4_stage_breakdown(active, active_fans, out_dir, logger)

    out_paths = {
        "q4_judges_mixed_effects_params": str(Path(out_dir) / "q4_judges_mixed_effects_params.csv"),
        "q4_fans_mixed_effects_params": str(Path(out_dir) / "q4_fans_mixed_effects_params.csv"),
        "q4_overall_mixed_effects_params": str(Path(out_dir) / "q4_overall_mixed_effects_params.csv"),
        "q4_partner_effects_judges": str(Path(out_dir) / "q4_partner_effects_judges.csv"),
        "q4_partner_effects_fans": str(Path(out_dir) / "q4_partner_effects_fans.csv"),
        "q4_partner_effects_overall": str(Path(out_dir) / "q4_partner_effects_overall.csv"),
        "q4_coeff_share_for_plot": str(coeff_share_path),
        "q4_coeff_share_plot_png": plot_paths.get("png", ""),
        "q4_coeff_share_plot_pdf": plot_paths.get("pdf", ""),
        "q4_pv_merge_audit": str(Path(out_dir) / "q4_pv_merge_audit.csv"),
        "q4_pv_uncertainty_summary": str(Path(out_dir) / "q4_pv_uncertainty_summary.csv"),
        "q4_personal_feature_share_plot_png": "",
        "q4_personal_feature_share_plot_pdf": "",
        "q4_partner_vs_personal_share_plot_png": "",
        "q4_partner_vs_personal_share_plot_pdf": "",
        "q4_top_homestate_share_plot_png": "",
        "q4_top_homestate_share_plot_pdf": "",
        "q4_top_home_region_share_plot_png": "",
        "q4_top_home_region_share_plot_pdf": "",
        "q4_personal_feature_share_rms_png": "",
        "q4_personal_feature_share_rms_pdf": "",
        "q4_personal_feature_contrib_compare_csv": "",
        **stage_paths,
    }

    write_csv(judges_params, out_paths["q4_judges_mixed_effects_params"])
    write_csv(fans_params, out_paths["q4_fans_mixed_effects_params"])
    write_csv(overall_params, out_paths["q4_overall_mixed_effects_params"])
    write_csv(partner_judges, out_paths["q4_partner_effects_judges"])
    write_csv(partner_fans, out_paths["q4_partner_effects_fans"])
    write_csv(partner_overall, out_paths["q4_partner_effects_overall"])

    fixed_plot_paths = {"png": "", "pdf": ""}
    try:
        fixed_plot_paths = plot_q4_fixed_effects_forest(out_dir, top_k_industry=6, logger=logger)
    except Exception as e:
        logger.warning("Q4 fixed effects forest plot skipped: %s", e)
    out_paths["q4_fixed_effects_forest_png"] = fixed_plot_paths.get("png", "")
    out_paths["q4_fixed_effects_forest_pdf"] = fixed_plot_paths.get("pdf", "")

    leaderboard_paths = {"png": "", "pdf": ""}
    try:
        leaderboard_paths = plot_q4_partner_leaderboard(out_dir)
    except Exception as e:
        logger.warning("Q4 partner leaderboard skipped: %s", e)
    out_paths["q4_partner_leaderboard_png"] = leaderboard_paths.get("png", "")
    out_paths["q4_partner_leaderboard_pdf"] = leaderboard_paths.get("pdf", "")

    consistency_paths = {"png": "", "pdf": ""}
    try:
        consistency_paths = plot_q4_partner_consistency(out_dir)
    except Exception as e:
        logger.warning("Q4 partner consistency plot skipped: %s", e)
    out_paths["q4_partner_consistency_png"] = consistency_paths.get("png", "")
    out_paths["q4_partner_consistency_pdf"] = consistency_paths.get("pdf", "")

    personal_paths = {"png": "", "pdf": ""}
    try:
        personal_paths = plot_q4_personal_feature_share(out_dir)
    except Exception as e:
        logger.warning("Q4 personal feature share plot skipped: %s", e)
    out_paths["q4_personal_feature_share_plot_png"] = personal_paths.get("png", "")
    out_paths["q4_personal_feature_share_plot_pdf"] = personal_paths.get("pdf", "")

    pv_personal_paths = {"png": "", "pdf": ""}
    try:
        pv_personal_paths = plot_q4_partner_vs_personal_share(out_dir)
    except Exception as e:
        logger.warning("Q4 partner vs personal share plot skipped: %s", e)
    out_paths["q4_partner_vs_personal_share_plot_png"] = pv_personal_paths.get("png", "")
    out_paths["q4_partner_vs_personal_share_plot_pdf"] = pv_personal_paths.get("pdf", "")

    geo_state_paths = {"png": "", "pdf": ""}
    try:
        geo_state_paths = plot_q4_geo_topk_share(out_dir, kind="homestate", top_k=10)
    except Exception as e:
        logger.warning("Q4 top homestate share plot skipped: %s", e)
    out_paths["q4_top_homestate_share_plot_png"] = geo_state_paths.get("png", "")
    out_paths["q4_top_homestate_share_plot_pdf"] = geo_state_paths.get("pdf", "")

    geo_region_paths = {"png": "", "pdf": ""}
    try:
        geo_region_paths = plot_q4_geo_topk_share(out_dir, kind="region", top_k=8)
    except Exception as e:
        logger.warning("Q4 top home region share plot skipped: %s", e)
    out_paths["q4_top_home_region_share_plot_png"] = geo_region_paths.get("png", "")
    out_paths["q4_top_home_region_share_plot_pdf"] = geo_region_paths.get("pdf", "")

    rms_paths = {"png": "", "pdf": "", "csv": ""}
    try:
        rms_paths = plot_q4_personal_feature_share_rms(out_dir)
    except Exception as e:
        logger.warning("Q4 personal feature RMS share plot skipped: %s", e)
    out_paths["q4_personal_feature_share_rms_png"] = rms_paths.get("png", "")
    out_paths["q4_personal_feature_share_rms_pdf"] = rms_paths.get("pdf", "")
    out_paths["q4_personal_feature_contrib_compare_csv"] = rms_paths.get("csv", "")

    try:
        run_q4_robustness(out_dir, enable=False, n_boot=200, n_perm=200, seed=seed)
    except Exception as e:
        logger.warning("Q4 robustness skipped: %s", e)

    logger.info("Q4 outputs saved to %s", out_dir)
    return out_paths


# ===========================
# Q4 Robustness Module
# ===========================
def _find_data_processed_dir(out_dir_path: Path) -> Path | None:
    for parent in [out_dir_path, *out_dir_path.parents]:
        candidate = parent / "data_processed"
        if candidate.exists():
            return candidate
    return None


def _extract_required_cols(formula_terms: List[str]) -> List[str]:
    cols: List[str] = []
    for term in formula_terms:
        t = str(term).strip()
        if t.startswith("C(") and t.endswith(")"):
            cols.append(t[2:-1])
        else:
            cols.append(t)
    return cols


def _result_to_params_df(result, model_type: str, target: str) -> pd.DataFrame:
    params = getattr(result, "params", pd.Series(dtype=float))
    bse = getattr(result, "bse", pd.Series(dtype=float))
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
    try:
        ci = result.conf_int()
    except Exception:
        ci = pd.DataFrame(index=params.index, columns=[0, 1])

    rows: List[Dict[str, object]] = []
    for term in params.index:
        rows.append(
            {
                "term": term,
                "coef": params[term],
                "std_err": bse[term] if term in bse.index else np.nan,
                "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
                "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
                "pvalue": pvalues[term] if term in pvalues.index else np.nan,
                "model_type": model_type,
                "target": target,
            }
        )
    return pd.DataFrame(rows)


def _fit_mixedlm_custom(
    df: pd.DataFrame,
    target: str,
    logger,
    model_tag: str,
    formula_terms: List[str],
    weights: pd.Series | None = None,
    partner_as_fixed: bool = False,
    force_mixed: bool = False,
) -> Tuple[pd.DataFrame, str, object]:
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    if "ballroom_partner" not in df.columns:
        df["ballroom_partner"] = "Unknown"
    df["ballroom_partner"] = df["ballroom_partner"].fillna("Unknown")

    if weights is not None:
        w_series = pd.to_numeric(weights, errors="coerce")
        if len(w_series) == len(df):
            w_series = pd.Series(w_series.to_numpy(), index=df.index)
        else:
            w_series = w_series.reindex(df.index)
        df["_w_pv"] = w_series

    required_cols = [target] + _extract_required_cols(formula_terms)
    before_rows = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = before_rows - len(df)
    if dropped > 0 and logger is not None:
        logger.warning("Q4 robust rows dropped due to NaN in required cols: %s", dropped)

    formula = f"{target} ~ " + " + ".join(formula_terms) if formula_terms else f"{target} ~ 1"

    use_wls = model_tag.startswith("fans") and weights is not None and not force_mixed
    if partner_as_fixed or use_wls:
        if "C(ballroom_partner)" not in formula:
            formula = formula + " + C(ballroom_partner)"
        if use_wls:
            model_type = "WLS"
            w = pd.to_numeric(df.get("_w_pv"), errors="coerce").fillna(1.0)
            result = smf.wls(formula, df, weights=w).fit(cov_type="HC1")
        else:
            model_type = "OLS"
            result = smf.ols(formula, df).fit(cov_type="HC1")
        params_df = _result_to_params_df(result, model_type, target)
        return params_df, model_type, result

    model_type = "MixedLM"
    result = None
    try:
        md = smf.mixedlm(formula, df, groups=df["ballroom_partner"])
        result = md.fit(reml=False, method="lbfgs", maxiter=200, disp=False)
    except Exception as e:
        if logger is not None:
            logger.warning("Q4 robust MixedLM(lbfgs) failed for %s: %s. Retrying with cg.", model_tag, e)
        try:
            result = md.fit(reml=False, method="cg", maxiter=200, disp=False)
        except Exception as e2:
            if logger is not None:
                logger.warning("Q4 robust MixedLM failed for %s: %s. Falling back to OLS with partner FE.", model_tag, e2)
            model_type = "OLS"
            formula = formula + " + C(ballroom_partner)"
            result = smf.ols(formula, df).fit(cov_type="HC1")

    params_df = _result_to_params_df(result, model_type, target)
    return params_df, model_type, result


def _short_term_label(term: str, max_len: int = 20) -> str:
    text = str(term)
    text = text.replace("C(", "").replace(")", "")
    text = text.replace("celebrity_homecountry_region", "homecountry")
    text = text.replace("celebrity_homestate", "homestate")
    text = text.replace("celebrity_industry", "industry")
    text = text.replace("celebrity_age_during_season", "age")
    text = text.replace("ballroom_partner", "partner")
    text = text.replace("celebrity_", "")
    if "[T." in text:
        text = text.split("[T.", 1)[1].rstrip("]")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _wrap_text(label: str, width: int = 24, max_lines: int = 2) -> str:
    import textwrap

    text = str(label).replace("_", " ").strip()
    wrapped = textwrap.wrap(text, width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1] + "..."
    return "\n".join(wrapped) if wrapped else "Unknown"


def _apply_robust_style(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_facecolor("white")


def _save_robust_fig(fig, base_path: Path) -> Dict[str, str]:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def _build_overall_df(all_df: pd.DataFrame) -> pd.DataFrame:
    overall_base = all_df.copy()
    if "week_num" not in overall_base.columns:
        overall_base["week_num"] = pd.to_numeric(overall_base["week"], errors="coerce")
    if "last_active_week" in overall_base.columns and overall_base["last_active_week"].notna().any():
        last_week = pd.to_numeric(overall_base["last_active_week"], errors="coerce")
        overall_df = overall_base[overall_base["week_num"] == last_week].copy()
    else:
        base = overall_base
        if "active" in base.columns:
            base = base[base["active"] == 1]
        overall_df = (
            base.sort_values(["season", "celebrity_name", "week_num"])
            .groupby(["season", "celebrity_name"], as_index=False)
            .tail(1)
            .copy()
        )
    overall_df["overall_raw"] = -pd.to_numeric(overall_df["placement"], errors="coerce")
    overall_df["z_overall"] = _zscore(overall_df["overall_raw"])
    return overall_df


def _compute_partner_share(params_df: pd.DataFrame, partner_effects: pd.DataFrame) -> Tuple[float, float, float]:
    age = _extract_age_effect(params_df)
    industries = _extract_industry_effects(params_df)
    personal_total = abs(age) + sum(abs(v) for v in industries.values())
    if partner_effects is None or partner_effects.empty:
        partner_total = 0.0
    else:
        partner_total = float(pd.to_numeric(partner_effects["coef"], errors="coerce").abs().sum())
    denom = personal_total + partner_total
    share = partner_total / denom if denom > 0 else np.nan
    return float(share), float(personal_total), float(partner_total)


def _prepare_q4_robust_inputs(out_dir: str, logger):
    out_dir_path = Path(out_dir)
    data_dir = _find_data_processed_dir(out_dir_path)
    if data_dir is None:
        raise FileNotFoundError("data_processed directory not found for Q4 robustness.")

    long_active_path = data_dir / "dwts_clean_long_active.csv"
    long_all_path = data_dir / "dwts_clean_long_all.csv"
    if not long_active_path.exists() or not long_all_path.exists():
        raise FileNotFoundError("Missing dwts_clean_long_active.csv or dwts_clean_long_all.csv for Q4 robustness.")

    long_active = pd.read_csv(long_active_path)
    long_all = pd.read_csv(long_all_path)

    q1_dir = out_dir_path.parent / "q1"
    q1_vote_point_path = q1_dir / "q1_vote_point.csv"
    if not q1_vote_point_path.exists():
        raise FileNotFoundError(f"Missing q1_vote_point.csv for Q4 robustness: {q1_vote_point_path}")
    q1_vote_point = pd.read_csv(q1_vote_point_path)

    q1_vote_point = q1_vote_point.copy()
    pv_series, pv_source_series = _select_bayes_pv(q1_vote_point, logger, "Q4-ROBUST")
    q1_vote_point["pv_bayes"] = pv_series
    q1_vote_point["pv_source"] = pv_source_series
    q1_vote_point["celebrity_name_key"] = _normalize_key(q1_vote_point["celebrity_name"])

    active = long_active.copy()
    active["celebrity_name_key"] = _normalize_key(active["celebrity_name"])
    active = active.merge(
        q1_vote_point[["season", "week", "celebrity_name_key", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name_key"],
        how="left",
    )
    active["pv_hat"] = active["pv_bayes"]
    active = _engineer_q4_features(active, logger)

    all_df = long_all.copy()
    all_df["celebrity_name_key"] = _normalize_key(all_df["celebrity_name"])
    all_df = all_df.merge(
        q1_vote_point[["season", "week", "celebrity_name_key", "pv_bayes", "pv_source"]],
        on=["season", "week", "celebrity_name_key"],
        how="left",
    )
    all_df = _engineer_q4_features(all_df, logger)

    active_fans = active[~active["pv_hat"].isna()].copy().reset_index(drop=True)
    w_pv, _, _ = _compute_uncertainty_weights(active_fans, q1_dir, out_dir_path, logger)
    if w_pv is None or len(w_pv) != len(active_fans):
        w_pv = pd.Series(np.ones(len(active_fans)), index=active_fans.index, name="w_pv")
    active_fans["w_pv"] = w_pv

    active_fans["logit_pv"] = _logit(active_fans["pv_hat"].to_numpy())
    active["z_judge"] = _zscore(active["judge_avg"])
    active_fans["z_fan"] = _zscore(active_fans["logit_pv"])

    overall_df = _build_overall_df(all_df)
    return {
        "active": active,
        "active_fans": active_fans,
        "overall_df": overall_df,
        "all_df": all_df,
    }


def run_q4_robustness(
    out_dir: str,
    enable: bool = False,
    n_boot: int = 200,
    n_perm: int = 200,
    seed: int = 42,
) -> Dict[str, str]:
    import logging
    import matplotlib.pyplot as plt
    import math

    if not enable:
        return {}

    out_dir_path = Path(out_dir)
    ensure_dir(str(out_dir_path))

    logger = logging.getLogger("q4_robustness")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    rng = np.random.default_rng(seed)

    data_bundle = _prepare_q4_robust_inputs(out_dir, logger)
    active = data_bundle["active"]
    active_fans = data_bundle["active_fans"]
    overall_df = data_bundle["overall_df"]
    all_df = data_bundle["all_df"]

    base_terms = [
        "C(celebrity_industry)",
        "C(celebrity_homestate)",
        "C(celebrity_homecountry_region)",
        "celebrity_age_during_season",
        "C(season)",
    ]
    specs = {
        "specA": {"label": "Spec-A (baseline)", "terms": base_terms},
        "specB": {
            "label": "Spec-B (no industry)",
            "terms": [t for t in base_terms if t != "C(celebrity_industry)"],
        },
        "specC": {
            "label": "Spec-C (no age)",
            "terms": [t for t in base_terms if t != "celebrity_age_during_season"],
        },
        "specD": {
            "label": "Spec-D (core)",
            "terms": ["C(celebrity_industry)", "celebrity_age_during_season"],
        },
    }

    targets = {
        "judges": {"df": active, "target": "z_judge", "weights": None},
        "fans": {"df": active_fans, "target": "z_fan", "weights": active_fans.get("w_pv")},
        "overall": {"df": overall_df, "target": "z_overall", "weights": None},
    }

    outputs: Dict[str, str] = {}
    summary_rows: List[Dict[str, object]] = []
    spec_results: Dict[Tuple[str, str], Dict[str, object]] = {}

    # R1: Spec variants (baseline + alternatives)
    for spec_name, spec_info in specs.items():
        terms = spec_info["terms"]
        for target_name, info in targets.items():
            params_df, model_type, result = _fit_mixedlm_custom(
                info["df"],
                info["target"],
                logger,
                f"{target_name}_{spec_name}",
                terms,
                weights=info["weights"],
                partner_as_fixed=False,
                force_mixed=False,
            )
            params_path = out_dir_path / f"q4_robust_params_{target_name}_{spec_name}.csv"
            write_csv(params_df, str(params_path))
            outputs[f"q4_robust_params_{target_name}_{spec_name}"] = str(params_path)

            partner_df = _extract_partner_effects(result, model_type, f"{target_name}_{spec_name}")
            partner_path = out_dir_path / f"q4_robust_partner_effects_{target_name}_{spec_name}.csv"
            write_csv(partner_df, str(partner_path))
            outputs[f"q4_robust_partner_effects_{target_name}_{spec_name}"] = str(partner_path)

            spec_results[(spec_name, target_name)] = {
                "params": params_df,
                "model": result,
                "model_type": model_type,
                "partner": partner_df,
                "terms": terms,
                "df": info["df"],
                "target_col": info["target"],
                "weights": info["weights"],
            }

    # Coefficient stability plots (baseline vs alt specs)
    for target_name in targets.keys():
        base_params = spec_results[("specA", target_name)]["params"]
        base_params = base_params[~base_params["term"].astype(str).str.startswith("C(ballroom_partner)")].copy()
        base_params = base_params[~base_params["term"].isin(["Intercept", "const"])].copy()
        base_params = base_params.dropna(subset=["coef"])
        base_map = base_params.set_index("term")["coef"].to_dict()

        rows = []
        for spec_name in ["specB", "specC", "specD"]:
            params_df = spec_results[(spec_name, target_name)]["params"]
            params_df = params_df[~params_df["term"].astype(str).str.startswith("C(ballroom_partner)")].copy()
            params_df = params_df[~params_df["term"].isin(["Intercept", "const"])].copy()
            params_df = params_df.dropna(subset=["coef"])
            merged = params_df[params_df["term"].isin(base_map.keys())].copy()
            for _, row in merged.iterrows():
                rows.append(
                    {
                        "term": row["term"],
                        "coef_base": base_map.get(row["term"], np.nan),
                        "coef_spec": row["coef"],
                        "spec": specs[spec_name]["label"],
                    }
                )
        pts = pd.DataFrame(rows)
        if not pts.empty:
            fig, ax = plt.subplots(figsize=(6.8, 5.4))
            for spec_label, g in pts.groupby("spec"):
                ax.scatter(g["coef_base"], g["coef_spec"], alpha=0.75, s=36, label=spec_label)
            min_v = float(np.nanmin(np.r_[pts["coef_base"].to_numpy(), pts["coef_spec"].to_numpy()]))
            max_v = float(np.nanmax(np.r_[pts["coef_base"].to_numpy(), pts["coef_spec"].to_numpy()]))
            span = max_v - min_v
            pad = 0.08 * span if span > 0 else 0.2
            ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], linestyle="--", color="gray")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Baseline Coefficient")
            ax.set_ylabel("Alt Spec Coefficient")
            ax.set_title(f"Coefficient Stability ({target_name.capitalize()})")
            ax.legend(loc="lower right", frameon=True)
            _apply_robust_style(ax)

            pts["abs_diff"] = (pts["coef_spec"] - pts["coef_base"]).abs()
            top = pts.nlargest(min(6, len(pts)), "abs_diff")
            for _, row in top.iterrows():
                ax.annotate(
                    _short_term_label(row["term"]),
                    (row["coef_base"], row["coef_spec"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )

            fig.tight_layout()
            base = out_dir_path / f"q4_robust_coef_stability_{target_name}"
            paths = _save_robust_fig(fig, base)
            plt.close(fig)
            outputs[f"q4_robust_coef_stability_{target_name}_png"] = paths["png"]
            outputs[f"q4_robust_coef_stability_{target_name}_pdf"] = paths["pdf"]

            for spec_label in pts["spec"].unique():
                sub = pts[pts["spec"] == spec_label]
                if len(sub) >= 2:
                    corr = sub["coef_base"].corr(sub["coef_spec"])
                    summary_rows.append(
                        {
                            "check_name": "R1_spec",
                            "target": target_name,
                            "metric_name": "coef_corr",
                            "baseline": 1.0,
                            "alt": float(corr) if pd.notna(corr) else np.nan,
                            "delta": float(corr - 1.0) if pd.notna(corr) else np.nan,
                            "note": spec_label,
                        }
                    )

    # R2: Partner random vs fixed
    for target_name, info in targets.items():
        terms = specs["specA"]["terms"]
        random_params, random_type, random_model = _fit_mixedlm_custom(
            info["df"],
            info["target"],
            logger,
            f"{target_name}_random",
            terms,
            weights=info["weights"],
            partner_as_fixed=False,
            force_mixed=True,
        )
        fixed_params, fixed_type, fixed_model = _fit_mixedlm_custom(
            info["df"],
            info["target"],
            logger,
            f"{target_name}_fixed",
            terms,
            weights=info["weights"],
            partner_as_fixed=True,
            force_mixed=False,
        )
        random_partner = _extract_partner_effects(random_model, random_type, f"{target_name}_random")
        fixed_partner = _extract_partner_effects(fixed_model, fixed_type, f"{target_name}_fixed")
        cmp_df = random_partner.rename(columns={"coef": "effect_random"}).merge(
            fixed_partner.rename(columns={"coef": "effect_fixed"}),
            on="partner_name",
            how="outer",
        )
        cmp_path = out_dir_path / f"q4_robust_partner_compare_{target_name}.csv"
        write_csv(cmp_df, str(cmp_path))
        outputs[f"q4_robust_partner_compare_{target_name}"] = str(cmp_path)

        if not cmp_df.empty:
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            ax.scatter(cmp_df["effect_random"], cmp_df["effect_fixed"], alpha=0.75, s=40, color="#4c78a8")
            min_v = float(np.nanmin(np.r_[cmp_df["effect_random"].to_numpy(), cmp_df["effect_fixed"].to_numpy()]))
            max_v = float(np.nanmax(np.r_[cmp_df["effect_random"].to_numpy(), cmp_df["effect_fixed"].to_numpy()]))
            span = max_v - min_v
            pad = 0.08 * span if span > 0 else 0.2
            ax.plot([min_v - pad, max_v + pad], [min_v - pad, max_v + pad], linestyle="--", color="gray")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Random Effect (MixedLM)")
            ax.set_ylabel("Fixed Effect (OLS/WLS)")
            ax.set_title(f"Partner Random vs Fixed ({target_name.capitalize()})")
            _apply_robust_style(ax)

            cmp_df["abs_diff"] = (cmp_df["effect_fixed"] - cmp_df["effect_random"]).abs()
            top = cmp_df.nlargest(min(6, len(cmp_df)), "abs_diff")
            for _, row in top.iterrows():
                ax.annotate(
                    _wrap_text(row["partner_name"], width=18, max_lines=1),
                    (row["effect_random"], row["effect_fixed"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )
            fig.tight_layout()
            base = out_dir_path / f"q4_robust_partner_random_vs_fixed_{target_name}"
            paths = _save_robust_fig(fig, base)
            plt.close(fig)
            outputs[f"q4_robust_partner_random_vs_fixed_{target_name}_png"] = paths["png"]
            outputs[f"q4_robust_partner_random_vs_fixed_{target_name}_pdf"] = paths["pdf"]

            if len(cmp_df) >= 2:
                corr = cmp_df["effect_random"].corr(cmp_df["effect_fixed"])
                summary_rows.append(
                    {
                        "check_name": "R2_random_vs_fixed",
                        "target": target_name,
                        "metric_name": "partner_corr",
                        "baseline": 1.0,
                        "alt": float(corr) if pd.notna(corr) else np.nan,
                        "delta": float(corr - 1.0) if pd.notna(corr) else np.nan,
                        "note": "random_vs_fixed",
                    }
                )

    # R3: Fans weighting sensitivity
    fans_info = targets["fans"]
    fans_terms = specs["specA"]["terms"]
    weights_base = fans_info["weights"]
    if weights_base is None or len(weights_base) != len(fans_info["df"]):
        weights_base = pd.Series(np.ones(len(fans_info["df"])), index=fans_info["df"].index, name="w_pv")

    w0 = pd.Series(weights_base.to_numpy(), index=fans_info["df"].index)
    w1 = pd.Series(np.ones(len(fans_info["df"])), index=fans_info["df"].index)
    q05, q95 = np.nanpercentile(w0, 5), np.nanpercentile(w0, 95)
    w2 = w0.clip(lower=q05, upper=q95)

    weight_sets = {"W0_baseline": w0, "W1_equal": w1, "W2_clip": w2}
    weight_rows = []

    baseline_params = None
    baseline_partner = None
    baseline_industry_terms: List[str] = []
    for w_tag, w_vals in weight_sets.items():
        params_df, model_type, model = _fit_mixedlm_custom(
            fans_info["df"],
            fans_info["target"],
            logger,
            f"fans_weight_{w_tag}",
            fans_terms,
            weights=w_vals,
            partner_as_fixed=True,
            force_mixed=False,
        )
        partner_df = _extract_partner_effects(model, model_type, f"fans_weight_{w_tag}")
        age_coef = params_df.loc[params_df["term"] == "celebrity_age_during_season", "coef"]
        age_val = float(age_coef.iloc[0]) if not age_coef.empty else np.nan

        industry_terms = params_df["term"].astype(str).str.startswith("C(celebrity_industry)[T.")
        industry_df = params_df[industry_terms].copy()

        if w_tag == "W0_baseline":
            baseline_params = params_df
            baseline_partner = partner_df
            baseline_industry_terms = (
                industry_df.reindex(industry_df["coef"].abs().sort_values(ascending=False).index)
                .head(5)["term"]
                .tolist()
            )

        industry_corr = np.nan
        partner_corr = np.nan
        if baseline_params is not None and baseline_partner is not None:
            if baseline_industry_terms:
                base_ind = baseline_params[baseline_params["term"].isin(baseline_industry_terms)].set_index("term")["coef"]
                alt_ind = params_df[params_df["term"].isin(baseline_industry_terms)].set_index("term")["coef"]
                if len(base_ind) >= 2 and len(alt_ind) >= 2:
                    industry_corr = base_ind.corr(alt_ind)
            if not baseline_partner.empty and not partner_df.empty:
                merged = baseline_partner.rename(columns={"coef": "base"}).merge(
                    partner_df.rename(columns={"coef": "alt"}),
                    on="partner_name",
                    how="inner",
                )
                if len(merged) >= 2:
                    partner_corr = merged["base"].corr(merged["alt"])

        weight_rows.append(
            {
                "weight_tag": w_tag,
                "age_coef": age_val,
                "industry_corr": float(industry_corr) if pd.notna(industry_corr) else np.nan,
                "partner_corr": float(partner_corr) if pd.notna(partner_corr) else np.nan,
            }
        )

    weight_df = pd.DataFrame(weight_rows)
    weight_path = out_dir_path / "q4_robust_fans_weighting_summary.csv"
    write_csv(weight_df, str(weight_path))
    outputs["q4_robust_fans_weighting_summary"] = str(weight_path)

    if not weight_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(weight_df["weight_tag"], weight_df["age_coef"], color="#4c78a8", alpha=0.85)
        axes[0].set_title("Age Coefficient (Fans)")
        axes[0].set_ylabel("Coefficient")
        axes[0].tick_params(axis="x", rotation=20)
        _apply_robust_style(axes[0])

        corr_df = weight_df.copy()
        x = np.arange(len(corr_df))
        width = 0.35
        axes[1].bar(x - width / 2, corr_df["industry_corr"], width, label="Industry corr", color="#f28e2b")
        axes[1].bar(x + width / 2, corr_df["partner_corr"], width, label="Partner corr", color="#59a14f")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(corr_df["weight_tag"], rotation=20)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Correlation vs Baseline")
        axes[1].legend(loc="lower right", frameon=True)
        _apply_robust_style(axes[1])

        fig.suptitle("Fans Weighting Sensitivity")
        fig.tight_layout()
        base = out_dir_path / "q4_robust_fans_weighting_sensitivity"
        paths = _save_robust_fig(fig, base)
        plt.close(fig)
        outputs["q4_robust_fans_weighting_sensitivity_png"] = paths["png"]
        outputs["q4_robust_fans_weighting_sensitivity_pdf"] = paths["pdf"]

        for _, row in weight_df.iterrows():
            summary_rows.append(
                {
                    "check_name": "R3_weighting",
                    "target": "fans",
                    "metric_name": "partner_corr",
                    "baseline": 1.0,
                    "alt": row["partner_corr"],
                    "delta": row["partner_corr"] - 1.0 if pd.notna(row["partner_corr"]) else np.nan,
                    "note": row["weight_tag"],
                }
            )

    # R4: LOSO and Stage robustness
    loso_rows: List[Dict[str, object]] = []
    seasons = sorted([s for s in active["season"].dropna().unique()])
    for season_left in seasons:
        sub_active = active[active["season"] != season_left].copy()
        sub_fans = active_fans[active_fans["season"] != season_left].copy()
        sub_all = all_df[all_df["season"] != season_left].copy()

        sub_active["z_judge"] = _zscore(sub_active["judge_avg"])
        sub_fans["logit_pv"] = _logit(sub_fans["pv_hat"].to_numpy())
        sub_fans["z_fan"] = _zscore(sub_fans["logit_pv"])
        sub_overall = _build_overall_df(sub_all)

        sub_targets = {
            "judges": {"df": sub_active, "target": "z_judge", "weights": None},
            "fans": {"df": sub_fans, "target": "z_fan", "weights": sub_fans.get("w_pv")},
            "overall": {"df": sub_overall, "target": "z_overall", "weights": None},
        }
        for t_name, info in sub_targets.items():
            params_df, model_type, model = _fit_mixedlm_custom(
                info["df"],
                info["target"],
                logger,
                f"{t_name}_loso",
                specs["specA"]["terms"],
                weights=info["weights"],
                partner_as_fixed=False,
                force_mixed=False,
            )
            partner_df = _extract_partner_effects(model, model_type, f"{t_name}_loso")
            age_coef = params_df.loc[params_df["term"] == "celebrity_age_during_season", "coef"]
            age_val = float(age_coef.iloc[0]) if not age_coef.empty else np.nan
            partner_share, _, _ = _compute_partner_share(params_df, partner_df)
            loso_rows.append(
                {
                    "season_left_out": season_left,
                    "target": t_name,
                    "coef_age": age_val,
                    "partner_share": partner_share,
                }
            )

    loso_df = pd.DataFrame(loso_rows)
    loso_path = out_dir_path / "q4_robust_loso_summary.csv"
    write_csv(loso_df, str(loso_path))
    outputs["q4_robust_loso_summary"] = str(loso_path)

    if not loso_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.2), sharex=True)
        for t_name, color in [("judges", "#1f77b4"), ("fans", "#ff7f0e"), ("overall", "#2ca02c")]:
            sub = loso_df[loso_df["target"] == t_name]
            axes[0].plot(sub["season_left_out"], sub["coef_age"], marker="o", label=t_name, color=color)
            axes[1].plot(sub["season_left_out"], sub["partner_share"], marker="o", label=t_name, color=color)
        axes[0].set_ylabel("Age Coefficient")
        axes[0].set_title("LOSO: Age Coefficient")
        axes[1].set_ylabel("Partner Share")
        axes[1].set_xlabel("Left-out Season")
        axes[1].set_title("LOSO: Partner Share")
        axes[1].legend(loc="best", frameon=True)
        _apply_robust_style(axes[0])
        _apply_robust_style(axes[1])
        fig.tight_layout()
        base = out_dir_path / "q4_robust_loso"
        paths = _save_robust_fig(fig, base)
        plt.close(fig)
        outputs["q4_robust_loso_png"] = paths["png"]
        outputs["q4_robust_loso_pdf"] = paths["pdf"]

        for t_name in loso_df["target"].unique():
            sub = loso_df[loso_df["target"] == t_name]
            if len(sub) >= 2:
                age_std = float(sub["coef_age"].std(ddof=0))
                share_std = float(sub["partner_share"].std(ddof=0))
                summary_rows.append(
                    {
                        "check_name": "R4_loso",
                        "target": t_name,
                        "metric_name": "age_coef_std",
                        "baseline": 0.0,
                        "alt": age_std,
                        "delta": age_std,
                        "note": "loso",
                    }
                )
                summary_rows.append(
                    {
                        "check_name": "R4_loso",
                        "target": t_name,
                        "metric_name": "partner_share_std",
                        "baseline": 0.0,
                        "alt": share_std,
                        "delta": share_std,
                        "note": "loso",
                    }
                )

    stage_rows: List[Dict[str, object]] = []
    if "stage" in active.columns:
        stages = ["early", "mid", "late"]
        for stage in stages:
            sub_active = active[active["stage"] == stage].copy()
            sub_fans = active_fans[active_fans["stage"] == stage].copy()
            sub_overall = overall_df[overall_df["stage"] == stage].copy()
            sub_targets = {
                "judges": {"df": sub_active, "target": "z_judge", "weights": None},
                "fans": {"df": sub_fans, "target": "z_fan", "weights": sub_fans.get("w_pv")},
                "overall": {"df": sub_overall, "target": "z_overall", "weights": None},
            }
            for t_name, info in sub_targets.items():
                if info["df"].empty:
                    continue
                params_df, model_type, model = _fit_mixedlm_custom(
                    info["df"],
                    info["target"],
                    logger,
                    f"{t_name}_stage_{stage}",
                    specs["specA"]["terms"],
                    weights=info["weights"],
                    partner_as_fixed=False,
                    force_mixed=False,
                )
                partner_df = _extract_partner_effects(model, model_type, f"{t_name}_stage_{stage}")
                partner_share, personal_total, partner_total = _compute_partner_share(params_df, partner_df)
                stage_rows.append(
                    {
                        "stage": stage,
                        "target": t_name,
                        "partner_share": partner_share,
                        "personal_total": personal_total,
                        "partner_total": partner_total,
                    }
                )

        stage_df = pd.DataFrame(stage_rows)
        stage_path = out_dir_path / "q4_robust_stage_summary.csv"
        write_csv(stage_df, str(stage_path))
        outputs["q4_robust_stage_summary"] = str(stage_path)

        if not stage_df.empty:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
            for ax, t_name, color in zip(
                axes,
                ["judges", "fans", "overall"],
                ["#1f77b4", "#ff7f0e", "#2ca02c"],
            ):
                sub = stage_df[stage_df["target"] == t_name].copy()
                sub = sub.set_index("stage").reindex(stages)
                partner_vals = sub["partner_share"].fillna(0.0).to_numpy()
                personal_vals = (1.0 - sub["partner_share"].fillna(0.0)).to_numpy()
                ax.bar(stages, partner_vals, color=color, label="Partner")
                ax.bar(stages, personal_vals, bottom=partner_vals, color="#bdbdbd", label="Personal")
                ax.set_title(t_name.capitalize())
                ax.set_ylim(0, 1.0)
                _apply_robust_style(ax)
            axes[0].set_ylabel("Share")
            axes[0].legend(loc="upper right", frameon=True)
            fig.suptitle("Stage Robustness: Partner vs Personal Share")
            fig.tight_layout()
            base = out_dir_path / "q4_robust_stage_share"
            paths = _save_robust_fig(fig, base)
            plt.close(fig)
            outputs["q4_robust_stage_share_png"] = paths["png"]
            outputs["q4_robust_stage_share_pdf"] = paths["pdf"]

    # R5: Trimmed residual robustness
    for target_name, info in targets.items():
        base_res = spec_results[("specA", target_name)]
        base_df = info["df"].copy()
        base_target = info["target"]
        model = base_res["model"]
        try:
            fitted = getattr(model, "fittedvalues", None)
            if fitted is None:
                fitted = model.predict(base_df)
            resid = pd.to_numeric(base_df[base_target], errors="coerce") - pd.to_numeric(fitted, errors="coerce")
        except Exception:
            resid = pd.Series(np.nan, index=base_df.index)
        threshold = resid.abs().quantile(0.99)
        trimmed_df = base_df[resid.abs() <= threshold].copy()
        trimmed_weights = None
        if target_name == "fans" and "w_pv" in trimmed_df.columns:
            trimmed_weights = trimmed_df["w_pv"]

        trim_params, trim_type, trim_model = _fit_mixedlm_custom(
            trimmed_df,
            base_target,
            logger,
            f"{target_name}_trimmed",
            specs["specA"]["terms"],
            weights=trimmed_weights,
            partner_as_fixed=False,
            force_mixed=False,
        )
        trim_partner = _extract_partner_effects(trim_model, trim_type, f"{target_name}_trimmed")

        base_params = base_res["params"]
        base_partner = base_res["partner"]
        cmp_rows = []
        for _, row in base_params.iterrows():
            term = row["term"]
            if term.startswith("C(ballroom_partner)"):
                continue
            alt_row = trim_params.loc[trim_params["term"] == term, "coef"]
            alt_val = float(alt_row.iloc[0]) if not alt_row.empty else np.nan
            cmp_rows.append(
                {
                    "group": "Fixed",
                    "name": term,
                    "coef_base": row["coef"],
                    "coef_trimmed": alt_val,
                    "delta": alt_val - row["coef"] if pd.notna(alt_val) else np.nan,
                }
            )

        if base_partner is not None and not base_partner.empty:
            merged = base_partner.rename(columns={"coef": "coef_base"}).merge(
                trim_partner.rename(columns={"coef": "coef_trimmed"}),
                on="partner_name",
                how="outer",
            )
            for _, row in merged.iterrows():
                cmp_rows.append(
                    {
                        "group": "Partner",
                        "name": row["partner_name"],
                        "coef_base": row["coef_base"],
                        "coef_trimmed": row["coef_trimmed"],
                        "delta": row["coef_trimmed"] - row["coef_base"]
                        if pd.notna(row["coef_trimmed"]) and pd.notna(row["coef_base"])
                        else np.nan,
                    }
                )

        cmp_df = pd.DataFrame(cmp_rows)
        cmp_path = out_dir_path / f"q4_robust_trimmed_compare_{target_name}.csv"
        write_csv(cmp_df, str(cmp_path))
        outputs[f"q4_robust_trimmed_compare_{target_name}"] = str(cmp_path)

        if not cmp_df.empty:
            cmp_df["abs_delta"] = cmp_df["delta"].abs()
            top = cmp_df.nlargest(min(12, len(cmp_df)), "abs_delta")
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            y = np.arange(len(top))
            ax.barh(y, top["delta"], color="#4c78a8")
            ax.set_yticks(y)
            ax.set_yticklabels([_wrap_text(_short_term_label(v), width=18, max_lines=1) for v in top["name"]])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Trimmed - Baseline Coefficient")
            ax.set_title(f"Trimmed Residual Sensitivity ({target_name.capitalize()})")
            _apply_robust_style(ax)
            fig.tight_layout()
            base = out_dir_path / f"q4_robust_trimmed_delta_{target_name}"
            paths = _save_robust_fig(fig, base)
            plt.close(fig)
            outputs[f"q4_robust_trimmed_delta_{target_name}_png"] = paths["png"]
            outputs[f"q4_robust_trimmed_delta_{target_name}_pdf"] = paths["pdf"]

        age_delta = cmp_df.loc[cmp_df["name"] == "celebrity_age_during_season", "delta"]
        if not age_delta.empty:
            summary_rows.append(
                {
                    "check_name": "R5_trimmed",
                    "target": target_name,
                    "metric_name": "age_delta",
                    "baseline": 0.0,
                    "alt": float(age_delta.iloc[0]),
                    "delta": float(age_delta.iloc[0]),
                    "note": "trim_top1pct",
                }
            )

    # R6: Bootstrap + Placebo
    boot_rows: List[Dict[str, object]] = []
    baseline_terms = specs["specA"]["terms"]
    baseline_industry = {}
    for t_name in targets.keys():
        params = spec_results[("specA", t_name)]["params"]
        industry_terms = params["term"].astype(str).str.startswith("C(celebrity_industry)[T.")
        top_terms = (
            params[industry_terms]
            .reindex(params.loc[industry_terms, "coef"].abs().sort_values(ascending=False).index)
            .head(5)["term"]
            .tolist()
        )
        baseline_industry[t_name] = top_terms

    def _sample_by_season(df: pd.DataFrame) -> pd.DataFrame:
        if "season" not in df.columns:
            return df.sample(frac=1.0, replace=True, random_state=rng.integers(0, 1_000_000))
        seasons_local = df["season"].dropna().unique()
        if len(seasons_local) == 0:
            return df.copy()
        sampled = rng.choice(seasons_local, size=len(seasons_local), replace=True)
        parts = [df[df["season"] == s] for s in sampled]
        return pd.concat(parts, ignore_index=True)

    for b in range(n_boot):
        boot_active = _sample_by_season(active)
        boot_fans = _sample_by_season(active_fans)
        boot_all = _sample_by_season(all_df)
        boot_active["z_judge"] = _zscore(boot_active["judge_avg"])
        boot_fans["logit_pv"] = _logit(boot_fans["pv_hat"].to_numpy())
        boot_fans["z_fan"] = _zscore(boot_fans["logit_pv"])
        boot_overall = _build_overall_df(boot_all)

        boot_targets = {
            "judges": {"df": boot_active, "target": "z_judge", "weights": None},
            "fans": {"df": boot_fans, "target": "z_fan", "weights": boot_fans.get("w_pv")},
            "overall": {"df": boot_overall, "target": "z_overall", "weights": None},
        }
        for t_name, info in boot_targets.items():
            params_df, model_type, model = _fit_mixedlm_custom(
                info["df"],
                info["target"],
                logger,
                f"{t_name}_boot",
                baseline_terms,
                weights=info["weights"],
                partner_as_fixed=False,
                force_mixed=False,
            )
            partner_df = _extract_partner_effects(model, model_type, f"{t_name}_boot")
            age_coef = params_df.loc[params_df["term"] == "celebrity_age_during_season", "coef"]
            age_val = float(age_coef.iloc[0]) if not age_coef.empty else np.nan
            partner_share, _, _ = _compute_partner_share(params_df, partner_df)
            max_abs = (
                float(pd.to_numeric(partner_df["coef"], errors="coerce").abs().max())
                if partner_df is not None and not partner_df.empty
                else np.nan
            )
            boot_rows.append({"boot_id": b, "target": t_name, "metric": "age_coef", "value": age_val})
            boot_rows.append({"boot_id": b, "target": t_name, "metric": "partner_share", "value": partner_share})
            boot_rows.append({"boot_id": b, "target": t_name, "metric": "max_abs_partner", "value": max_abs})
            for term in baseline_industry.get(t_name, []):
                val_row = params_df.loc[params_df["term"] == term, "coef"]
                val = float(val_row.iloc[0]) if not val_row.empty else np.nan
                boot_rows.append({"boot_id": b, "target": t_name, "metric": f"industry::{term}", "value": val})

    boot_df = pd.DataFrame(boot_rows)
    boot_path = out_dir_path / "q4_robust_bootstrap_samples.csv"
    write_csv(boot_df, str(boot_path))
    outputs["q4_robust_bootstrap_samples"] = str(boot_path)

    if not boot_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for t_name, color in [("judges", "#1f77b4"), ("fans", "#ff7f0e"), ("overall", "#2ca02c")]:
            sub = boot_df[(boot_df["target"] == t_name) & (boot_df["metric"] == "age_coef")]
            axes[0].hist(sub["value"].dropna(), bins=24, alpha=0.5, label=t_name, color=color)
            sub2 = boot_df[(boot_df["target"] == t_name) & (boot_df["metric"] == "partner_share")]
            axes[1].hist(sub2["value"].dropna(), bins=24, alpha=0.5, label=t_name, color=color)
        axes[0].set_title("Bootstrap Age Coefficient")
        axes[1].set_title("Bootstrap Partner Share")
        axes[0].set_xlabel("Age Coef")
        axes[1].set_xlabel("Partner Share")
        axes[0].legend(loc="best", frameon=True)
        axes[1].legend(loc="best", frameon=True)
        _apply_robust_style(axes[0])
        _apply_robust_style(axes[1])
        fig.tight_layout()
        base = out_dir_path / "q4_robust_bootstrap_dist"
        paths = _save_robust_fig(fig, base)
        plt.close(fig)
        outputs["q4_robust_bootstrap_dist_png"] = paths["png"]
        outputs["q4_robust_bootstrap_dist_pdf"] = paths["pdf"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, t_name in zip(axes, ["judges", "fans", "overall"]):
            terms = ["celebrity_age_during_season"] + baseline_industry.get(t_name, [])
            records = []
            for term in terms:
                metric = "age_coef" if term == "celebrity_age_during_season" else f"industry::{term}"
                values = boot_df[(boot_df["target"] == t_name) & (boot_df["metric"] == metric)]["value"].dropna()
                if values.empty:
                    continue
                ci_low, ci_high = np.percentile(values, [2.5, 97.5])
                base_coef = spec_results[("specA", t_name)]["params"]
                base_val_row = base_coef.loc[base_coef["term"] == term, "coef"]
                base_val = float(base_val_row.iloc[0]) if not base_val_row.empty else np.nan
                records.append({"term": term, "ci_low": ci_low, "ci_high": ci_high, "base": base_val})
            if records:
                rec_df = pd.DataFrame(records)
                y = np.arange(len(rec_df))
                ax.hlines(y, rec_df["ci_low"], rec_df["ci_high"], color="#4c78a8", linewidth=2)
                ax.plot(rec_df["base"], y, "o", color="#d62728", label="Baseline")
                ax.set_yticks(y)
                ax.set_yticklabels([_short_term_label(t, max_len=18) for t in rec_df["term"]])
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_title(t_name.capitalize())
                _apply_robust_style(ax)
        axes[0].set_ylabel("Term")
        axes[0].legend(loc="lower right", frameon=True)
        fig.suptitle("Bootstrap 95% CI (Age + Top Industry)")
        fig.tight_layout()
        base = out_dir_path / "q4_robust_bootstrap_forest"
        paths = _save_robust_fig(fig, base)
        plt.close(fig)
        outputs["q4_robust_bootstrap_forest_png"] = paths["png"]
        outputs["q4_robust_bootstrap_forest_pdf"] = paths["pdf"]

        for t_name in ["judges", "fans", "overall"]:
            age_vals = boot_df[(boot_df["target"] == t_name) & (boot_df["metric"] == "age_coef")]["value"].dropna()
            share_vals = boot_df[(boot_df["target"] == t_name) & (boot_df["metric"] == "partner_share")]["value"].dropna()
            if not age_vals.empty:
                ci_low, ci_high = np.percentile(age_vals, [2.5, 97.5])
                summary_rows.append(
                    {
                        "check_name": "R6_bootstrap",
                        "target": t_name,
                        "metric_name": "age_ci_width",
                        "baseline": 0.0,
                        "alt": float(ci_high - ci_low),
                        "delta": float(ci_high - ci_low),
                        "note": "bootstrap",
                    }
                )
            if not share_vals.empty:
                ci_low, ci_high = np.percentile(share_vals, [2.5, 97.5])
                summary_rows.append(
                    {
                        "check_name": "R6_bootstrap",
                        "target": t_name,
                        "metric_name": "partner_share_ci_width",
                        "baseline": 0.0,
                        "alt": float(ci_high - ci_low),
                        "delta": float(ci_high - ci_low),
                        "note": "bootstrap",
                    }
                )

    placebo_rows: List[Dict[str, object]] = []

    def _shuffle_partner(df: pd.DataFrame) -> pd.DataFrame:
        shuffled = df.copy()
        if "season" not in shuffled.columns or "ballroom_partner" not in shuffled.columns:
            return shuffled
        for _, idx in shuffled.groupby("season").groups.items():
            vals = shuffled.loc[idx, "ballroom_partner"].to_numpy()
            shuffled.loc[idx, "ballroom_partner"] = rng.permutation(vals)
        return shuffled

    for p in range(n_perm):
        perm_active = _shuffle_partner(active)
        perm_fans = _shuffle_partner(active_fans)
        perm_overall = _shuffle_partner(overall_df)
        perm_targets = {
            "judges": {"df": perm_active, "target": "z_judge", "weights": None},
            "fans": {"df": perm_fans, "target": "z_fan", "weights": perm_fans.get("w_pv")},
            "overall": {"df": perm_overall, "target": "z_overall", "weights": None},
        }
        for t_name, info in perm_targets.items():
            params_df, model_type, model = _fit_mixedlm_custom(
                info["df"],
                info["target"],
                logger,
                f"{t_name}_placebo",
                baseline_terms,
                weights=info["weights"],
                partner_as_fixed=False,
                force_mixed=False,
            )
            partner_df = _extract_partner_effects(model, model_type, f"{t_name}_placebo")
            max_abs = (
                float(pd.to_numeric(partner_df["coef"], errors="coerce").abs().max())
                if partner_df is not None and not partner_df.empty
                else np.nan
            )
            partner_share, _, _ = _compute_partner_share(params_df, partner_df)
            placebo_rows.append(
                {
                    "perm_id": p,
                    "target": t_name,
                    "max_abs_partner": max_abs,
                    "partner_share": partner_share,
                }
            )

    placebo_df = pd.DataFrame(placebo_rows)
    placebo_path = out_dir_path / "q4_robust_placebo_null.csv"
    write_csv(placebo_df, str(placebo_path))
    outputs["q4_robust_placebo_null"] = str(placebo_path)

    if not placebo_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, t_name in zip(axes, ["judges", "fans", "overall"]):
            sub = placebo_df[placebo_df["target"] == t_name]
            ax.hist(sub["max_abs_partner"].dropna(), bins=24, color="#bdbdbd", alpha=0.8)
            observed = spec_results[("specA", t_name)]["partner"]
            obs_max = (
                float(pd.to_numeric(observed["coef"], errors="coerce").abs().max())
                if observed is not None and not observed.empty
                else np.nan
            )
            if pd.notna(obs_max):
                ax.axvline(obs_max, color="#d62728", linewidth=1.5)
            ax.set_title(t_name.capitalize())
            ax.set_xlabel("Max |Partner Effect| (Placebo)")
            _apply_robust_style(ax)
            if pd.notna(obs_max):
                null_vals = sub["max_abs_partner"].dropna()
                if not null_vals.empty:
                    p95 = float(np.percentile(null_vals, 95))
                    summary_rows.append(
                        {
                            "check_name": "R6_placebo",
                            "target": t_name,
                            "metric_name": "null_p95_max_abs",
                            "baseline": obs_max,
                            "alt": p95,
                            "delta": obs_max - p95,
                            "note": "placebo",
                        }
                    )
        fig.suptitle("Placebo Test: Null Distribution vs Observed")
        fig.tight_layout()
        base = out_dir_path / "q4_robust_placebo"
        paths = _save_robust_fig(fig, base)
        plt.close(fig)
        outputs["q4_robust_placebo_png"] = paths["png"]
        outputs["q4_robust_placebo_pdf"] = paths["pdf"]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir_path / "q4_robustness_summary.csv"
    write_csv(summary_df, str(summary_path))
    outputs["q4_robustness_summary"] = str(summary_path)

    logger.info("Q4 robustness outputs saved to %s", out_dir)
    return outputs
