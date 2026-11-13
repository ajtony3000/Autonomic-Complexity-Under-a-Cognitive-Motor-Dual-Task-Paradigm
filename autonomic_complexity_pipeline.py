#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomic Complexity Under a Cognitive-Motor Dual-Task Paradigm:
End-to-End ECG Signal Processing, BIDS Phenotype Harmonization,
Hierarchical Linear Mixed-Effects Modeling (LMEM), and Q1 Visualization Pipeline.

Dataset: OpenNeuro ds007554 ("CMx7-MM", Ajra et al., 2026)
Authors: Aswin Tony Joseph (GMC Thiruvananthapuram), Dr. Rahul Venugopal (NIMHANS)
Target Journal: Autonomic Neuroscience: Basic and Clinical (Elsevier)
License: MIT
"""

import logging
import subprocess
import traceback
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ==============================================================================
# 0. GLOBAL CONFIGURATION & HEADLESS MATPLOTLIB GUARDS
# ==============================================================================
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Prevent memory overflow during high-density 1,882.45 Hz ECG rendering
matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.use("Agg")

# Standardized Q1 Journal Typography (Elsevier / Nature Standard)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Physiological & Experimental Constants
SAMPLING_RATE: float = 1882.45
SQI_THRESHOLD: float = 0.50
RR_MIN_MS: float = 300.0   # 200 bpm ceiling
RR_MAX_MS: float = 1500.0  # 40 bpm floor
BAD_RR_THRESHOLD_PCT: float = 10.0

TASK_ORDER: List[str] = [
    "passivemotor",
    "activemotor",
    "motorimagery",
    "mentalarithmetic",
    "nback",
    "nbackarithmetic",
    "full",
]

TASK_LABELS_MULTILINE: List[str] = [
    "Passive\nMotor",
    "Active\nMotor",
    "Motor\nImagery",
    "Mental\nArithmetic",
    "N-Back\n(2-Back)",
    "N-Back +\nArithmetic",
    "Full\nDual-Task",
]

SESSIONS: List[str] = ["ses-01", "ses-02", "ses-03"]
SESSION_NAMES: List[str] = ["Session 1", "Session 2", "Session 3"]
OKABE_ITO_PALETTE: List[str] = ["#3B6FB6", "#2A9D8F", "#D95F02"]

TARGET_METRICS: List[str] = [
    "HRV_RMSSD",
    "HRV_SDNN",
    "HRV_SD1SD2",
    "HRV_CSI",
    "HRV_DFA_alpha1",
    "mean_resp_rate",
    "HRV_SampEn",
]


def setup_directories(base_dir: Path) -> Dict[str, Path]:
    """Initialize directory structure and configure file/stream logging."""
    paths = {
        "base": base_dir,
        "physio": base_dir / "ds007554-physio",
        "phenotype": base_dir / "ds007554-phenotype",
        "results": base_dir / "Pipeline_Results",
        "qc_plots": base_dir / "Pipeline_Results" / "QC_Plots",
        "figures": base_dir / "Pipeline_Results" / "Publication_Figures",
        "tables": base_dir / "Pipeline_Results" / "Statistical_Tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(paths["results"] / "pipeline_run_log.txt"),
        ],
    )
    return paths


# ==============================================================================
# 1. OPENNEURO ds007554 DATA RETRIEVAL MODULE
# ==============================================================================
def download_openneuro_ds007554(paths: Dict[str, Path]) -> None:
    """
    Download compressed ECG telemetry and phenotype TSV files from OpenNeuro S3.
    Requires `awscli` installed in the environment (`pip install awscli`).
    """
    log = logging.getLogger(__name__)
    log.info("Syncing ECG telemetry from s3://openneuro.org/ds007554/ ...")

    cmd_physio = [
        "aws", "s3", "sync", "--no-sign-request",
        "s3://openneuro.org/ds007554/",
        str(paths["physio"]),
        "--exclude", "*",
        "--include", "*_recording-ECG_physio.tsv.gz",
    ]
    subprocess.run(cmd_physio, check=False)

    # Download core BIDS phenotype and participant tables via direct HTTPS
    s3_base = "https://s3.amazonaws.com/openneuro.org/ds007554"
    files_to_fetch = {
        "participants.tsv": f"{s3_base}/participants.tsv",
        "kss_values.tsv": f"{s3_base}/phenotype/kss_values.tsv",
        "cognitive_load.tsv": f"{s3_base}/phenotype/cognitive_load.tsv",
    }
    for fname, url in files_to_fetch.items():
        dest = paths["phenotype"] / fname
        if not dest.exists():
            log.info("Downloading %s ...", fname)
            urllib.request.urlretrieve(url, dest)


# ==============================================================================
# 2. AUTOMATED ECG -> HRV & EDR SIGNAL PROCESSING PIPELINE
# ==============================================================================
def process_ecg_repository(paths: Dict[str, Path]) -> pd.DataFrame:
    """
    Execute the 9-stage NeuroKit2 ECG preprocessing, SQI screening, Nabian (2018)
    artifact-corrected R-peak detection, RR bounding (300-1500 ms, <=10% bad),
    and ECG-Derived Respiration (EDR) extraction across all BIDS recordings.
    """
    log = logging.getLogger(__name__)
    filelist = sorted(paths["physio"].glob("**/*_recording-ECG_physio.tsv.gz"))

    if not filelist:
        log.error("No ECG .tsv.gz files found under %s", paths["physio"])
        raise FileNotFoundError(f"No ECG files in {paths['physio']}")

    log.info("Found %d ECG recording(s) to process.", len(filelist))
    masterlist: List[pd.DataFrame] = []

    for file_no, file_path in enumerate(filelist, start=1):
        fname = file_path.name
        parts = fname.split("_")

        sub_id = next((p for p in parts if p.startswith("sub-")), "Unknown")
        session_id = next((p for p in parts if p.startswith("ses-")), "Unknown")
        task_name = next(
            (p for p in parts if p.startswith("task-")), "Unknown"
        ).replace("task-", "")

        log.info(
            "[%d/%d] Processing: %s | %s | %s",
            file_no, len(filelist), sub_id, session_id, task_name
        )

        try:
            df_raw = pd.read_csv(
                file_path, sep="\t", compression="gzip", header=None
            )
            ecg_raw_signal = df_raw.iloc[:, 0].values

            # 1. Automated electrode polarity check & inversion
            ecg_signal, is_inverted = nk.ecg_invert(
                ecg_raw_signal, sampling_rate=SAMPLING_RATE
            )

            # 2. Primary bandpass filtering & initial QRS detection
            signals, ecg_info = nk.ecg_process(
                ecg_signal, sampling_rate=SAMPLING_RATE
            )

            # 3. Continuous Signal Quality Index (SQI) gate
            quality_scores = nk.ecg_quality(
                signals["ECG_Clean"],
                rpeaks=ecg_info["ECG_R_Peaks"],
                sampling_rate=SAMPLING_RATE,
            )
            mean_sqi = float(np.mean(quality_scores))

            if mean_sqi < SQI_THRESHOLD:
                log.warning(
                    "SQI %.3f < %.2f threshold — skipping noisy file: %s",
                    mean_sqi, SQI_THRESHOLD, fname
                )
                continue

            # 4. Nabian et al. (2018) R-peak detection with artifact correction
            peaks_info = nk.ecg_peaks(
                signals["ECG_Clean"],
                correct_artifacts=True,
                method="nabian2018",
                sampling_rate=SAMPLING_RATE,
            )

            # Overwrite cached peak arrays so downstream functions use corrected peaks
            ecg_info["ECG_R_Peaks"] = peaks_info["ECG_R_Peaks"]
            signals["ECG_R_Peaks"] = 0
            signals.loc[ecg_info["ECG_R_Peaks"], "ECG_R_Peaks"] = 1

            # 5. Physiological RR interval bounding (300 ms to 1,500 ms; 40-200 bpm)
            rr_ms = (
                np.diff(ecg_info["ECG_R_Peaks"]) / SAMPLING_RATE * 1000.0
            )
            bad_mask = (rr_ms < RR_MIN_MS) | (rr_ms > RR_MAX_MS)
            n_bad = int(np.sum(bad_mask))
            pct_bad = (
                100.0 * n_bad / len(rr_ms) if len(rr_ms) > 0 else 0.0
            )
            data_quality_flag = (
                "POOR" if pct_bad > BAD_RR_THRESHOLD_PCT else "OK"
            )

            # 6. ECG-Derived Respiration (EDR) extraction (van Gent et al., 2019)
            ecg_rate_signal = nk.ecg_rate(
                peaks_info,
                sampling_rate=SAMPLING_RATE,
                desired_length=len(ecg_signal),
            )
            edr_signal = nk.ecg_rsp(
                ecg_rate_signal, sampling_rate=SAMPLING_RATE
            )
            rsp_signals, _ = nk.rsp_process(
                edr_signal, sampling_rate=SAMPLING_RATE
            )
            mean_rsp_rate = float(np.mean(rsp_signals["RSP_Rate"]))
            sd_rsp_rate = float(np.std(rsp_signals["RSP_Rate"], ddof=1))

            # 7. Multi-domain HRV computation & Summary QA plot export
            hrv_df = nk.hrv(
                ecg_info, sampling_rate=SAMPLING_RATE, show=True
            )
            hrv_fig = plt.gcf()
            hrv_fig.set_size_inches(15, 8)
            hrv_fig.savefig(
                paths["qc_plots"]
                / f"{sub_id}_{session_id}_{task_name}_hrv_summary.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")

            # 8. Append quality control and respiratory metadata
            hrv_df["Subject"] = sub_id
            hrv_df["Session"] = session_id
            hrv_df["Task"] = task_name
            hrv_df["is_inverted"] = is_inverted
            hrv_df["mean_resp_rate"] = mean_rsp_rate
            hrv_df["sd_resp_rate"] = sd_rsp_rate
            hrv_df["n_rpeaks"] = len(ecg_info["ECG_R_Peaks"])
            hrv_df["mean_sqi"] = round(mean_sqi, 4)
            hrv_df["pct_rr_bad"] = round(pct_bad, 2)
            hrv_df["data_quality_flag"] = data_quality_flag

            # 9. Export morphological P-Q-R-S-T alignment QA plot
            nk.ecg_plot(signals, info=ecg_info)
            qual_fig = plt.gcf()
            qual_fig.set_size_inches(15, 8)
            qual_fig.savefig(
                paths["qc_plots"]
                / f"{sub_id}_{session_id}_{task_name}_quality_check.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close("all")

            masterlist.append(hrv_df)

        except Exception:
            log.error("FAILED for %s:\n%s", fname, traceback.format_exc())
            plt.close("all")
            continue

    if not masterlist:
        raise RuntimeError("No recordings were processed successfully.")

    master_df = pd.concat(masterlist, axis=0, ignore_index=True)
    id_cols = [
        "Subject", "Session", "Task", "data_quality_flag", "mean_sqi",
        "n_rpeaks", "pct_rr_bad", "is_inverted", "mean_resp_rate", "sd_resp_rate",
    ]
    other_cols = [c for c in master_df.columns if c not in id_cols]
    master_df = master_df[id_cols + other_cols]

    output_csv = paths["results"] / "Autonomic_Complexity_Results_CCS_Validated.csv"
    master_df.to_csv(output_csv, index=False)
    log.info("Saved validated HRV feature matrix to: %s", output_csv)
    return master_df


# ==============================================================================
# 3. PHENOTYPE STRING HARMONIZATION & RELATIONAL MERGING
# ==============================================================================
def merge_hrv_and_phenotypes(
    hrv_csv_path: Path,
    participants_tsv: Path,
    kss_tsv: Path,
    cogload_tsv: Path,
    output_master_csv: Path,
) -> pd.DataFrame:
    """
    Filter QC-passed HRV observations ('OK'), harmonize BIDS phenotype naming
    discrepancies ('sess-' -> 'ses-'; 'arithmetic' -> 'mentalarithmetic'), and
    merge demographics, KSS baseline sleepiness, and post-task cognitive load.
    """
    df_hrv = pd.read_csv(hrv_csv_path)
    clean_df = df_hrv[df_hrv["data_quality_flag"] == "OK"].copy()
    clean_df["HRV_SampEn"] = clean_df["HRV_SampEn"].replace(
        [np.inf, -np.inf], np.nan
    )
    clean_df = clean_df.dropna(subset=["HRV_SampEn"]).copy()

    # Exclude telemetry dropout subject sub-024 if any partial block remained
    clean_df = clean_df[clean_df["Subject"] != "sub-024"].copy()

    # 1. Merge participants.tsv
    df_demo = pd.read_csv(participants_tsv, sep="\t").rename(
        columns={"participant_id": "Subject"}
    )
    final_df = pd.merge(clean_df, df_demo, on="Subject", how="left")

    # 2. Reshape & normalize KSS sleepiness scores
    df_kss = pd.read_csv(kss_tsv, sep="\t").rename(
        columns={"participant_id": "Subject"}
    )
    df_kss_long = pd.melt(
        df_kss,
        id_vars=["Subject"],
        var_name="Session_KSS",
        value_name="KSS_Score",
    )
    df_kss_long["Session"] = df_kss_long["Session_KSS"].apply(
        lambda x: str(x).split("_")[0].replace("sess-", "ses-")
    )
    df_kss_long["KSS_Score"] = pd.to_numeric(
        df_kss_long["KSS_Score"], errors="coerce"
    )
    df_kss_long = df_kss_long[["Subject", "Session", "KSS_Score"]]

    # 3. Reshape & normalize post-task Cognitive Load ratings
    df_cog = pd.read_csv(cogload_tsv, sep="\t").rename(
        columns={"participant_id": "Subject"}
    )
    df_cog_long = pd.melt(
        df_cog,
        id_vars=["Subject"],
        var_name="Session_Task",
        value_name="Cognitive_Load_Score",
    )
    df_cog_long["Session"] = df_cog_long["Session_Task"].apply(
        lambda x: str(x).split("_")[0].replace("sess-", "ses-")
    )
    df_cog_long["Task"] = df_cog_long["Session_Task"].apply(
        lambda x: "".join(str(x).split("_")[1:]).lower()
    )
    df_cog_long["Task"] = df_cog_long["Task"].replace(
        {"arithmetic": "mentalarithmetic"}
    )
    df_cog_long["Cognitive_Load_Score"] = pd.to_numeric(
        df_cog_long["Cognitive_Load_Score"], errors="coerce"
    )
    df_cog_long = df_cog_long[
        ["Subject", "Session", "Task", "Cognitive_Load_Score"]
    ]

    # 4. Grand Relational Merge
    master_df = pd.merge(
        final_df, df_kss_long, on=["Subject", "Session"], how="left"
    )
    master_df = pd.merge(
        master_df,
        df_cog_long,
        on=["Subject", "Session", "Task"],
        how="left",
    )

    redundant_cols = [c for c in master_df.columns if "sess-" in c]
    master_df = master_df.drop(columns=redundant_cols, errors="ignore")
    master_df.to_csv(output_master_csv, index=False)
    return master_df


# ==============================================================================
# 4. DESCRIPTIVE TABLES, NORMALITY SCREENING & UNIVARIATE BASELINE ANALYSIS
# ==============================================================================
def run_descriptive_and_univariate_analyses(
    master_df: pd.DataFrame,
    participants_tsv: Path,
    paths: Dict[str, Path],
) -> None:
    """
    Generate Supplementary Table S1 (Task x Session SampEn summary),
    Supplementary Table S2 (Shapiro-Wilk & Levene screening), and
    Supplementary Table S3 + Figure S1 (Univariate Age & Sex vs. Baseline SampEn).
    """
    # --- A. Supplementary Table S1: Task x Session Descriptive Statistics ---
    stats_table = (
        master_df.groupby(["Task", "Session"])["HRV_SampEn"]
        .agg(
            N="count",
            Mean="mean",
            SD="std",
            Median="median",
            IQR=lambda x: x.quantile(0.75) - x.quantile(0.25),
            Min="min",
            Max="max",
        )
        .reset_index()
    )
    stats_table["Task"] = pd.Categorical(
        stats_table["Task"], categories=TASK_ORDER, ordered=True
    )
    stats_table = stats_table.sort_values(by=["Task", "Session"])
    stats_table.to_csv(
        paths["tables"] / "Supplementary_Table_S1_Task_Session_SampEn.csv",
        index=False,
    )

    # --- B. Supplementary Table S2: Task-wise Shapiro-Wilk & Levene's Test ---
    normality_rows = []
    groups = []
    for task in TASK_ORDER:
        sub_series = master_df[master_df["Task"] == task]["HRV_SampEn"].dropna()
        groups.append(sub_series.values)
        w_stat, p_val = stats.shapiro(sub_series)
        normality_rows.append(
            {"Task": task, "N": len(sub_series), "Shapiro_W": w_stat, "p_value": p_val}
        )
    lev_stat, lev_p = stats.levene(*groups)
    normality_rows.append(
        {"Task": "Levene_Homoscedasticity_All_Tasks", "N": len(master_df), "Shapiro_W": lev_stat, "p_value": lev_p}
    )
    pd.DataFrame(normality_rows).to_csv(
        paths["tables"] / "Supplementary_Table_S2_Normality_Levene.csv",
        index=False,
    )

    # --- C. Supplementary Table S3 & Figure S1: Univariate Age & Sex Baseline ---
    df_demo = pd.read_csv(participants_tsv, sep="\t")
    baseline_subj = (
        master_df[master_df["Task"] == "passivemotor"]
        .groupby("Subject")["HRV_SampEn"]
        .mean()
        .reset_index()
        .rename(columns={"HRV_SampEn": "Baseline_SampEn"})
    )
    df_merged = pd.merge(
        df_demo, baseline_subj, left_on="participant_id", right_on="Subject", how="inner"
    )
    df_merged["age"] = pd.to_numeric(df_merged["age"], errors="coerce")
    df_merged["sex"] = df_merged["sex"].astype(str).str.strip().str.upper()

    # Age vs. Baseline SampEn (Complete cases, N = 16)
    age_sub = df_merged.dropna(subset=["age", "Baseline_SampEn"]).copy()
    r_p, p_p = stats.pearsonr(age_sub["age"], age_sub["Baseline_SampEn"])
    rho_s, p_s = stats.spearmanr(age_sub["age"], age_sub["Baseline_SampEn"])
    slope, intercept, _, p_lin, std_err = stats.linregress(
        age_sub["age"], age_sub["Baseline_SampEn"]
    )

    # Biological Sex vs. Baseline SampEn (Males = 7, Females = 16)
    males = df_merged[df_merged["sex"] == "M"]["Baseline_SampEn"].dropna()
    females = df_merged[df_merged["sex"] == "F"]["Baseline_SampEn"].dropna()
    t_stat, p_ttest = stats.ttest_ind(males, females, equal_var=False)
    u_stat, p_mann = stats.mannwhitneyu(males, females, alternative="two-sided")

    n1, n2 = len(males), len(females)
    s1, s2 = np.var(males, ddof=1), np.var(females, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    cohens_d = (np.mean(males) - np.mean(females)) / pooled_sd

    print("=" * 60)
    print(f"UNIVARIATE ANALYSIS: AGE vs BASELINE SAMPEN (N = {len(age_sub)})")
    print("=" * 60)
    print(f"Mean Age: {age_sub['age'].mean():.2f} ± {age_sub['age'].std():.2f} years")
    print(f"Pearson r    : {r_p:.4f} (p = {p_p:.4f})")
    print(f"Spearman rho : {rho_s:.4f} (p = {p_s:.4f})")
    print(f"Linear Slope : β = {slope:.4f} per year (SE = {std_err:.4f}, p = {p_lin:.4f})")
    print("\n" + "=" * 60)
    print(f"UNIVARIATE ANALYSIS: SEX vs BASELINE SAMPEN (Males: {n1}, Females: {n2})")
    print("=" * 60)
    print(f"Males Baseline SampEn   : {males.mean():.4f} ± {males.std():.4f}")
    print(f"Females Baseline SampEn : {females.mean():.4f} ± {females.std():.4f}")
    print(f"Difference (Male - Fem) : {males.mean() - females.mean():.4f}")
    print(f"Welch's t-test          : t = {t_stat:.4f}, p = {p_ttest:.4f}")
    print(f"Mann-Whitney U test     : U = {u_stat:.1f}, p = {p_mann:.4f}")
    print(f"Cohen's d effect size   : d = {cohens_d:.4f}\n")

    # Render Supplementary Figure S1 (Age & Sex Univariate Plot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    sns.regplot(
        data=age_sub,
        x="age",
        y="Baseline_SampEn",
        ax=axes[0],
        color="#1f77b4",
        scatter_kws={"s": 60, "alpha": 0.8},
        line_kws={"linewidth": 2},
    )
    axes[0].set_title(
        f"Age vs Baseline SampEn (r = {r_p:.3f}, p = {p_p:.3f})",
        fontsize=12, fontweight="bold"
    )
    axes[0].set_xlabel("Age (years)", fontsize=11)
    axes[0].set_ylabel("Baseline Sample Entropy (Passive Motor)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    sns.boxplot(
        data=df_merged,
        x="sex",
        y="Baseline_SampEn",
        hue="sex",
        order=["M", "F"],
        ax=axes[1],
        palette={"M": "#1f77b4", "F": "#ff7f0e"},
        width=0.4,
        legend=False,
        boxprops={"alpha": 0.7},
    )
    sns.stripplot(
        data=df_merged,
        x="sex",
        y="Baseline_SampEn",
        order=["M", "F"],
        ax=axes[1],
        color="black",
        size=6,
        jitter=0.15,
        alpha=0.8,
    )
    axes[1].set_title(
        f"Sex vs Baseline SampEn (d = {cohens_d:.2f}, p = {p_ttest:.3f})",
        fontsize=12, fontweight="bold"
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Male", "Female"])
    axes[1].set_xlabel("Biological Sex", fontsize=11)
    axes[1].set_ylabel("Baseline Sample Entropy (Passive Motor)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "SuppFig_S1_Univariate_Baseline_Age_Sex.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")


# ==============================================================================
# 5. MULTI-METRIC HIERARCHICAL LMEM & VARIANCE PARTITIONING PIPELINE
# ==============================================================================
def prepare_lmem_covariates(
    df: pd.DataFrame, metric: str, impute_age: bool = True
) -> pd.DataFrame:
    """
    Prepare clean model DataFrame with Missing-Indicator age adjustment (or
    complete-case filtering), mean-centered KSS, and binary biological sex coding.
    """
    task_df = df.copy()
    task_df[metric] = pd.to_numeric(task_df[metric], errors="coerce")
    task_df = task_df.replace([np.inf, -np.inf], np.nan)
    task_df["age_numeric"] = pd.to_numeric(task_df["age"], errors="coerce")

    base_cols = [metric, "sex", "Task", "Session", "KSS_Score"]
    if not impute_age:
        base_cols.append("age_numeric")

    task_df = task_df.dropna(subset=base_cols).copy()

    if impute_age:
        task_df["age_missing"] = task_df["age_numeric"].isna().astype(int)
        mean_obs_age = task_df.loc[
            task_df["age_missing"] == 0, "age_numeric"
        ].mean()
        task_df["age_c"] = (task_df["age_numeric"] - mean_obs_age).fillna(0.0)
    else:
        task_df["age_c"] = (
            task_df["age_numeric"] - task_df["age_numeric"].mean()
        )

    task_df["kss_c"] = task_df["KSS_Score"] - task_df["KSS_Score"].mean()
    task_df["sex_male"] = (
        task_df["sex"].astype(str).str.strip().str.upper() == "M"
    ).astype(int)
    return task_df


def run_7_metric_lmem_comparison(
    master_df: pd.DataFrame, paths: Dict[str, Path]
) -> Tuple[Optional[object], pd.DataFrame]:
    """
    Execute the Primary Task LMEM across all 7 target physiological metrics,
    compute VIF and Shapiro-Wilk residual diagnostics, and extract 3-level
    partitioned variance components (VarF, VarR_subj, VarR_sess, VarE, R2m, R2c, ICC).
    """
    variance_summary_rows = []
    primary_sampen_result = None
    primary_sampen_df = None

    for idx, metric in enumerate(TARGET_METRICS, start=1):
        if metric not in master_df.columns:
            continue

        task_df = prepare_lmem_covariates(master_df, metric, impute_age=True)
        formula_task = (
            f"{metric} ~ C(Task, Treatment('passivemotor')) + "
            "age_c + age_missing + sex_male + kss_c"
        )

        # 1. Primary Single-Basket Nested Specification (Main Text Table 3)
        model_primary = smf.mixedlm(
            formula=formula_task,
            data=task_df,
            groups=task_df["Subject"],
            vc_formula={"Session_Block": "0 + C(Session)"},
        )
        res_primary = model_primary.fit(method="powell")

        # 2. Explicit 3-Level Partitioned Specification (Subject + Session_in_Subj)
        model_part = smf.mixedlm(
            formula=formula_task,
            data=task_df,
            groups=task_df["Subject"],
            re_formula="~1",
            vc_formula={"Session_in_Subj": "0 + C(Session)"},
        )
        res_part = model_part.fit(method="powell")

        # Variance decomposition on the 3-level partitioned model
        var_subj = (
            0.0 if res_part.cov_re.empty else float(res_part.cov_re.iloc[0, 0])
        )
        var_sess = (
            float(res_part.vcomp.sum())
            if (hasattr(res_part, "vcomp") and res_part.vcomp is not None)
            else 0.0
        )
        var_rand = var_subj + var_sess
        var_resid = float(res_part.scale)

        linear_pred = res_part.model.exog @ res_part.fe_params
        var_fixed = float(np.var(linear_pred))
        var_total = var_fixed + var_rand + var_resid

        r2_m = var_fixed / var_total
        r2_c = (var_fixed + var_rand) / var_total
        icc_tot = var_rand / (var_rand + var_resid)

        # Residual normality & VIF check on primary specification
        w_stat, p_shapiro = stats.shapiro(res_primary.resid)
        _, X_mat = dmatrices(formula_task, data=task_df, return_type="dataframe")
        max_vif = max(
            variance_inflation_factor(X_mat.values, i)
            for i in range(1, X_mat.shape[1])
        )

        variance_summary_rows.append({
            "Metric": metric,
            "N_Subjects": task_df["Subject"].nunique(),
            "N_Obs": len(task_df),
            "VarF_Fixed": round(var_fixed, 6),
            "VarR_Subject": round(var_subj, 6),
            "VarR_Session": round(var_sess, 6),
            "VarE_Residual": round(var_resid, 6),
            "VarT_Total": round(var_total, 6),
            "R2_Marginal": round(r2_m, 4),
            "R2_Conditional": round(r2_c, 4),
            "ICC_Total": round(icc_tot, 4),
            "Shapiro_W": round(w_stat, 4),
            "Shapiro_p": round(p_shapiro, 4),
            "Max_VIF": round(max_vif, 4),
        })

        print("=" * 80)
        print(f" [{idx}/7] LMEM & VARIANCE SUMMARY FOR: {metric}")
        print("=" * 80)
        print(res_primary.summary())
        print(
            f"Shapiro-Wilk Residual W = {w_stat:.4f} (p = {p_shapiro:.4f}) | "
            f"R2m = {r2_m * 100:.2f}% | R2c = {r2_c * 100:.2f}% | ICC = {icc_tot * 100:.1f}%"
        )

        if metric == "HRV_SampEn":
            primary_sampen_result = res_primary
            primary_sampen_df = task_df

    var_df = pd.DataFrame(variance_summary_rows)
    var_df.to_csv(
        paths["tables"] / "Supplementary_Table_S5_7Metric_Variance_R2_ICC.csv",
        index=False,
    )
    return primary_sampen_result, primary_sampen_df


# ==============================================================================
# 6. SECONDARY COGNITIVE LOAD LMEM & SENSITIVITY ANALYSES
# ==============================================================================
def run_cognitive_load_and_sensitivity_models(
    master_df: pd.DataFrame, paths: Dict[str, Path]
) -> object:
    """
    Execute:
      1. Primary Task Complete-Case LMEM (No age imputation, N = 18, 297 obs)
      2. Secondary Cognitive Load LMEM (Full cohort, N = 29, 449 obs)
      3. Secondary Cognitive Load Outlier-Removed LMEM (N = 29, 448 obs)
      4. Secondary Cognitive Load Complete-Case LMEM (N = 18, 297 obs & 296 obs)
    """
    # --- 1. Primary Task Complete-Case LMEM (No Imputation) ---
    no_imp_task_df = prepare_lmem_covariates(
        master_df, "HRV_SampEn", impute_age=False
    )
    formula_no_imp = (
        "HRV_SampEn ~ C(Task, Treatment('passivemotor')) + "
        "age_c + sex_male + kss_c"
    )
    res_task_no_imp = smf.mixedlm(
        formula=formula_no_imp,
        data=no_imp_task_df,
        groups=no_imp_task_df["Subject"],
        vc_formula={"Session_Block": "0 + C(Session)"},
    ).fit(method="powell")
    print("\n--- PRIMARY TASK LMEM (Complete Cases, No Imputation, N=18) ---")
    print(res_task_no_imp.summary())

    # --- 2. Secondary Cognitive Load LMEM (Full Cohort, N = 29, 449 obs) ---
    load_df = master_df.dropna(
        subset=["HRV_SampEn", "Cognitive_Load_Score", "sex", "Session", "KSS_Score"]
    ).copy()
    load_df = prepare_lmem_covariates(load_df, "HRV_SampEn", impute_age=True)
    load_df["cogload_c"] = (
        load_df["Cognitive_Load_Score"] - load_df["Cognitive_Load_Score"].mean()
    )

    formula_load = "HRV_SampEn ~ cogload_c + age_c + age_missing + sex_male + kss_c"
    res_load = smf.mixedlm(
        formula=formula_load,
        data=load_df,
        groups=load_df["Subject"],
        vc_formula={"Session_Block": "0 + C(Session)"},
    ).fit(method="powell")
    print("\n--- SECONDARY COGNITIVE LOAD LMEM (Full Cohort, N=29, 449 obs) ---")
    print(res_load.summary())

    # --- 3. Outlier Screening (> 3 SD) on Full Cognitive Load Model ---
    z_resid = (res_load.resid - res_load.resid.mean()) / res_load.resid.std()
    outlier_mask = np.abs(z_resid) > 3.0
    if outlier_mask.sum() > 0:
        robust_df = load_df[~outlier_mask].copy()
        robust_df["cogload_c"] = (
            robust_df["Cognitive_Load_Score"]
            - robust_df["Cognitive_Load_Score"].mean()
        )
        res_load_robust = smf.mixedlm(
            formula=formula_load,
            data=robust_df,
            groups=robust_df["Subject"],
            vc_formula={"Session_Block": "0 + C(Session)"},
        ).fit(method="powell")
        print("\n--- ROBUST COGNITIVE LOAD LMEM (Outlier Removed, N=29, 448 obs) ---")
        print(res_load_robust.summary())

    # --- 4. Complete-Case Cognitive Load LMEM (N = 18, 297 obs & 296 obs) ---
    no_imp_load_df = master_df.dropna(
        subset=["HRV_SampEn", "Cognitive_Load_Score", "age", "sex", "Session", "KSS_Score"]
    ).copy()
    no_imp_load_df = prepare_lmem_covariates(
        no_imp_load_df, "HRV_SampEn", impute_age=False
    )
    no_imp_load_df["cogload_c"] = (
        no_imp_load_df["Cognitive_Load_Score"]
        - no_imp_load_df["Cognitive_Load_Score"].mean()
    )
    formula_load_no_imp = "HRV_SampEn ~ cogload_c + age_c + sex_male + kss_c"
    res_load_no_imp = smf.mixedlm(
        formula=formula_load_no_imp,
        data=no_imp_load_df,
        groups=no_imp_load_df["Subject"],
        vc_formula={"Session_Block": "0 + C(Session)"},
    ).fit(method="powell")
    print("\n--- COMPLETE-CASE COGNITIVE LOAD LMEM (No Imputation, N=18, 297 obs) ---")
    print(res_load_no_imp.summary())

    return res_load


# ==============================================================================
# 7. PUBLICATION-READY Q1 VISUALIZATION SUITE
# ==============================================================================
def generate_publication_figures(
    task_df: pd.DataFrame,
    res_task: object,
    res_load: object,
    paths: Dict[str, Path],
) -> None:
    """
    Generate all main-text and supplementary figures at 600 DPI PNG + Vector PDF:
      - Fig 2: Synchronized Violin + Boxplot + Jittered Dot Plot (Table 2 Replacement)
      - Fig 4: Primary Task LMEM 3-Panel Residual Diagnostics
      - Fig 5: Forest Plot of Task Fixed Effects (beta ± 95% CI)
      - Fig 6: Task x Session Mean Sample Entropy Heatmap
      - Supp Fig S2: Secondary Cognitive Load LMEM Residual Diagnostics
      - Supp Fig S3: Within-Subject Spaghetti Trajectory Plot
    """
    # --- A. Main Figure 2: Synchronized Violin + Box + Jittered Dots ---
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    offsets = [-0.26, 0.0, 0.26]
    violin_width = 0.22
    box_width = 0.08
    np.random.seed(42)

    for s_idx, (ses, color) in enumerate(zip(SESSIONS, OKABE_ITO_PALETTE)):
        for t_idx, task in enumerate(TASK_ORDER):
            sub_data = (
                task_df[(task_df["Task"] == task) & (task_df["Session"] == ses)][
                    "HRV_SampEn"
                ]
                .dropna()
                .values
            )
            if len(sub_data) == 0:
                continue

            pos = t_idx + offsets[s_idx]
            parts = ax.violinplot(
                sub_data,
                positions=[pos],
                widths=violin_width,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.28)
                pc.set_linewidth(1.0)
                pc.set_zorder(2)

            bp = ax.boxplot(
                sub_data,
                positions=[pos],
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                zorder=4,
            )
            for box in bp["boxes"]:
                box.set(
                    facecolor="white",
                    edgecolor="#222222",
                    linewidth=1.1,
                    alpha=0.9,
                )
            for median in bp["medians"]:
                median.set(color="#D62728", linewidth=1.8)
            for whisker in bp["whiskers"]:
                whisker.set(color="#222222", linewidth=1.0)
            for cap in bp["caps"]:
                cap.set(color="#222222", linewidth=1.0)

            jitter = np.random.uniform(-0.045, 0.045, size=len(sub_data))
            ax.scatter(
                np.repeat(pos, len(sub_data)) + jitter,
                sub_data,
                color=color,
                edgecolor="#111111",
                linewidth=0.5,
                s=22,
                alpha=0.80,
                zorder=3,
            )

    for x_sep in np.arange(0.5, len(TASK_ORDER) - 0.5, 1.0):
        ax.axvline(x_sep, color="#E0E0E0", linestyle="-", linewidth=0.8, zorder=1)

    baseline_mean = task_df[task_df["Task"] == "passivemotor"]["HRV_SampEn"].mean()
    ax.axhline(
        baseline_mean, color="#555555", linestyle="--", linewidth=1.1, alpha=0.7, zorder=1
    )

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_LABELS_MULTILINE, fontweight="bold")
    ax.set_xlim(-0.6, len(TASK_ORDER) - 0.4)
    ax.set_ylim(0.45, 2.45)
    ax.set_ylabel("Heart Rate Sample Entropy (SampEn)")
    ax.set_xlabel("Experimental Task Condition", labelpad=10)
    ax.yaxis.grid(True, linestyle=":", color="#CCCCCC", alpha=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(
            facecolor=OKABE_ITO_PALETTE[i],
            edgecolor="#222222",
            alpha=0.65,
            label=SESSION_NAMES[i],
        )
        for i in range(len(SESSIONS))
    ]
    legend_handles.append(
        plt.Line2D(
            [0], [0],
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            label=f"Passive Motor Baseline Mean ({baseline_mean:.2f})",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "Figure2_Q1_SampEn_Distribution.png",
        dpi=600, bbox_inches="tight"
    )
    plt.savefig(
        paths["figures"] / "Figure2_Q1_SampEn_Distribution.pdf",
        bbox_inches="tight"
    )
    plt.close("all")

    # --- B. Main Figure 4: Primary Task LMEM Residual Diagnostics ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)
    stats.probplot(res_task.resid, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot (Normality)", fontweight="bold")

    axes[1].scatter(res_task.fittedvalues, res_task.resid, alpha=0.5, color="#3B6FB6")
    axes[1].axhline(0, color="#D62728", linestyle="--", linewidth=1.8)
    axes[1].set_xlabel("Fitted Sample Entropy")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs. Fitted", fontweight="bold")

    axes[2].hist(res_task.resid, bins=20, edgecolor="black", color="#A6CEE3")
    axes[2].set_xlabel("Residuals")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Residual Distribution", fontweight="bold")

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "Figure4_LMEM_Diagnostics.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    # --- C. Main Figure 5: Forest Plot of Task Fixed Effects ---
    conf = res_task.conf_int()
    conf["coef"] = res_task.params
    task_stats = conf[conf.index.str.contains("Task")].copy()
    task_stats["Task_Name"] = task_stats.index.map(
        lambda x: x.replace("C(Task, Treatment('passivemotor'))[T.", "").replace("]", "")
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    y_pos = np.arange(len(task_stats))
    effects = task_stats["coef"].values
    ci_low = task_stats[0].values
    ci_high = task_stats[1].values

    ax.errorbar(
        effects,
        y_pos,
        xerr=[effects - ci_low, ci_high - effects],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=1.8,
        linewidth=2.2,
        color="#174EA6",
    )
    ax.axvline(
        0, color="#D62728", linestyle="--", linewidth=2.0,
        label="Reference Baseline (Passive Motor = 0)"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(task_stats["Task_Name"].values, fontweight="bold")
    ax.set_xlabel("Fixed-Effect Difference in Sample Entropy (β ± 95% CI)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.3, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "Figure5_ForestPlot_TaskEffects.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    # --- D. Main Figure 6: Task x Session Mean SampEn Heatmap ---
    heatmap_data = task_df.pivot_table(
        values="HRV_SampEn", index="Task", columns="Session", aggfunc="mean"
    ).reindex(TASK_ORDER)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        cbar_kws={"label": "Mean Sample Entropy (SampEn)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Recording Session")
    ax.set_ylabel("Experimental Task Condition")
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "Figure6_Heatmap_Task_Session.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    # --- E. Supplementary Figure S2: Cognitive Load LMEM Residual Diagnostics ---
    std_res_load = (
        (res_load.resid - res_load.resid.mean()) / res_load.resid.std()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    sns.residplot(
        x=res_load.fittedvalues,
        y=std_res_load,
        lowess=True,
        scatter_kws={"alpha": 0.5, "color": "steelblue"},
        line_kws={"color": "darkred", "lw": 2},
        ax=axes[0],
    )
    axes[0].set_title("Homoscedasticity: Residuals vs. Fitted", fontweight="bold")
    axes[0].set_xlabel("Fitted Values (Predicted Sample Entropy)")
    axes[0].set_ylabel("Standardized Residuals")

    sm.qqplot(
        std_res_load,
        line="45",
        fit=True,
        markerfacecolor="steelblue",
        markeredgecolor="steelblue",
        alpha=0.5,
        ax=axes[1],
    )
    axes[1].set_title("Normality: Q-Q Plot of Residuals", fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "SuppFig_S2_CognitiveLoad_Diagnostics.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    # --- F. Supplementary Figure S3: Within-Subject Spaghetti Plot ---
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    spaghetti_df = (
        task_df.groupby(["Subject", "Task"])["HRV_SampEn"].mean().reset_index()
    )
    for subject in spaghetti_df["Subject"].unique():
        sub_row = (
            spaghetti_df[spaghetti_df["Subject"] == subject]
            .set_index("Task")
            .reindex(TASK_ORDER)
            .dropna(subset=["HRV_SampEn"])
        )
        ax.plot(
            sub_row.index,
            sub_row["HRV_SampEn"],
            marker="o",
            alpha=0.25,
            linewidth=1.2,
            color="gray",
        )

    group_mean = (
        spaghetti_df.groupby("Task")["HRV_SampEn"].mean().reindex(TASK_ORDER)
    )
    ax.plot(
        group_mean.index,
        group_mean.values,
        marker="D",
        markersize=8,
        linewidth=3.5,
        color="darkred",
        label="Equal-Weighted Group Mean Trajectory",
    )
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_LABELS_MULTILINE, fontweight="bold")
    ax.set_ylabel("HRV Sample Entropy (SampEn)")
    ax.set_xlabel("Experimental Task Condition")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        paths["figures"] / "SuppFig_S3_Spaghetti_Trajectories.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close("all")


# ==============================================================================
# 8. MAIN EXECUTION ENTRY POINT
# ==============================================================================
def main(base_dir_str: str = "./content") -> None:
    """Run the complete end-to-end signal processing and LMEM analysis pipeline."""
    paths = setup_directories(Path(base_dir_str))
    hrv_csv = paths["results"] / "Autonomic_Complexity_Results_CCS_Validated.csv"

    # Step 1: Download & process raw ECG telemetry if validated CSV is not yet cached
    if not hrv_csv.exists():
        download_openneuro_ds007554(paths)
        process_ecg_repository(paths)

    # Step 2: Harmonize & merge BIDS phenotype spreadsheets
    master_csv = paths["results"] / "Master_HRV_Behavioral_Dataset.csv"
    master_df = merge_hrv_and_phenotypes(
        hrv_csv_path=hrv_csv,
        participants_tsv=paths["phenotype"] / "participants.tsv",
        kss_tsv=paths["phenotype"] / "kss_values.tsv",
        cogload_tsv=paths["phenotype"] / "cognitive_load.tsv",
        output_master_csv=master_csv,
    )

    # Step 3: Run descriptive tables, normality screening, and univariate Age/Sex tests
    run_descriptive_and_univariate_analyses(
        master_df=master_df,
        participants_tsv=paths["phenotype"] / "participants.tsv",
        paths=paths,
    )

    # Step 4: Execute the 7-metric LMEM comparison & 3-level variance partitioning
    res_task, task_df = run_7_metric_lmem_comparison(
        master_df=master_df, paths=paths
    )

    # Step 5: Execute Secondary Cognitive Load LMEM & Complete-Case/Outlier Sensitivity
    res_load = run_cognitive_load_and_sensitivity_models(
        master_df=master_df, paths=paths
    )

    # Step 6: Render Q1 Publication & Supplementary Figures
    if res_task is not None and task_df is not None:
        generate_publication_figures(
            task_df=task_df,
            res_task=res_task,
            res_load=res_load,
            paths=paths,
        )


if __name__ == "__main__":
    main()