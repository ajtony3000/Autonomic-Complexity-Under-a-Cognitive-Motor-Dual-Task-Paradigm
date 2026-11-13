# Autonomic Complexity Under a Cognitive-Motor Dual-Task Paradigm: Hierarchical LMEM & ECG-to-HRV Pipeline

**Authors:** Aswin Tony Joseph (Government Medical College, Thiruvananthapuram) and Dr. Rahul Venugopal (Centre for Consciousness Studies, Department of Neurophysiology, NIMHANS, Bengaluru).

This repository contains the end-to-end Python signal processing, phenotype harmonization, hierarchical Linear Mixed-Effects Modeling (LMEM), and visualization pipeline for the manuscript *"Autonomic Complexity Under a Cognitive-Motor Dual-Task Paradigm: A Hierarchical Linear Mixed-Effects Modeling of Heart Rate Sample Entropy"*. The workflow processes raw 1,882.45 Hz electrocardiographic (ECG) telemetry and behavioral ratings from the OpenNeuro ds007554 ("CMx7-MM") dataset across 7 cognitive-motor task conditions and 3 recording sessions (29 subjects, 450 validated observations).

## Pipeline Architecture

* **Automated BIDS Telemetry Ingestion:** Streams compressed ECG physiological files and behavioral phenotype tables directly from the OpenNeuro ds007554 AWS S3 bucket.


* **Polarity Standardization:** Automatically detects and corrects inverted electrode leads using `nk.ecg_invert()` prior to QRS complex extraction.


* **Bandpass Filtering & SQI Screening:** Cleans raw voltage traces via `nk.ecg_process()` and rejects recordings with mean `nk.ecg_quality()` scores below 0.50.


* **Artifact-Corrected R-Peak Detection:** Applies the Nabian et al. (2018) algorithm (`nk.ecg_peaks(correct_artifacts=True, method='nabian2018')`) to handle ectopic beats.


* **Physiological RR Bounding:** Filters inter-beat intervals outside 300 to 1,500 ms (40 to 200 bpm) and flags recordings with over 10% out-of-bounds intervals.


* **ECG-Derived Respiration (EDR):** Extracts continuous respiratory waveforms (`nk.ecg_rsp()`, `nk.rsp_process()`) to verify normocapnic breathing (0.15 to 0.40 Hz).


* **BIDS Phenotype Harmonization:** Aligns session prefixes and task labels across Karolinska Sleepiness Scale (KSS) and post-task cognitive load files.


* **Multi-Metric Nested LMEMs:** Fits 3-level REML mixed models (`smf.mixedlm()`, `method='powell'`) across 7 HRV and respiratory metrics with VIF and Shapiro-Wilk checks.


* **Variance Partitioning & Sensitivity:** Computes Nakagawa's marginal and conditional R-squared, Intraclass Correlation Coefficients (ICC), and complete-case or outlier-free models.


* **Publication Visualization Suite:** Exports 600 DPI PNG and vector PDF distribution plots, forest plots, heatmaps, residual diagnostics, and trajectory plots.



## Repository Directory Structure

Running the pipeline automatically constructs the following directory hierarchy and outputs:

```text
├── autonomic_complexity_pipeline.py
├── requirements.txt
├── README.md
└── content/
    ├── ds007554-physio/
    ├── ds007554-phenotype/
    │   ├── participants.tsv
    │   ├── kss_values.tsv
    │   └── cognitive_load.tsv
    └── Pipeline_Results/
        ├── Autonomic_Complexity_Results_CCS_Validated.csv
        ├── Master_HRV_Behavioral_Dataset.csv
        ├── pipeline_run_log.txt
        ├── QC_Plots/
        ├── Publication_Figures/
        └── Statistical_Tables/

```

## Installation and Prerequisites

1. Clone this repository and navigate to the project root directory.
2. Create a virtual environment (Python 3.10 or higher recommended) and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

## Usage

### Option 1: Full End-to-End Execution (Raw S3 Telemetry Download + HRV Extraction + LMEMs)

If the validated HRV feature matrix is not yet present in the results directory, running the main script automatically streams the raw ECG recordings from OpenNeuro S3, performs signal quality screening, extracts HRV and EDR features, runs all statistical models, and exports all figures:

```bash
python autonomic_complexity_pipeline.py

```

### Option 2: Fast Statistical & Visualization Execution (From Pre-Computed CSV)

Because raw 1,882.45 Hz ECG feature extraction across all subjects is computationally intensive, placing the pre-computed Autonomic_Complexity_Results_CCS_Validated.csv file inside the Pipeline_Results folder bypasses raw signal processing and executes the phenotype merge, LMEM suite, variance partitioning, and figure generation in seconds.

## Summary of Primary Empirical Findings

| Experimental Task / Predictor | Coefficient (β) | Standard Error | p-value | Physiological Interpretation |
| --- | --- | --- | --- | --- |
| Intercept (Passive Motor) | 1.520 | 0.042 | <0.001 | Reference baseline complexity

 |
| Active Motor | 0.015 | 0.041 | 0.713 | Preserved complexity

 |
| Motor Imagery | -0.044 | 0.040 | 0.274 | Preserved complexity

 |
| Mental Arithmetic | -0.039 | 0.041 | 0.350 | Sub-threshold cognitive load

 |
| N-Back (2-Back) | -0.122 | 0.040 | 0.002 | Selective complexity collapse

 |
| N-Back + Arithmetic | -0.106 | 0.042 | 0.011 | Selective complexity collapse

 |
| Full Dual-Task | -0.119 | 0.041 | 0.004 | Selective complexity collapse

 |
| Biological Sex (Male) | -0.237 | 0.046 | <0.001 | Lower male baseline complexity

 |
| Baseline Sleepiness (KSS) | -0.020 | 0.014 | 0.159 | Non-significant modulator

 |
| Perceived Cognitive Load | -0.018 | 0.004 | <0.001 | Negative within-subject coupling

 |

In the primary task model (450 observations across 29 participants), fixed effects account for a marginal R-squared of 0.186 (0.189 in the 3-level partitioned model) and a conditional R-squared of 0.500 (0.515 partitioned), with an Intraclass Correlation Coefficient (ICC) of 0.372 to 0.384 demonstrating that 37% to 38% of total variance stems from stable inter-individual and session-block baseline differences.

## Citation and Data Availability

The multimodal physiological dataset analyzed by this pipeline is publicly available on OpenNeuro under accession number ds007554 (Ajra et al., 2026, *Scientific Data*, [https://doi.org/10.18112/openneuro.ds007554.v1.0.0](https://doi.org/10.18112/openneuro.ds007554.v1.0.0)). If you use this code or statistical workflow in your research, please cite our manuscript:

```bibtex
@article{joseph2026autonomic,
  title={Autonomic Complexity Under a Cognitive-Motor Dual-Task Paradigm: A Hierarchical Linear Mixed-Effects Modeling of Heart Rate Sample Entropy},
  author={Joseph, Aswin Tony and Venugopal, Rahul},
 Journal:To be published 

```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
