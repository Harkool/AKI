# Diuretic Dose and Acute Kidney Injury: A Causal Inference Study

[![DOI](https://img.shields.io/badge/DOI-pending-orange)](https://github.com/Harkool/diuretic-aki-causal-inference)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

> **Causal Inference Study on the Association Between Diuretic Dose and Acute Kidney Injury Risk in ICU Patients: A Multi-center Retrospective Cohort Study**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Study Design](#-study-design)
- [Dataset](#-dataset)
- [Methods](#-methods)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🔬 Overview

This repository contains the code and documentation for a **multi-center retrospective cohort study** investigating the causal relationship between diuretic dose and acute kidney injury (AKI) risk in ICU patients.

### Research Objectives

**Primary Objective:**
- Evaluate the causal relationship between diuretic dose and AKI incidence

**Secondary Objectives:**
1. Explore the shape of dose-response relationship (linear/threshold/nonlinear)
2. Identify high-risk populations (heterogeneity analysis)
3. Investigate the potential role of electrolyte imbalance
4. Provide clinical recommendations for dose optimization

### Study Centers

- **Xinjiang Medical University First Affiliated Hospital** (新疆医科大学第一附属医院)
- **Xiangya Hospital, Central South University** (中南大学湘雅医院)

---

## ✨ Key Features

🎯 **Novel Approach**
- Shifts from "use vs. no use" to "how much to use" paradigm
- Combines machine learning prediction with causal inference
- Identifies patient subgroups with differential treatment effects

🔧 **Robust Methodology**
- Multiple causal inference methods (PSM, IPTW, Doubly Robust)
- Comprehensive sensitivity analyses (E-value, different specifications)
- Machine learning for heterogeneity detection (Causal Forest, T-Learner)

📊 **Clinical Relevance**
- Risk stratification system for clinical decision support
- Actionable dose optimization recommendations
- Identifies vulnerable populations requiring closer monitoring

---

## 📐 Study Design

### Study Type
Multi-center retrospective cohort study based on Electronic Health Records (EHR)

### Study Population
- **Sample Size:** 2,397 ICU patients
- **AKI Incidence:** 394 cases (16.4%)
  - Stage 1: 279 (11.6%)
  - Stage 2: 64 (2.7%)
  - Stage 3: 51 (2.1%)

### Inclusion Criteria
1. ICU hospitalized patients
2. Received diuretic treatment (furosemide or torasemide)
3. Complete baseline and post-treatment creatinine data
4. Age ≥18 years

### Exclusion Criteria
1. End-stage renal disease (ESRD) requiring dialysis at admission
2. Hospital stay <48 hours
3. >40% missing data on key variables

---

## 📊 Dataset

### Data File
`AKI.csv`

- **Rows:** 2,398 (including header)
- **Columns:** 67 variables
- **Encoding:** UTF-8 with BOM

### Key Variables

#### Primary Exposure
- **利尿剂总剂量** (Diuretic total dose, mg)

#### Primary Outcome
- **AKI class** (KDIGO criteria)
  - 0 = No AKI
  - 1 = Stage 1 (Cr increase 1.5-1.9× or ≥26.5 μmol/L)
  - 2 = Stage 2 (Cr increase 2.0-2.9×)
  - 3 = Stage 3 (Cr increase ≥3.0× or ≥353.6 μmol/L or RRT)

#### Important Confounders
- Demographics: age, sex, ethnicity
- Baseline kidney function: creatinine, eGFR
- Comorbidities: diabetes, hypertension, heart failure, sepsis
- Concomitant medications: NSAIDs, ACEI/ARB, contrast agents, antibiotics
- Laboratory values: electrolytes, blood counts, liver function

#### Secondary Outcomes
- Electrolyte disorders (hyponatremia, hypernatremia, hypokalemia, hyperkalemia)
- Creatinine changes (absolute and relative)

For detailed variable descriptions, see [Data Dictionary](docs/data_dictionary.md).

---

## 🛠 Methods

### Five-Step Analytical Framework

#### Step 1: Predictive Modeling
**Objective:** Identify AKI risk factors and assess predictive importance of diuretic dose

**Models:**
- Logistic Regression (baseline)
- Random Forest
- XGBoost / LightGBM (primary)
- Deep Learning (exploratory)

**Evaluation:**
- AUC-ROC, AUPRC
- Calibration plots, Brier score
- Decision curve analysis
- SHAP values for feature importance

#### Step 2: Causal Inference
**Objective:** Estimate causal effect of diuretic dose on AKI

**Dose Stratification:**
- Q1 (low dose): 2-20 mg
- Q2 (moderate-low): >20-40 mg
- Q3 (moderate-high): >40-80 mg
- Q4 (high dose): >80 mg

**Methods:**
1. **Propensity Score Matching (PSM)**
   - 1:1 nearest neighbor matching
   - Caliper: 0.2×SD(logit PS)

2. **Inverse Probability Treatment Weighting (IPTW)**
   - Propensity score estimation
   - Weight calculation and trimming
   - Balance checking (SMD < 0.1)

3. **Doubly Robust Estimation**
   - Combines propensity score and outcome models
   - Enhanced robustness

**Effect Estimates:**
- Average Treatment Effect (ATE)
- Risk Difference (RD)
- Risk Ratio (RR)
- Odds Ratio (OR)
- 95% Confidence Intervals

**Trend Test:**
- Cochran-Armitage trend test
- Linear regression for trend

#### Step 3: Association Pathway Exploration
**Research Question:** Does electrolyte imbalance play a role in the dose-AKI relationship?

⚠️ **Important:** Due to single time-point data, we conduct **association pathway analysis** rather than formal causal mediation analysis.

**Analyses:**
1. Pathway a: Dose → Electrolyte disorder
2. Pathway b: Electrolyte disorder → AKI (adjusting for dose)
3. Total effect: Dose → AKI
4. Direct effect: Dose → AKI (adjusting for electrolytes)
5. Stratified analysis by electrolyte status
6. Joint dose-electrolyte-AKI analysis

#### Step 4: Heterogeneity Analysis
**Objective:** Identify patient subgroups more sensitive to high-dose diuretics

**Traditional Subgroup Analysis:**
- Baseline kidney function (by eGFR)
- Age (<65 vs. ≥65 years)
- Diabetes status
- Heart failure history
- Diuretic type
- Concomitant ACEI/ARB use

**Machine Learning Heterogeneity Analysis:**

1. **T-Learner**
   - Estimate Individual Treatment Effects (ITE)
   - Analyze ITE distribution
   - Identify high vs. low responders

2. **Causal Forest**
   - CATE estimation
   - Feature importance analysis
   - Best Linear Projection (BLP)
   - Explain sources of heterogeneity

**Clinical Risk Stratification:**
- Low risk (score 0-1)
- Moderate risk (score 2-3)
- High risk (score 4-6)

#### Step 5: Sensitivity Analysis
**Objective:** Verify robustness and assess potential biases

**Analyses:**
1. **E-value analysis:** Assess impact of unmeasured confounding
2. **Method comparison:** PSM vs. IPTW vs. Doubly Robust
3. **Different grouping schemes:** Tertiles vs. quartiles vs. quintiles
4. **Exclusion analyses:**
   - Exclude extreme doses
   - Restrict to IV administration
   - Complete case analysis
5. **Negative and positive controls**
6. **Dose-response shape testing:** Linear vs. threshold vs. nonlinear

---

## 💻 Installation

### Prerequisites

**Python Requirements:**
```bash
python >= 3.8
numpy >= 1.20.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
lightgbm >= 3.3.0
shap >= 0.40.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
```

**R Requirements:**
```r
R >= 4.0.0
grf >= 2.0.0
MatchIt >= 4.0.0
cobalt >= 4.3.0
EValue >= 4.1.0
```

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/Harkool/diuretic-aki-causal-inference.git
cd diuretic-aki-causal-inference
```

2. **Create Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Install R packages**
```r
# Run in R console
install.packages(c("grf", "MatchIt", "cobalt", "EValue", "tidyverse"))
```

---

## 🚀 Usage

### Quick Start

1. **Data Preprocessing**
```bash
python src/01_data_preprocessing.py
```

2. **Predictive Modeling**
```bash
python src/02_predictive_modeling.py
```

3. **Causal Inference**
```bash
python src/03_causal_inference.py
```

4. **Heterogeneity Analysis**
```r
# Run in R
source("src/04_heterogeneity_analysis.R")
```

5. **Generate Figures and Tables**
```bash
python src/05_generate_outputs.py
```

### Detailed Workflow

See [notebooks/](notebooks/) for step-by-step Jupyter notebooks:
- `01_exploratory_analysis.ipynb`
- `02_feature_engineering.ipynb`
- `03_prediction_models.ipynb`
- `04_causal_analysis.ipynb`
- `05_heterogeneity.ipynb`
- `06_sensitivity_analysis.ipynb`

---

## 📈 Results

### Main Findings

**Finding 1: Dose-Response Relationship**
- Clear dose-dependent increase in AKI risk
- Q1 (reference): AKI rate ~13-15%
- Q4 (high dose): AKI rate ~20-22%
- P for trend < 0.001

**Finding 2: High Dose Significantly Increases AKI Risk**
- High dose (>80mg) vs. Low dose (≤80mg):
  - Risk Difference: 6-8%
  - Risk Ratio: 1.4-1.5
  - Odds Ratio: 1.5-1.6
  - P < 0.001

**Finding 3: Role of Electrolyte Imbalance**
- High dose increases electrolyte disorder risk
- Electrolyte disorders associated with AKI
- Effect remains significant after adjusting for electrolytes

**Finding 4: Heterogeneous Effects**

**High-risk subgroups (stronger effects):**
- Impaired baseline kidney function (higher creatinine)
- Age ≥65 years
- Diabetes mellitus
- History of heart failure
- Concomitant ACEI/ARB use

### Outputs

**Main Tables:**
- Table 1: Baseline characteristics by AKI stage
- Table 2: Predictive model performance
- Table 3: Causal effect estimates (dose quartiles)
- Table 4: Method comparison (PSM/IPTW/DR)
- Table 5: Association pathway analysis
- Table 6: Subgroup analysis
- Table 7: Clinical risk score

**Main Figures:**
- Figure 1: Study flowchart
- Figure 2: Prediction model results (ROC/PR/Calibration/SHAP)
- Figure 3: Dose-response curve
- Figure 4: Continuous dose-AKI risk
- Figure 5: Association pathway
- Figure 6: Subgroup analysis forest plot
- Figure 7: Heterogeneity analysis (ITE/Feature importance)
- Figure 8: Clinical risk stratification
- Figure 9: Sensitivity analysis forest plot

All outputs available in [results/](results/) and [figures/](figures/).

---

## 📁 Project Structure

```
diuretic-aki-causal-inference/
├── README.md                          # This file
├── README_CN.md                       # Chinese version
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
├── .gitignore                         # Git ignore rules
│
├── data/                              # Data directory (not tracked)
│   ├── raw/                          # Raw data
│   ├── processed/                    # Processed data
│   └── sample/                       # Sample data for demo
│
├── docs/                              # Documentation
│   ├── data_dictionary.md            # Variable descriptions
│   ├── study_protocol.md             # Study protocol (Chinese)
│   ├── methods_details.md            # Detailed methods
│   └── supplementary_materials.pdf   # Supplementary materials
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── 01_data_preprocessing.py      # Data cleaning and preprocessing
│   ├── 02_predictive_modeling.py     # ML prediction models
│   ├── 03_causal_inference.py        # Causal inference analysis
│   ├── 04_heterogeneity_analysis.R   # Heterogeneity analysis (R)
│   ├── 05_sensitivity_analysis.py    # Sensitivity analyses
│   ├── 06_generate_outputs.py        # Generate tables and figures
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── data_utils.py
│       ├── model_utils.py
│       ├── causal_utils.py
│       └── plot_utils.py
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_prediction_models.ipynb
│   ├── 04_causal_analysis.ipynb
│   ├── 05_heterogeneity.ipynb
│   └── 06_sensitivity_analysis.ipynb
│
├── results/                           # Analysis results
│   ├── tables/                       # Tables (CSV, Excel)
│   ├── figures/                      # Figures (PNG, PDF)
│   └── models/                       # Saved models
│
└── tests/                             # Unit tests
    ├── __init__.py
    ├── test_data_utils.py
    ├── test_model_utils.py
    └── test_causal_utils.py
```

---

## 🔒 Data Availability

Due to patient privacy protection and hospital data policies, the complete dataset is not publicly available.

After publication, reasonable data requests will be considered under the following conditions:
1. Clear scientific research purpose
2. Institutional Review Board (IRB) approval
3. Signed data use agreement
4. Compliance with Chinese data security and privacy laws

Please contact the corresponding author for data access requests.

---

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex

```

**For code:**
```bibtex

```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** The license applies to the code only. Data usage is subject to separate data use agreements and institutional policies.

---

## 🙏 Acknowledgments

- **Data Contributors:**
  - Xinjiang Medical University First Affiliated Hospital
  - Xiangya Hospital, Central South University

- **Technical Support:**
  - State Key Laboratory of Natural Medicines, China Pharmaceutical University
  - Key Laboratory of Drug Metabolism, China Pharmaceutical University

- **Funding:**
  - [List funding sources if applicable]

---

## 👤 Contact

**Hao Liu**

📧 **Email:** lenhartkoo@foxmail.com

🏢 **Affiliation:**  
State Key Laboratory of Natural Medicines  
Key Laboratory of Drug Metabolism  
China Pharmaceutical University  
Nanjing, Jiangsu 210009, China

---

## 🔄 Version History

- **v1.0.0** (2025-11-14): Initial release
  - Multi-center data (Xinjiang + Xiangya)
  - Complete five-step analysis framework
  - Comprehensive sensitivity analyses

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

**Please make sure to:**
1. Update tests as appropriate
2. Follow the existing code style
3. Add clear commit messages
4. Update documentation

---

## 📚 References

**Key Methodological References:**

1. **Causal Inference:**
   - Austin PC. An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behav Res*. 2011;46(3):399-424.
   - Funk MJ, et al. Doubly robust estimation of causal effects. *Am J Epidemiol*. 2011;173(7):761-767.

2. **Machine Learning for Heterogeneity:**
   - Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. *J Am Stat Assoc*. 2018;113(523):1228-1242.
   - Künzel SR, et al. Metalearners for estimating heterogeneous treatment effects using machine learning. *PNAS*. 2019;116(10):4156-4165.

3. **Sensitivity Analysis:**
   - VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*. 2017;167(4):268-274.

4. **AKI Definition:**
   - Kellum JA, et al. Kidney disease: improving global outcomes (KDIGO) acute kidney injury work group. KDIGO clinical practice guideline for acute kidney injury. *Kidney Int Suppl*. 2012;2(1):1-138.

---

## 📊 Project Status

![Status](https://img.shields.io/badge/Status-Active-success)
![Build](https://img.shields.io/badge/Build-Passing-success)
![Coverage](https://img.shields.io/badge/Coverage-85%25-yellowgreen)

**Current Phase:** Manuscript preparation

**Next Steps:**
- [ ] Complete final analyses
- [ ] Submit to journal
- [ ] Prepare code for public release
- [ ] Create detailed documentation
- [ ] Add interactive visualizations
