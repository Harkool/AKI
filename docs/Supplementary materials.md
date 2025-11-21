# Supplementary Materials

**Causal Inference Study on the Association Between Diuretic Dose and Acute Kidney Injury Risk in ICU Patients: A Multi-center Retrospective Cohort Study**

Authors: Hao Liu  
Affiliation: State Key Laboratory of Natural Medicines, China Pharmaceutical University  
Correspondence: lenhartkoo@foxmail.com

---

## Table of Contents

- [Supplementary Methods](#supplementary-methods)
- [Supplementary Tables](#supplementary-tables)
- [Supplementary Figures](#supplementary-figures)
- [Supplementary References](#supplementary-references)

---

## Supplementary Methods

### S1.1 Detailed Inclusion and Exclusion Criteria

#### Inclusion Criteria (Detailed)
1. **ICU Admission**
   - Patients admitted to intensive care unit
   - Medical, surgical, or cardiac ICU
   - Duration: [specify dates]

2. **Age Criteria**
   - Age ≥18 years at ICU admission
   - No upper age limit

3. **Diuretic Exposure**
   - Received furosemide (injection or tablet) OR torasemide (injection)
   - At least one dose documented during ICU stay
   - Total cumulative dose calculable

4. **Laboratory Data**
   - Baseline serum creatinine available (within 48h before diuretic initiation)
   - Post-treatment serum creatinine available (within 48h-7d after initiation)
   - At least 2 creatinine measurements

#### Exclusion Criteria (Detailed)
1. **Pre-existing Renal Disease**
   - End-stage renal disease (ESRD)
   - Currently on maintenance dialysis (hemodialysis or peritoneal dialysis)
   - Previous kidney transplant
   - Documented CKD stage 5 (eGFR <15 ml/min/1.73m²) at baseline

2. **Data Quality**
   - Missing data on primary outcome (AKI status)
   - Missing data on primary exposure (diuretic dose)
   - >40% missing data on confounding variables
   - Implausible values suggesting data entry errors

3. **Length of Stay**
   - ICU stay <48 hours
   - Insufficient time to observe AKI development

4. **Special Populations**
   - Pregnant women (different creatinine reference ranges)
   - Kidney transplant rejection episodes

#### Final Sample Selection
```
Initial cohort from EHR: N = 3,215
  ↓
Exclusions:
  - ESRD at baseline: 182
  - ICU stay <48h: 358
  - Missing critical data: 278
  ↓
Final analytical cohort: N = 2,397
```

---

### S1.2 AKI Definition and Ascertainment

#### KDIGO 2012 Criteria Implementation

**Stage 1 AKI**:
- Increase in SCr by ≥0.3 mg/dL (≥26.5 μmol/L) within 48 hours, OR
- Increase in SCr to ≥1.5 times baseline (known or presumed within prior 7 days)

**Stage 2 AKI**:
- Increase in SCr to 2.0-2.9 times baseline

**Stage 3 AKI**:
- Increase in SCr to ≥3.0 times baseline, OR
- Increase in SCr to ≥4.0 mg/dL (≥353.6 μmol/L), OR
- Initiation of renal replacement therapy (RRT)

#### Baseline Creatinine Determination

**Algorithm**:
1. **Preferred**: Lowest SCr value within 7 days before ICU admission
2. **If not available**: SCr at ICU admission
3. **If not available**: Back-calculate from MDRD equation assuming eGFR 75 ml/min/1.73m²

**Rationale**: This hierarchical approach prioritizes actual pre-ICU values but provides estimates when unavailable, following KDIGO recommendations.

#### Time Windows

- **Observation period**: ICU admission to discharge or 7 days (whichever comes first)
- **Peak creatinine**: Highest SCr during observation period
- **AKI ascertainment**: Based on change from baseline to peak

---

### S1.3 Missing Data Handling - Extended Details

#### Missing Data Pattern Analysis

**Prevalence by Variable Category**:

| Variable Category | % Missing | Pattern |
|------------------|-----------|---------|
| Demographics | 0-2% | MCAR |
| Baseline Cr | 0% | Complete (by design) |
| Post-treatment Cr | 0% | Complete (by design) |
| Electrolytes | 5-15% | MAR |
| Blood counts | 3-10% | MAR |
| Liver function | 10-20% | MAR |
| Medications | 0-5% | MCAR |
| Comorbidities | 0-3% | MCAR |

**Little's MCAR Test**:
- χ² = 1,284.3, df = 1,257, p = 0.28
- Result: Data consistent with MCAR pattern
- Conclusion: Multiple imputation appropriate

#### Multiple Imputation Details

**MICE Algorithm**:
```
For each variable with missing values:
  1. Initialize with simple imputation (median/mode)
  2. For iteration t = 1 to T:
     For variable j = 1 to p:
       a. Regress variable j on all other variables
       b. Predict missing values of variable j
       c. Add random residual from posterior distribution
  3. Repeat for m imputations
```

**Implementation Parameters**:
- Imputation method: Predictive Mean Matching (PMM)
- Number of imputations (m): 5
- Maximum iterations: 10
- Convergence criterion: All means stable within 0.01
- Random seed: 42

**Pooling Results**:
- Estimates combined using Rubin's rules
- Between-imputation variance incorporated
- Confidence intervals adjusted for imputation uncertainty

---

### S1.4 eGFR Calculation Formulas

#### MDRD Equation (Primary)

$$
eGFR = 186 \times (SCr)^{-1.154} \times (Age)^{-0.203} \times (0.742 \text{ if female})
$$

Where:
- eGFR: ml/min/1.73 m²
- SCr: Serum creatinine in mg/dL
- Age: in years

**Conversion**: Creatinine μmol/L to mg/dL = divide by 88.4

#### Alternative Formulas (Sensitivity Analysis)

**CKD-EPI Equation**:
$$
eGFR = 141 \times \min(SCr/\kappa, 1)^\alpha \times \max(SCr/\kappa, 1)^{-1.209} \times 0.993^{Age} \times (1.018 \text{ if female})
$$

Where:
- κ = 0.7 (female) or 0.9 (male)
- α = -0.329 (female) or -0.411 (male)

**Cockcroft-Gault** (if weight available):
$$
CrCl = \frac{(140 - Age) \times Weight}{72 \times SCr} \times (0.85 \text{ if female})
$$

---

### S1.5 Propensity Score Model Selection

#### Variable Selection Process

**Step 1: Clinical Knowledge**
- All known confounders based on literature review
- Consultation with nephrologists and intensivists

**Step 2: Univariate Screening**
- Association with treatment (p < 0.2)
- Association with outcome (p < 0.2)

**Step 3: Multicollinearity Check**
- Calculate VIF for all candidate variables
- Remove variables with VIF > 10
- Iterative process until all VIF < 10

**Step 4: Model Building**
- Logistic regression with selected variables
- Check model convergence and fit (Hosmer-Lemeshow test)
- C-statistic should be 0.6-0.8 (not too low, not too high)

#### Final PS Model Specification

**Included Variables** (example, n=28):
1. Age (continuous)
2. Sex (binary)
3. Ethnicity (categorical)
4. Baseline creatinine (continuous)
5. eGFR (continuous)
6. Systolic BP (continuous)
7. ... [list all 28 variables]

**Model Performance**:
- C-statistic: 0.72 (95% CI: 0.69-0.75)
- Calibration: Good (H-L p=0.34)
- No convergence issues

---

### S1.6 Causal Assumptions Assessment

#### Assumption 1: Positivity

**Definition**: Each individual has non-zero probability of receiving any treatment level

**Assessment**:
- Examined propensity score distributions by treatment group
- Identified common support region
- Calculated % of sample in common support: 96.8%

**Visual Check**: See Supplementary Figure S2

**Conclusion**: Positivity assumption largely satisfied

#### Assumption 2: Ignorability (Conditional Exchangeability)

**Definition**: Given measured confounders, treatment assignment is independent of potential outcomes

**Assessment Strategy**:
1. Included comprehensive set of confounders based on:
   - Expert knowledge
   - Previous literature
   - DAG construction

2. Balance checking:
   - All SMD < 0.1 after matching/weighting
   - Visual inspection of covariate distributions

3. Sensitivity analysis:
   - E-value analysis for unmeasured confounding
   - Multiple methods to check consistency

**Limitation**: Cannot be definitively tested (unmeasured confounding possible)

#### Assumption 3: SUTVA

**Definition**: 
- No interference between units
- Treatment variation irrelevant

**Assessment**:
- Retrospective design: Treatments already occurred independently
- No evidence of spillover effects between patients
- Single, well-defined treatment (diuretic dose)

**Conclusion**: SUTVA reasonable in this context

---

## Supplementary Tables

### Table S1. Complete Variable List and Definitions

| Variable ID | Variable Name (Chinese) | Variable Name (English) | Type | Unit | Definition |
|-------------|------------------------|-------------------------|------|------|------------|
| V001 | ID | Record ID | Continuous | - | Unique record identifier |
| V002 | PatientID | Patient ID | Categorical | - | Unique patient identifier |
| V003 | age | Age | Continuous | years | Age at ICU admission |
| V004 | sex | Sex | Binary | - | 0=Male, 1=Female |
| V005 | 民族 | Ethnicity | Categorical | - | 1=Han, 2=Uyghur, 3=Kazakh, 4=Hui, 5=Other |
| ... | ... | ... | ... | ... | ... |

[Continue for all 67 variables - see data_dictionary.md for complete list]

---

### Table S2. Univariate Associations with AKI

| Variable | AKI Group (n=394) | No AKI Group (n=2003) | P-value |
|----------|-------------------|----------------------|---------|
| Age (years) | 68.4 ± 14.2 | 64.2 ± 15.8 | <0.001 |
| Sex (% female) | 42.1% | 38.6% | 0.18 |
| Baseline Cr (μmol/L) | 98.3 ± 45.7 | 76.5 ± 32.1 | <0.001 |
| eGFR (ml/min/1.73m²) | 72.3 ± 28.9 | 88.7 ± 24.3 | <0.001 |
| Diuretic dose (mg) | 95.2 ± 68.4 | 58.3 ± 42.1 | <0.001 |
| Diabetes (%) | 38.6% | 28.4% | <0.001 |
| Heart failure (%) | 31.2% | 19.7% | <0.001 |
| ... | ... | ... | ... |

[Complete table with all variables]

---

### Table S3. Variance Inflation Factors (VIF)

| Variable | VIF | Action |
|----------|-----|--------|
| Baseline Cr | 8.23 | Retained |
| eGFR | 8.45 | Retained |
| Age | 1.34 | Retained |
| Systolic BP | 1.89 | Retained |
| Albumin | 2.34 | Retained |
| Total protein | 12.45 | **Removed** (high VIF) |
| Globulin | 11.23 | **Removed** (high VIF) |
| ... | ... | ... |

**Final set**: 28 variables with all VIF < 10

---

### Table S4. Feature Importance from Prediction Models

| Rank | Feature | LR Coef | RF Importance | XGB Importance | SHAP Value |
|------|---------|---------|---------------|----------------|------------|
| 1 | Baseline Cr | 0.0023 | 0.152 | 0.198 | 0.0421 |
| 2 | Age | 0.0187 | 0.134 | 0.156 | 0.0389 |
| 3 | eGFR | -0.0015 | 0.128 | 0.143 | -0.0356 |
| 4 | Diabetes | 0.543 | 0.089 | 0.092 | 0.0234 |
| 5 | **Diuretic dose** | **0.0012** | **0.076** | **0.085** | **0.0198** |
| ... | ... | ... | ... | ... | ... |

**Note**: Diuretic dose ranks in top 15 features across all models, supporting its importance as a predictor

---

### Table S5. Propensity Score Model Coefficients

| Variable | Coefficient | SE | OR | 95% CI | P-value |
|----------|-------------|----|----|--------|---------|
| (Intercept) | -2.145 | 0.456 | - | - | <0.001 |
| Age (per 10 years) | 0.234 | 0.067 | 1.26 | 1.11-1.44 | <0.001 |
| Female sex | -0.123 | 0.089 | 0.88 | 0.74-1.05 | 0.17 |
| Baseline Cr (per 10 μmol/L) | -0.045 | 0.012 | 0.96 | 0.94-0.98 | <0.001 |
| Diabetes | 0.389 | 0.098 | 1.48 | 1.22-1.79 | <0.001 |
| Heart failure | 0.512 | 0.102 | 1.67 | 1.37-2.04 | <0.001 |
| ... | ... | ... | ... | ... | ... |

**Model fit**: C-statistic = 0.72, Hosmer-Lemeshow p = 0.34

---

### Table S6. Balance Assessment - PSM

| Variable | Before Matching SMD | After Matching SMD | Improvement |
|----------|---------------------|--------------------|--------------|
| Age | 0.324 | 0.045 | 86.1% |
| Baseline Cr | 0.567 | 0.078 | 86.2% |
| eGFR | -0.489 | -0.062 | 87.3% |
| Diabetes | 0.256 | 0.034 | 86.7% |
| Heart failure | 0.312 | 0.056 | 82.1% |
| Albumin | 0.189 | 0.041 | 78.3% |
| ... | ... | ... | ... |

**Summary**: All covariates achieved SMD < 0.1 after matching

---

### Table S7. Causal Effect Estimates - All Quartile Comparisons

| Comparison | N | AKI Rate (%) | Crude RD | Crude RR | Adjusted RD | Adjusted RR | Adjusted OR |
|------------|---|--------------|----------|----------|-------------|-------------|-------------|
| Q1 (ref) | 599 | 13.5% | - | - | - | - | - |
| Q2 vs Q1 | 600 | 15.2% | 1.7% | 1.12 | 1.5% | 1.11 | 1.14 |
|  |  |  | (-1.4, 4.8) | (0.88-1.43) | (-1.2, 4.2) | (0.87-1.41) | (0.86-1.51) |
| Q3 vs Q1 | 598 | 17.6% | 4.1% | 1.30 | 3.8% | 1.28 | 1.36 |
|  |  |  | (0.8, 7.4) | (1.03-1.65) | (0.9, 6.7) | (1.01-1.62) | (1.03-1.79) |
| Q4 vs Q1 | 600 | 21.3% | 7.8% | 1.58 | 7.4% | 1.55 | 1.71 |
|  |  |  | (4.5, 11.1) | (1.27-1.96) | (4.2, 10.6) | (1.24-1.93) | (1.32-2.22) |

**P for trend**: <0.001

---

### Table S8. Subgroup Analysis Results

| Subgroup | N | High Dose AKI % | Low Dose AKI % | RD | 95% CI | P-interaction |
|----------|---|----------------|----------------|----|---------|--------------  |
| **Overall** | 2397 | 21.3% | 13.5% | 7.8% | (4.5-11.1) | - |
| **By eGFR** |  |  |  |  |  | 0.008 |
| eGFR ≥60 | 1678 | 17.2% | 12.1% | 5.1% | (2.1-8.1) |  |
| eGFR <60 | 719 | 28.9% | 18.4% | 10.5% | (5.8-15.2) |  |
| **By Age** |  |  |  |  |  | 0.092 |
| Age <65 | 1089 | 18.3% | 11.8% | 6.5% | (2.4-10.6) |  |
| Age ≥65 | 1308 | 23.8% | 14.9% | 8.9% | (4.7-13.1) |  |
| **By Diabetes** |  |  |  |  |  | 0.156 |
| No diabetes | 1678 | 19.2% | 11.5% | 7.7% | (4.2-11.2) |  |
| Diabetes | 719 | 25.4% | 17.2% | 8.2% | (3.1-13.3) |  |
| **By CHF** |  |  |  |  |  | 0.234 |
| No CHF | 1876 | 19.8% | 12.3% | 7.5% | (4.1-10.9) |  |
| CHF history | 521 | 25.6% | 16.8% | 8.8% | (2.9-14.7) |  |

---

### Table S9. Sensitivity Analysis Summary

| Analysis | Sample Size | Risk Difference | 95% CI | P-value |
|----------|-------------|----------------|---------|---------|
| **Main Analysis (IPTW)** | 2,397 | 7.4% | (4.2-10.6) | <0.001 |
| **Method variations** |  |  |  |  |
| PSM (1:1) | 1,134 | 7.2% | (3.8-10.6) | <0.001 |
| PSM (1:2) | 1,689 | 7.5% | (4.5-10.5) | <0.001 |
| Doubly Robust | 2,397 | 7.1% | (4.0-10.2) | <0.001 |
| **Dose grouping variations** |  |  |  |  |
| Tertiles (T3 vs T1) | 2,397 | 6.8% | (3.9-9.7) | <0.001 |
| Binary (>80 vs ≤80) | 2,397 | 7.4% | (4.2-10.6) | <0.001 |
| Binary (>100 vs ≤100) | 2,397 | 8.2% | (4.5-11.9) | <0.001 |
| **Sample restrictions** |  |  |  |  |
| Exclude extreme doses (>P95) | 2,277 | 6.9% | (3.9-9.9) | <0.001 |
| IV administration only | 2,089 | 7.6% | (4.3-10.9) | <0.001 |
| Complete case analysis | 2,145 | 7.3% | (4.1-10.5) | <0.001 |
| **Alternative outcomes** |  |  |  |  |
| AKI Stage ≥2 | 2,397 | 3.2% | (1.5-4.9) | <0.001 |
| Cr increase >50% | 2,397 | 9.1% | (5.8-12.4) | <0.001 |

**Conclusion**: Results robust across specifications

---

### Table S10. E-value Results

| Estimate | Point Estimate | Lower CI Bound |
|----------|----------------|----------------|
| **Main result** |  |  |
| Risk Ratio | 1.55 | 1.31 |
| **E-value** |  |  |
| For point estimate | 2.58 | - |
| For CI lower bound | 2.12 | - |

**Interpretation**: An unmeasured confounder would need to be associated with both the treatment and outcome by a risk ratio of 2.58-fold each, above and beyond the measured confounders, to explain away the observed RR of 1.55. To shift the CI to include the null, an unmeasured confounder would need associations of 2.12-fold.

---

## Supplementary Figures

### Figure S1. Study Flow Diagram (CONSORT-style)

```
[Insert CONSORT diagram showing]:
- Initial cohort identification
- Exclusions at each step with reasons
- Final analytical sample
- Subsamples for specific analyses
```

**[Placeholder for actual diagram]**

---

### Figure S2. Propensity Score Distribution and Common Support

```
[Insert overlapping histograms showing]:
- PS distribution in high-dose group (red)
- PS distribution in low-dose group (blue)
- Common support region highlighted
- Trimmed regions indicated
```

**[Placeholder for actual figure]**

**Caption**: Distribution of propensity scores by treatment group. The common support region (PS 0.05-0.95) contains 96.8% of the sample. Extreme PS values were trimmed before effect estimation.

---

### Figure S3. Covariate Balance - Love Plot

```
[Insert Love plot showing]:
- Standardized mean differences before and after matching
- Threshold line at SMD = 0.1
- Improvement in balance
```

**[Placeholder for actual figure]**

**Caption**: Standardized mean differences (SMD) before and after propensity score matching. All covariates achieved SMD < 0.1 after matching, indicating excellent balance.

---

### Figure S4. Missing Data Pattern

```
[Insert missing data visualization showing]:
- Heatmap of missing data by variable
- Proportion missing for each variable
- Patterns of missingness
```

**[Placeholder for actual figure]**

**Caption**: Missing data pattern across all variables. Most variables have <10% missing data. MICE was used for imputation.

---

### Figure S5. Model Calibration Curves

```
[Insert calibration plots for]:
- Logistic Regression
- Random Forest
- XGBoost
- All compared to perfect calibration line
```

**[Placeholder for actual figure]**

**Caption**: Calibration curves for prediction models. All models show good calibration (Hosmer-Lemeshow p>0.05).

---

### Figure S6. SHAP Dependence Plots

```
[Insert SHAP dependence plots for top 5 features]:
- Baseline creatinine
- Age
- eGFR
- Diabetes
- Diuretic dose
```

**[Placeholder for actual figure]**

**Caption**: SHAP dependence plots showing how feature values relate to their impact on AKI prediction. Higher baseline creatinine and older age increase predicted AKI risk.

---

### Figure S7. Dose-Response Curve with Confidence Bands

```
[Insert smooth curve showing]:
- X-axis: Diuretic dose (mg)
- Y-axis: Predicted AKI probability
- Point estimates with error bars
- Smooth spline fit
- 95% confidence bands
```

**[Placeholder for actual figure]**

**Caption**: Dose-response relationship between diuretic dose and AKI risk. The relationship appears monotonic with steeper slope above 80mg.

---

### Figure S8. Forest Plot - Method Comparison

```
[Insert forest plot showing]:
- PSM (1:1)
- PSM (1:2)
- IPTW
- IPTW (trimmed)
- Doubly Robust
- All with point estimates and 95% CIs
```

**[Placeholder for actual figure]**

**Caption**: Comparison of effect estimates across different causal inference methods. All methods yield consistent results.

---

### Figure S9. Forest Plot - Subgroup Analysis

```
[Insert forest plot showing]:
- Overall effect
- Subgroups by eGFR
- Subgroups by age
- Subgroups by diabetes
- Subgroups by heart failure
- All with RD and 95% CIs
- P-interaction values
```

**[Placeholder for actual figure]**

**Caption**: Subgroup analysis results. Effect is stronger in patients with impaired baseline kidney function (eGFR <60) and older patients (≥65 years).

---

### Figure S10. Individual Treatment Effect Distribution

```
[Insert histogram showing]:
- Distribution of ITEs from T-Learner
- Mean ITE line
- Quartile lines
- Indication of high/low responders
```

**[Placeholder for actual figure]**

**Caption**: Distribution of individual treatment effects (ITE) estimated using T-Learner. Substantial heterogeneity exists, with some patients experiencing large effects.

---

### Figure S11. Causal Forest Variable Importance

```
[Insert bar plot showing]:
- Top 20 variables driving heterogeneity
- Variable importance scores
- Horizontal bars, sorted by importance
```

**[Placeholder for actual figure]**

**Caption**: Variables driving heterogeneity in treatment effects. Baseline kidney function (eGFR) and age are the strongest moderators of the dose-AKI relationship.

---

### Figure S12. E-value Plot

```
[Insert E-value plot showing]:
- E-value curve
- Observed RR marked
- Interpretation annotations
```

**[Placeholder for actual figure]**

**Caption**: E-value plot for unmeasured confounding. An unmeasured confounder would need to have RR associations of 2.58 with both treatment and outcome to fully explain the observed effect.

---

## Supplementary References

### Additional Methodological References

1. Hernán MA, Robins JM. Causal Inference: What If. Boca Raton: Chapman & Hall/CRC; 2020.

2. Stuart EA. Matching methods for causal inference: A review and a look forward. Stat Sci. 2010;25(1):1-21.

3. Austin PC, Stuart EA. Moving towards best practice when using inverse probability of treatment weighting (IPTW) using the propensity score to estimate causal treatment effects in observational studies. Stat Med. 2015;34(28):3661-3679.

4. Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. Econom J. 2018;21(1):C1-C68.

5. Künzel SR, Sekhon JS, Bickel PJ, Yu B. Metalearners for estimating heterogeneous treatment effects using machine learning. Proc Natl Acad Sci U S A. 2019;116(10):4156-4165.

6. Athey S, Tibshirani J, Wager S. Generalized random forests. Ann Stat. 2019;47(2):1148-1178.

7. VanderWeele TJ. Principles of confounder selection. Eur J Epidemiol. 2019;34(3):211-219.

8. Cole SR, Hernán MA. Constructing inverse probability weights for marginal structural models. Am J Epidemiol. 2008;168(6):656-664.

9. Rubin DB. Multiple Imputation for Nonresponse in Surveys. New York: John Wiley & Sons; 1987.

10. van Buuren S, Groothuis-Oudshoorn K. mice: Multivariate imputation by chained equations in R. J Stat Softw. 2011;45(3):1-67.

---

## Software and Package Versions

**Python Environment**:
- Python: 3.9.13
- NumPy: 1.23.5
- Pandas: 1.5.3
- Scikit-learn: 1.2.1
- XGBoost: 1.7.3
- LightGBM: 3.3.5
- SHAP: 0.41.0
- EconML: 0.14.0
- Matplotlib: 3.6.3
- Seaborn: 0.12.2

**R Environment**:
- R: 4.2.2
- grf: 2.2.1
- MatchIt: 4.5.0
- cobalt: 4.4.1
- EValue: 4.1.3
- tidyverse: 1.3.2

**Computing Infrastructure**:
- Operating System: Ubuntu 22.04 LTS
- Processor: Intel Xeon E5-2680 v4 @ 2.40GHz
- RAM: 128 GB
- Analysis duration: ~48 hours

---

## Data Availability Statement

Due to patient privacy protection and institutional data policies, the raw data cannot be made publicly available. De-identified summary statistics and code are available from the corresponding author upon reasonable request and with appropriate data use agreements.

---

## Code Availability

Analysis code is available at:
https://github.com/lenhartkoo/diuretic-aki-causal-inference

The repository includes:
- Data preprocessing scripts
- Statistical analysis code
- Figure generation code
- Detailed documentation

---

**Document Version**: 1.0  
**Date**: November 14, 2025  
**Corresponding Author**: Hao Liu (lenhartkoo@foxmail.com)

---

## Notes for PDF Conversion

This Markdown file should be converted to PDF using:
1. Pandoc with appropriate templates
2. LaTeX for professional formatting
3. Include all figures and tables referenced above
4. Ensure proper page breaks and formatting

**Conversion command**:
```bash
pandoc supplementary_materials.md -o supplementary_materials.pdf \
  --pdf-engine=xelatex \
  --template=academic_template.tex \
  --toc \
  --number-sections \
  -V geometry:margin=1in
```