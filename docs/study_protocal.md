## Study Protocol

### Basic Information

| Item                       | Content                                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Study title**            | Causal Inference of Diuretic Dose and Acute Kidney Injury Risk in ICU Patients: A Multi-Center Retrospective Cohort Study |
| **Study type**             | Retrospective cohort study                                                                                                |
| **Study sites**            | The First Affiliated Hospital of Xinjiang Medical University; Xiangya Hospital, Central South University                  |
| **Principal investigator** | Liu Hao                                                                                                                   |
| **Affiliation**            | State Key Laboratory of Natural Drug Active Components and Pharmacodynamics, China Pharmaceutical University              |
| **Contact**                | [lenhartkoo@foxmail.com](mailto:lenhartkoo@foxmail.com)                                                                   |
| **Study ID**               | [to be assigned]                                                                                                          |
| **Ethics approval**        | Approved by both hospitals’ ethics committees                                                                             |
| **Study period**           | January 2024 – December 2025                                                                                              |

### Table of contents

1. Background
2. Objectives & Hypotheses
3. Study Design
4. Study Population
5. Variable Definitions
6. Sample Size Calculation
7. Statistical Analysis Plan
8. Quality Control
9. Ethics Considerations
10. Study Timeline
11. Expected Outcomes
12. References

---

## 1. Background

### 1.1 Importance of the Clinical Problem

Acute kidney injury (AKI) is a common complication among ICU patients (incidence ~20-50 %), and is closely associated with increased in-hospital mortality, longer length of stay, and higher medical costs. Drug-related factors are an important potentially preventable contributor to AKI.

Diuretics, especially loop diuretics (e.g., furosemide, torasemide), are cornerstone drugs in ICU fluid management, widely used for heart failure, renal dysfunction, oedema, etc. However, diuretic use may also increase AKI risk through multiple mechanisms:

1. **Haemodynamic effects**: excessive diuresis may lead to hypovolaemia and pre-renal AKI.
2. **Electrolyte disturbances**: hyponatraemia, hypokalaemia may impair tubular function.
3. **Direct nephrotoxicity**: high-dose loop diuretics may injure renal tubules.
4. **RAAS activation**: excessive diuresis may trigger renin-angiotensin-aldosterone system, contributing to renal injury.

### 1.2 Limitations of Existing Research

Although several studies have explored the relationship between diuretics and AKI, there are the following limitations:

1. Most studies focus on “use vs non-use”, rather than “how much dose”.
2. Lack of systematic dose–response relationship analyses.
3. Insufficient control of confounding factors, making causal inference difficult.
4. Heterogeneous effects (which patients are more sensitive) are under-explored.
5. The potential mediating role of electrolyte disturbances is not systematically studied.

### 1.3 Knowledge Gaps

Remaining unclear are:

* Whether there is a causal relationship between diuretic dose and AKI risk.
* The precise shape of the dose–response relationship (linear, threshold, nonlinear).
* Which patient subgroups are more sensitive to high-dose diuretics.
* Whether electrolyte disturbances mediate the dose → AKI relationship.
* What is the optimal dose strategy in clinical practice.

### 1.4 Innovation of This Study

This study’s innovations are:

1. **Paradigm shift**: from “use vs non-use” to “how much dose”.
2. **Methodological innovation**: combining machine learning and causal inference methods.
3. **Systematic analysis**: comprehensively exploring dose–response, heterogeneity, and pathways.
4. **Clinical practicality**: providing actionable dose-optimization suggestions and risk stratification tools.

---

## 2. Objectives & Hypotheses

### 2.1 Primary Objective

To evaluate the causal effect of diuretic dose on AKI incidence in ICU patients.

### 2.2 Secondary Objectives

1. Explore the dose–response shape between diuretic dose and AKI risk.
2. Identify patient subgroups sensitive to high-dose diuretics.
3. Explore the potential association of electrolyte disturbances in the dose → AKI relationship.
4. Construct a clinical decision support risk-scoring system.
5. Provide evidence-based dose-optimization recommendations.

### 2.3 Hypotheses

**Primary hypothesis**:

* H1: Diuretic dose has a causal relationship with AKI risk; higher dose increases AKI risk.
* H0: Diuretic dose has no causal relationship with AKI risk.

**Secondary hypotheses**:

* The dose–response curve is monotonically increasing.
* There is a dose threshold (e.g., ~80 mg), above which risk increases markedly.
* Patients with baseline renal dysfunction, older age, diabetes are more sensitive to high-dose diuretics.
* Electrolyte disturbances play a partial mediating/associative role in the dose → AKI relationship.

---

## 3. Study Design

### 3.1 Study Type

Multi-center retrospective cohort study.

### 3.2 Study Sites

* The First Affiliated Hospital of Xinjiang Medical University
* Xiangya Hospital, Central South University

### 3.3 Study Period

Data collection period: [to be specified, e.g., Jan 2020 – Dec 2023].

### 3.4 Data Source

Electronic Health Records (EHR) systems.

### 3.5 Study Framework

We will adopt a **five-step analysis framework**:

```
Step 1: Predictive modelling → identify AKI risk factors and examine the predictive importance of diuretic dose  
    ↓  
Step 2: Causal inference → estimate the causal effect of diuretic dose on AKI  
    ↓  
Step 3: Pathway exploration → explore the potential association of electrolyte disturbances  
    ↓  
Step 4: Heterogeneity analysis → identify high-risk patient subgroups  
    ↓  
Step 5: Sensitivity analysis → verify robustness of results  
```

---

## 4. Study Population

### 4.1 Inclusion Criteria

1. ICU inpatients
2. Age ≥ 18 years
3. Treated with diuretic therapy (furosemide injection or oral, or torasemide injection)
4. Available baseline serum creatinine and follow-up creatinine post-diuretic therapy
5. Length of stay ≥ 48 h (to allow observation of AKI occurrence)

### 4.2 Exclusion Criteria

1. **Pre-existing end-stage renal disease at admission**

   * On maintenance dialysis
   * History of renal transplantation
2. **Data quality issues**

   * Key variables missing >40%
   * Clear data entry errors
3. **Special populations**

   * Pregnant patients (creatinine reference ranges differ)
   * Admission due to renal-transplant rejection (special population)

### 4.3 Sample Size

**Actual sample size**: 2,397 ICU patients.

**Justification of adequacy**:

* Based on α = 0.05 (two-sided), power = 0.80, expected effect size OR ≈1.5, baseline AKI incidence ~15-20%, high-dose group ~25%, the required total sample size ~1,500 patients.
* Actual sample size of 2,397 exceeds this, providing sufficient power and enabling subgroup analyses.

---

## 5. Variable Definitions

### 5.1 Primary Exposure Variable

**Total diuretic dose** (continuous variable, unit: mg).

* Extract all diuretic administration records during hospitalization.
* Calculate cumulative dose (converted to furosemide-equivalent): torasemide 1 mg = furosemide 2 mg.
* Dose categorisation:

  1. Quartiles (main analysis): Q1 (≤P25) / Q2 (P25-P50) / Q3 (P50-P75) / Q4 (>P75)
  2. Binary grouping (sensitivity): low (≤80 mg or ≤P75) vs high (>80 mg)
  3. Continuous variable (exploratory)

### 5.2 Primary Outcome Variable

**Acute Kidney Injury (AKI)**

* Definition: according to KDIGO Clinical Practice Guideline for Acute Kidney Injury (2012) criteria:

  1. Increase in serum creatinine ≥26.5 µmol/L (0.3 mg/dL) within 48 h; or
  2. Increase in serum creatinine ≥1.5 × baseline (known or presumed within prior 7 days).
* Staging:

  * Stage 1: creatinine 1.5-1.9× baseline OR ≥26.5 µmol/L rise
  * Stage 2: creatinine 2.0-2.9× baseline
  * Stage 3: creatinine ≥3.0× baseline OR ≥353.6 µmol/L OR initiation of renal replacement therapy
* Baseline creatinine: preferentially the lowest value within 7 days pre-admission; if unavailable, use value at ICU admission.
* Measurement window: baseline before/at ICU admission or before diuretic; outcome within 48 h after diuretic or during ICU stay.

### 5.3 Secondary Outcome Variables

1. **Electrolyte disturbances**:

   * Hyponatraemia: Na < 135 mmol/L
   * Hypernatraemia: Na > 145 mmol/L
   * Hypokalaemia: K < 3.5 mmol/L
   * Hyperkalaemia: K > 5.5 mmol/L
   * Any electrolyte disorder (binary)
2. **Creatinine change**:

   * Absolute change (µmol/L)
   * Relative change (ratio)
3. **AKI severity**: ordinal variable (0/1/2/3)

### 5.4 Confounders

**Selection principle**:

1. Based on clinical knowledge and literature: factors influencing both diuretic dosing and AKI risk.
2. Screened via univariate analysis (P < 0.2)
3. Exclude variables with multicollinearity (VIF < 10)

**Potential confounders**:
A. Demographics: age, sex, ethnicity
B. Baseline renal function: baseline creatinine, eGFR (MDRD), CKD stage
C. Vital signs: systolic BP, diastolic BP
D. Labs: WBC, Hb, platelets, Hct; liver function: total bilirubin, albumin, globulin, AST/ALT; BUN
E. Comorbidities: diabetes, hypertension, heart failure history, coronary artery disease, sepsis, oedema, brain injury, cardiac surgery, cancer
F. Concomitant medications: NSAIDs, ACEI/ARB, contrast media, aminoglycosides, immunosuppressants; others: β-blockers, CCBs, statins, PPIs, hypoglycaemic drugs, anticoagulants
G. Diuretic–related variables: diuretic type (furosemide vs torasemide), route (IV vs oral), mannitol dose (continuous)

### 5.5 Derived Variables

1. eGFR_before, eGFR_after, CKD_stage
2. AKI_binary (0/1), AKI_stage (0/1/2/3)
3. dose_quartile, high_dose (0/1)
4. hyponatraemia, hypernatraemia, hypokalaemia, hyperkalaemia, electrolyte_disorder

---

## 6. Sample Size Calculation

### 6.1 Theoretical Sample Size

Based on comparing high-dose vs low-dose groups:

* α = 0.05 (two-sided)
* Power 1-β = 0.80
* Control group (low dose) AKI incidence p₀ = 0.15
* Treatment group (high dose) AKI incidence p₁ = 0.22
* Expected OR ≈1.6 (RR ≈1.47)
* Using two-sample comparison of proportions → about 750 in each group → ~1,500 total.

### 6.2 Actual Sample Size

* Actual included: 2,397 patients.
* Approximate grouping: low-dose (Q1-Q3) ~1,800; high-dose (Q4) ~600.
* Thus sample size is sufficient, enabling subgroup and heterogeneity analyses.

### 6.3 Post-hoc Power Analysis

With actual N and high-dose proportion ~25%:

* Detecting OR = 1.5 → power > 95%
* Detecting OR = 1.3 → power > 80%
  Hence the study has adequate statistical power to detect clinically meaningful effects.

---

## 7. Statistical Analysis Plan

(See separate SAP and `methods_details.md`.)

### 7.1 Software

* **Python 3.8+**: data preprocessing, ML, causal inference (pandas, scikit-learn, xgboost, econml, shap)
* **R 4.0+**: causal forests, sensitivity analyses (grf, MatchIt, cobalt, EValue)
* Significance level α = 0.05 (two-sided).

### 7.2 Descriptive Statistics

* Baseline characteristics by AKI stage:

  * Continuous: mean±SD or median (IQR)
  * Categorical: count (%)
  * Comparisons: t-test or Mann-Whitney U; χ² or Fisher’s exact.
* Dose distribution: group sizes, AKI incidence per dose-group, dose summary (mean, median, range).

### 7.3 Five-Step Framework

**Step 1: Predictive modelling**

* Purpose: Identify AKI risk factors; assess predictive importance of diuretic dose.
* Methods: Logistic regression, Random Forest, XGBoost.
* Evaluation: AUC-ROC, AUPRC, calibration, SHAP values.

**Step 2: Causal inference**

* Purpose: Estimate causal effect of diuretic dose on AKI.
* Methods: Propensity score matching (PSM), inverse probability of treatment weighting (IPTW), doubly robust estimation.
* Effect estimates: ATE, RD, RR, OR with 95% CI.

**Step 3: Pathway exploration**

* Purpose: Explore potential association of electrolyte disturbances in dose → AKI path.
* Methods: Path analysis, stratified analysis.
* Note: only association (single-time-point data), not causal mediation.

**Step 4: Heterogeneity analysis**

* Purpose: Identify high-risk patient subgroups.
* Methods: Traditional subgroup analysis, T-Learner, causal forests.
* Output: Risk scoring system.

**Step 5: Sensitivity analysis**

* Purpose: Check robustness.
* Methods: E-value, method-comparison, exclusion analyses.

### 7.4 Missing Data

* Report missingness per variable.
* Analyse missing- vs non-missing baseline differences.
* Variables missing >40%: exclude.
* Variables missing ≤40%: Multiple imputation (MICE), m = 5, PMM method.
* Sensitivity: complete-case analysis.

### 7.5 Outlier Handling

* Detect extreme values: boxplot, Z-score (|Z|>3).
* Handle by winsorisation (1st & 99th percentiles).
* Data entry errors: corrected or removed after clinical review.

---

## 8. Quality Control

### 8.1 Data Quality

* Dual independent extraction of key variables; discrepancies adjudicated by third reviewer.
* Double data entry validation.
* Logic checks (e.g., post-medication creatinine vs baseline).
* Range checks (e.g., age 0–120).
* Time logic checks (medication time < outcome time).

### 8.2 Analysis Quality

* Version control (Git).
* Code review.
* Detailed comments.
* Two analysts independently perform main analysis and cross-validate.
* Sensitivity analyses for robustness.

### 8.3 Bias Control

* Selection bias: explicit inclusion/exclusion criteria; flow-chart of screening process.
* Information bias: use standard definitions (KDIGO); blinding of outcome assessment when feasible.
* Confounding bias: systematic identification, causal inference methods, sensitivity analysis for residual confounding.

---

## 9. Ethics Considerations

### 9.1 Ethics Review

Approved by ethics committees of both sites (approval numbers to be filled).

### 9.2 Informed Consent

Retrospective design using de-identified data; ethics committees granted waiver of informed consent.

### 9.3 Privacy Protection

* Remove direct identifiers (name, ID number).
* Remove indirect identifiers (specific admission dates replaced by relative days).
* Use study IDs.
* Store data on encrypted drive; access control; encrypted transmission; comply with PRC Personal Information Protection Law.

### 9.4 Risks & Benefits

**Risks**: minimal (retrospective, no intervention); risk of privacy breach (mitigated by above).
**Benefits**: no direct benefit to participants; indirect benefit: improved future patient management.

### 9.5 Conflict of Interest

Investigators declare no conflicts of interest.

---

## 10. Study Timeline

| Phase                    | Time         | Major Tasks                                           |
| ------------------------ | ------------ | ----------------------------------------------------- |
| Preparation              | Jan–Feb 2024 | Protocol design, ethics approval, literature review   |
| Data collection          | Mar–Apr 2024 | Data extraction, data entry, verification             |
| Data cleaning            | May–Jun 2024 | Data cleaning, QC, variable derivation                |
| Predictive modelling     | Jul–Aug 2024 | Feature engineering, model training, evaluation       |
| Causal inference         | Sep–Oct 2024 | PSM/IPTW analyses, dose–response trend                |
| Pathway exploration      | Nov 2024     | Electrolyte association analysis, stratified analyses |
| Heterogeneity analysis   | Dec 2024     | Subgroup analyses, causal forests, risk scoring       |
| Sensitivity analysis     | Jan 2025     | E-value, method comparisons, exclusion analyses       |
| Results consolidation    | Feb–Mar 2025 | Table/chart creation, interpretation                  |
| Manuscript writing       | Apr–Jun 2025 | Drafting, internal review, revising                   |
| Submission & publication | Jul–Dec 2025 | Journal submission, revision, publication             |

---

## 11. Expected Outcomes

### 11.1 Scientific Findings (anticipated)

1. **Dose–response relationship**

   * Confirm causal relationship between diuretic dose and AKI risk.
   * Describe shape of dose–response (linear, threshold, nonlinear).
   * Identify high-risk dose threshold (anticipated ~80 mg).
2. **High-risk population identification**

   * Baseline eGFR < 60 mL/min/1.73 m²
   * Older age (≥65 years)
   * Diabetes mellitus
   * History of heart failure
   * Concurrent ACEI/ARB use
3. **Role of electrolyte disturbances**

   * High dose increases electrolyte disorder risk.
   * Electrolyte disorder associated with AKI risk.
   * Electrolyte disorder partially explains the dose → AKI link.
4. **Clinical decision tool**

   * Risk scoring system (0-6 points).
   * Risk stratification (low/medium/high).
   * Dose-optimization recommendations.

### 11.2 Academic Output

**Publications**: Aim for SCI Q1 journals (e.g., *Critical Care Medicine*, *Intensive Care Medicine*, *JAMA Network Open*, *Kidney International*) — expected IF >5.0.
**Conference presentations**: Chinese Society of Nephrology, ASN, ESICM annual meetings.
**Other outputs**: Open-source code and data (in compliance with ethics/privacy), GitHub repository with reproducible workflow, guideline-reference evidence.

### 11.3 Clinical Significance

**Practice guidance**:

* Evidence-based diuretic dose selection.
* Identification of high-risk patients for close monitoring.
  **Improve patient outcomes**:
* Reduce drug-related AKI.
* Lower ICU/hospital stay and mortality.
* Decrease healthcare costs.
  **Policy impact**:
* Provide evidence for guideline development.
* Promote rational diuretic use.

---

## 12. References
   1. Kellum JA, Lameire N; KDIGO AKI Guideline Work Group. KDIGO clinical practice guideline for acute kidney injury. *Kidney Int Suppl*. 2012;2(1):1-138.
   2. Hoste EA, Bagshaw SM, Bellomo R, et al. Epidemiology of acute kidney injury in critically ill patients: the multinational AKI-EPI study. *Intensive Care Med*. 2015;41(8):1411-1423.
   3. Mehta RL, Cerdá J, Burdmann EA, et al. International Society of Nephrology’s 0by25 initiative for acute kidney injury (zero preventable deaths by 2025). *Lancet*. 2015;385(9987):2616-2643.
   4. Kellum JA, Ronco C. Acute kidney injury. *Lancet*. 2024;403(10435):203–218. ([Lancet][1])
   5. Ostermann M, Zarbock A, Goldstein S, et al. Recommendations on acute kidney injury biomarkers from the Acute Disease Quality Initiative (ADQI) Consensus Conference. *JAMA Netw Open*. 2020;3(10):e2019209.
   6. Ostermann M, Ostermann J, Ronco C. Treatment of acute kidney injury: a review of current approaches and emerging innovations. *J Clin Med*. 2023;13(9):2455. ([MDPI][2])
   7. Chatzis G, Hristara-Papaiouanou A, et al. Novel diagnostic and prognostic methods in acute kidney injury. *Front Nephrol*. 2025;2:1586794. ([Frontiers][3])
   8. Marchiset A, Jamme M. When the renal (function) begins to fall: a mini-review of acute kidney injury related to acute respiratory distress syndrome in critically ill patients. *Front Nephrol*. 2022;2:877529. ([Frontiers][4])
   9. Monnet X, Teboul JL, et al. Fluid management in acute kidney injury: from evaluating fluid responsiveness to assessing fluid tolerance. *Eur Heart J Acute Cardiovasc Care*. 2022;11(10):786-793. ([OUP Academic][5])
   10. Beaubien-Souligny W, Eljaiek R, et al. Venous excess Doppler ultrasound (VExUS) for the nephrologist: pearls and pitfalls. *Kidney Med*. 2022;4(6):100482. ([CoLab][6])
   11. Dhanesh D, et al. Prognostic value of venous congestion assessed by VExUS score in ICU patients. *Afr J Biomed Sci*. 2024; (online ahead of print). ([afjbs.com][7])
   12. Prowle JR, Kirwan CJ, Bellomo R. Fluid management for the prevention and attenuation of acute kidney injury. *Nat Rev Nephrol*. 2014;10(1):37-47.
   13. Wichmann S, Barbateskovic M, Liang N, et al. Loop diuretics in adult intensive care patients with fluid overload: a systematic review with meta-analysis and trial sequential analysis. *Ann Intensive Care*. 2022;12:52. ([SpringerOpen][8])
   14. Piñeiro G, Cucchiari D, Jacas A, et al. New insights into diuretic use to treat congestion in the ICU. *Front Nephrol*. 2022;2:879766. ([Frontiers][9])
   15. Ferrari F, Zanza C, Tesauro M, et al. Clinical pharmacology of loop diuretics in critical care. *Clin Pharmacokinet*. 2025;64:987-997. ([SpringerLink][10])
   16. Ostermann M, Awdishu L, Legrand M. Using diuretic therapy in the critically ill patient. *Intensive Care Med*. 2024;50(8):1331-1334. ([SpringerLink][11])
   17. Elhadi M, et al. Diuretics in critically ill patients: a narrative review of their use, benefits and harms. *Br J Anaesth*. 2025; (in press). ([Science Direct][12])
   18. Sharma S, et al. Diuretics in acute kidney injury. *Indian J Crit Care Med*. 2022;26(Suppl 3):S131-S139. ([ijccm.org][13])
   19. Zhang Y, et al. Association between the use of loop diuretics and prognosis in critically ill patients with acute kidney injury: a propensity-score matched analysis of the MIMIC-IV database. *Int Urol Nephrol*. 2024;56(6):1241-1253. ([SpringerLink][14])
   20. Kotani Y, Yoshida T; BROTHER Study Group. Effect of early diuretic administration on AKI progression after cardiac surgery: a post-hoc analysis of a multicenter retrospective cohort study. *Signa Vitae*. 2023;19(6):175-183. ([signavitae.com][15])
   21. Perner A, Bestle MH, et al. Loop diuretics and mortality in patients with acute kidney injury. In: Ronco C, Bellomo R, Kellum JA, eds. *Critical Care Nephrology*. Springer; 2016:321-332. ([SpringerLink][16])
   22. Wichmann S, et al. Diuretic strategies in patients with resistance to loop diuretics in the intensive care unit. *Heart Lung*. 2021;50(4):518-526. ([Science Direct][17])
   23. Ostermann M. AKI management: diuretics. In: Ronco C, ed. *Acute Kidney Injury and Critical Care Nephrology*. Springer; 2024: Chapter 33. ([SpringerLink][18])
   24. Zhang L, et al. Association between loop diuretics and mortality in patients with cardiac surgery-associated acute kidney injury. *Crit Care Explor*. 2023;5(11):e0934. ([X-MOL][19])
   25. Kim J, et al. Rethinking diuretic use in acute kidney injury: effective prevention or uncertain treatment? *J Intensive Care*. 2025;13:16. ([BioMed Central][20])
   26. Iida N, et al. The utility of point-of-care ultrasound in critical care nephrology and assessment of venous congestion. *Front Nephrol*. 2024;3:1402641. ([Frontiers][21])
   27. Austin PC. An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behav Res*. 2011;46(3):399-424.
   28. Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. *J Am Stat Assoc*. 2018;113(523):1228-1242.
   29. VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Ann Intern Med*. 2017;167(4):268-274.
   30. Hernán MA, Robins JM. *Causal Inference: What If*. Boca Raton: Chapman & Hall/CRC; 2020.

[1]: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736%2824%2902385-7/fulltext?utm_source=chatgpt.com "Acute kidney injury - The Lancet"
[2]: https://www.mdpi.com/2077-0383/13/9/2455?utm_source=chatgpt.com "Treatment of Acute Kidney Injury: A Review of Current Approaches and ..."
[3]: https://www.frontiersin.org/journals/nephrology/articles/10.3389/fneph.2025.1586794/full?utm_source=chatgpt.com "Editorial: Novel diagnostic and prognostic methods in acute kidney ..."
[4]: https://www.frontiersin.org/journals/nephrology/articles/10.3389/fneph.2022.877529/full?utm_source=chatgpt.com "Frontiers | When the Renal (Function) Begins to Fall: A Mini-Review of ..."
[5]: https://academic.oup.com/ehjacc/article/11/10/786/6693622?utm_source=chatgpt.com "Fluid management in acute kidney injury: from evaluating fluid ..."
[6]: https://colab.ws/articles/10.1016%2Fj.xkme.2022.100482?utm_source=chatgpt.com "Venous Excess Doppler Ultrasound for the Nephrologist: Pearls and ..."
[7]: https://www.afjbs.com/uploads/paper/6f0897a1cdba726934de9ec7a7b04a13.pdf?utm_source=chatgpt.com "Dhanesh_Article_2_2024 4 - afjbs.com"
[8]: https://annalsofintensivecare.springeropen.com/articles/10.1186/s13613-022-01024-6?utm_source=chatgpt.com "Loop diuretics in adult intensive care patients with fluid overload: a ..."
[9]: https://www.frontiersin.org/journals/nephrology/articles/10.3389/fneph.2022.879766/pdf?utm_source=chatgpt.com "New Insights Into Diuretic Use to Treat Congestion in the ... - Frontiers"
[10]: https://link.springer.com/article/10.1007/s40262-025-01524-1?utm_source=chatgpt.com "Clinical Pharmacology of Loop Diuretics in Critical Care"
[11]: https://link.springer.com/article/10.1007/s00134-024-07441-4?utm_source=chatgpt.com "Using diuretic therapy in the critically ill patient"
[12]: https://www.sciencedirect.com/science/article/pii/S000709122500159X?utm_source=chatgpt.com "Diuretics in critically ill patients: a narrative review of their ..."
[13]: https://www.ijccm.org/doi/pdf/10.5005/jp-journals-10071-23406?utm_source=chatgpt.com "INVITED ARTICLE Diuretics in Acute Kidney Injury - ijccm.org"
[14]: https://link.springer.com/article/10.1007/s11255-024-04271-7?utm_source=chatgpt.com "Association between the use of loop diuretics and prognosis in ..."
[15]: https://www.signavitae.com/articles/10.22514/sv.2023.112?utm_source=chatgpt.com "The effect of early diuretics administration on acute kidney injury ..."
[16]: https://link.springer.com/chapter/10.1007/978-3-319-33429-5_21?utm_source=chatgpt.com "Loop Diuretics and Mortality in Patients with Acute Kidney Injury"
[17]: https://www.sciencedirect.com/science/article/pii/S0883944121001222?utm_source=chatgpt.com "Diuretic strategies in patients with resistance to loop-diuretics in ..."
[18]: https://link.springer.com/chapter/10.1007/978-3-031-66541-7_33?utm_source=chatgpt.com "AKI Management: Diuretics | SpringerLink"
[19]: https://www.x-mol.com/paper/1732451440050065408?utm_source=chatgpt.com "Association Between Loop Diuretics and Mortality in ..."
[20]: https://jintensivecare.biomedcentral.com/articles/10.1186/s40560-025-00804-z?utm_source=chatgpt.com "Rethinking diuretic use in acute kidney injury: effective prevention or ..."
[21]: https://www.frontiersin.org/journals/nephrology/articles/10.3389/fneph.2024.1402641/full?utm_source=chatgpt.com "Frontiers | The utility of point-of-care ultrasound in critical care ..."


---

**Version history**

| Version | Date       | Revision content                      | Revised by |
| ------- | ---------- | ------------------------------------- | ---------- |
| 1.0     | 2024-01-15 | Initial version                       | Liu Hao    |
| 1.1     | 2024-03-10 | Added sample size calculation details | Liu Hao    |
| 1.2     | 2024-05-20 | Updated statistical analysis plan     | Liu Hao    |

Contact information: Liu Hao, email: [lenhartkoo@foxmail.com](mailto:lenhartkoo@foxmail.com); China Pharmaceutical University – State Key Laboratory of Natural Drug Active Components and Pharmacodynamics; Address: Nanjing 210009, Jiangsu Province, PR China.

> This study protocol is in accordance with the Declaration of Helsinki and Chinese regulations on medical research ethics.

---