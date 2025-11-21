# Data Dictionary

## Overview

This document provides detailed descriptions of all variables in the dataset.

**Dataset:** Multi-center ICU patient cohort  
**Sample Size:** 2,397 patients  
**Variables:** 67  
**File:** `1763120918461_副本新疆附一_湘雅医院_多中心_利尿剂AKI_2_.csv`

---

## Variable Categories

- [Identifiers](#identifiers)
- [Demographics](#demographics)
- [Primary Exposure](#primary-exposure)
- [Primary Outcome](#primary-outcome)
- [Vital Signs](#vital-signs)
- [Electrolytes](#electrolytes)
- [Hematology](#hematology)
- [Liver Function](#liver-function)
- [Kidney Function](#kidney-function)
- [Medications](#medications)
- [Comorbidities](#comorbidities)

---

## Variable Details

### Identifiers

| Variable | Chinese Name | Type | Description |
|----------|--------------|------|-------------|
| ID | ID | Integer | Record ID |
| PatientID | PatientID | String | Unique patient identifier |

---

### Demographics

| Variable | Chinese Name | Type | Values | Description |
|----------|--------------|------|--------|-------------|
| age | 年龄 | Integer | - | Age in years |
| sex | 性别 | Binary | 0=Male, 1=Female | Patient sex |
| ethnicity | 民族 | Categorical | 1=Han<br>2=Uyghur<br>3=Kazakh<br>4=Hui<br>5=Other | Ethnicity |

---

### Primary Exposure

| Variable | Chinese Name | Type | Unit | Description |
|----------|--------------|------|------|-------------|
| 利尿剂总剂量 | Diuretic total dose | Continuous | mg | **PRIMARY EXPOSURE**<br>Total diuretic dose |
| 药品分类 | Drug classification | Categorical | 1=Furosemide injection<br>2=Furosemide tablet<br>3=Torasemide injection | Type of diuretic |
| 给药途径 | Administration route | Categorical | 1=Intravenous<br>2=Intramuscular<br>3=Oral/Other | Route of administration |
| 甘露醇 dose | Mannitol dose | Continuous | mg | Mannitol dose (confounder) |

---

### Primary Outcome

| Variable | Chinese Name | Type | Values | Description |
|----------|--------------|------|--------|-------------|
| **AKI class** | AKI分级 | Ordinal | 0=No AKI<br>1=Stage 1<br>2=Stage 2<br>3=Stage 3 | **PRIMARY OUTCOME**<br>AKI stage (KDIGO) |
| 用药前肌酐 | Creatinine before | Continuous | μmol/L | Baseline creatinine |
| 用药后肌酐 | Creatinine after | Continuous | μmol/L | Post-treatment creatinine |
| AKI差值 | AKI difference | Continuous | μmol/L | Absolute change in creatinine<br>= After - Before |
| AKI比值 | AKI ratio | Continuous | - | Relative change in creatinine<br>= After / Before |

**AKI Definition (KDIGO Criteria):**
- **Stage 1:** Cr increase 1.5-1.9× baseline OR increase ≥26.5 μmol/L (0.3 mg/dL)
- **Stage 2:** Cr increase 2.0-2.9× baseline
- **Stage 3:** Cr increase ≥3.0× baseline OR ≥353.6 μmol/L (4.0 mg/dL) OR RRT

---

### Vital Signs

| Variable | Chinese Name | Type | Unit | Normal Range |
|----------|--------------|------|------|--------------|
| 收缩压 | Systolic BP | Continuous | mmHg | 90-140 |
| 舒张压 | Diastolic BP | Continuous | mmHg | 60-90 |

---

### Electrolytes (Secondary Outcomes)

| Variable | Chinese Name | Type | Unit | Normal Range | Abnormality Definition |
|----------|--------------|------|------|--------------|------------------------|
| 钙测定 | Calcium | Continuous | mmol/L | 2.1-2.6 | - |
| 氯测定 | Chloride | Continuous | mmol/L | 96-108 | - |
| 钠测定 | Sodium | Continuous | mmol/L | 136-145 | Hypo: <135<br>Hyper: >145 |
| 钾测定 | Potassium | Continuous | mmol/L | 3.5-5.3 | Hypo: <3.5<br>Hyper: >5.5 |

---

### Hematology

| Variable | Chinese Name | Type | Unit | Normal Range |
|----------|--------------|------|------|--------------|
| 白细胞 | WBC | Continuous | ×10⁹/L | 3.5-9.5 |
| 血小板 | Platelet | Continuous | ×10⁹/L | 125-350 |
| 红细胞压积 | Hematocrit | Continuous | % | M: 40-50<br>F: 36-45 |
| 血红蛋白 | Hemoglobin | Continuous | g/L | M: 130-175<br>F: 115-150 |
| 血小板平均体积 | MPV | Continuous | fL | 7.4-10.4 |
| 血小板分布宽度 | PDW | Continuous | % | 10-18 |
| 单核细胞百分比 | Monocyte % | Continuous | % | 3-10 |
| 淋巴细胞百分比 | Lymphocyte % | Continuous | % | 20-50 |
| 中性粒细胞百分比 | Neutrophil % | Continuous | % | 40-75 |
| 平均红细胞体积 | MCV | Continuous | fL | 82-100 |

---

### Liver Function

| Variable | Chinese Name | Type | Unit | Normal Range |
|----------|--------------|------|------|--------------|
| 总胆红素 | Total bilirubin | Continuous | μmol/L | 3.4-20.5 |
| 直接胆红素 | Direct bilirubin | Continuous | μmol/L | 0-6.8 |
| 血清白蛋白 | Albumin | Continuous | g/L | 40-55 |
| 球蛋白 | Globulin | Continuous | g/L | 20-30 |
| 总蛋白 | Total protein | Continuous | g/L | 65-85 |
| 总胆汁酸 | Total bile acid | Continuous | μmol/L | 0-10 |
| 天门冬氨酸氨基转移酶 | AST | Continuous | U/L | 15-40 |
| 丙氨酸氨基转移酶 | ALT | Continuous | U/L | 9-50 |

---

### Kidney Function

| Variable | Chinese Name | Type | Unit | Normal Range |
|----------|--------------|------|------|--------------|
| 尿素 | BUN | Continuous | mmol/L | 3.1-8.0 |

**Calculated Variables:**
- **eGFR** (estimated Glomerular Filtration Rate): Calculated using MDRD formula
- **CKD Stage**: Based on eGFR
  - Stage 1: eGFR ≥90 ml/min/1.73m²
  - Stage 2: eGFR 60-89
  - Stage 3: eGFR 30-59
  - Stage 4: eGFR 15-29
  - Stage 5: eGFR <15

---

### Medications

All medication variables are **binary** (0=No, 1=Yes).

#### Nephrotoxic or Kidney-Affecting Drugs

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| NSAID | NSAID | Non-steroidal anti-inflammatory drugs |
| 造影剂 | Contrast agent | Radiographic contrast agents |
| 抗生素 | Antibiotic | Antibiotics (including aminoglycosides) |
| 免疫抑制剂 | Immunosuppressant | Immunosuppressive agents |

#### Cardiovascular Drugs

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| β拮抗剂 | Beta blocker | β-adrenergic receptor antagonists |
| 抗心律失常药物 | Antiarrhythmic | Antiarrhythmic drugs |
| 钙阻滞剂 | Calcium blocker | Calcium channel blockers |
| 抗心绞痛药物 | Antianginal | Antianginal medications |
| ACEI | ACEI | ACE inhibitors |
| ARB | ARB | Angiotensin receptor blockers |
| 地高辛 | Digoxin | Digoxin |

#### Antithrombotic Drugs

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| 氯吡格雷 | Clopidogrel | Clopidogrel |
| 华法林 | Warfarin | Warfarin |

#### Other Medications

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| 质子泵抑制剂 | PPI | Proton pump inhibitors |
| 口服降糖药 | Oral hypoglycemic | Oral hypoglycemic agents |
| 胰岛素 | Insulin | Insulin |
| 他汀 | Statin | Statin drugs |

---

### Comorbidities

All comorbidity variables are **binary** (0=No, 1=Yes).

#### Acute Conditions

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| 心脏手术 | Cardiac surgery | History of cardiac surgery |
| 高血压 | Hypertension | Hypertension (current) |
| 水肿 | Edema | Edema |
| 脑损伤 | Brain injury | Brain injury |
| 脓毒症 | Sepsis | Sepsis |
| 糖尿病 | Diabetes | Diabetes mellitus (current) |

#### Chronic Medical History

| Variable | Chinese Name | Description |
|----------|--------------|-------------|
| 高血压史 | Hypertension history | History of hypertension |
| 糖尿病史 | Diabetes history | History of diabetes mellitus |
| 冠心病史 | CAD history | History of coronary artery disease |
| 慢性心衰史 | CHF history | History of chronic heart failure |
| 癌症史 | Cancer history | History of cancer |

---

## Data Quality

### Missing Data
- Variables with >40% missing: [To be determined during analysis]
- Imputation method: Multiple Imputation by Chained Equations (MICE)

### Outliers
- Handled using Winsorization at 1st and 99th percentiles
- Extreme values reviewed for data entry errors

### Data Validation
- Range checks performed on all continuous variables
- Consistency checks between related variables (e.g., creatinine before/after)

---

## Variable Selection for Analysis

### Confounders (for Causal Inference)
Selected based on:
1. Clinical knowledge
2. Literature review
3. Univariate association (P<0.2)
4. Multicollinearity check (VIF<10)

### Final Confounder Set
Approximately 20-30 variables selected after screening.

---

## Derived Variables

Variables calculated during preprocessing:

1. **eGFR_before**: Baseline eGFR (MDRD formula)
2. **eGFR_after**: Post-treatment eGFR
3. **CKD_stage**: Chronic kidney disease stage
4. **AKI_binary**: Binary AKI (0=No, 1=Yes)
5. **Dose_quartile**: Diuretic dose quartiles (Q1-Q4)
6. **High_dose**: Binary high dose (>80mg)
7. **Electrolyte_disorder**: Any electrolyte abnormality
8. **Hyponatremia**: Sodium <135 mmol/L
9. **Hypernatremia**: Sodium >145 mmol/L
10. **Hypokalemia**: Potassium <3.5 mmol/L
11. **Hyperkalemia**: Potassium >5.5 mmol/L

---

## Notes

1. All Chinese variable names are preserved from the original dataset
2. File encoding: UTF-8 with BOM
3. Decimal separator: Period (.)
4. Missing values: Empty cells or specific codes (to be defined)
5. Date format: [If applicable]

---

## Contact

For questions about the data dictionary:
- Email: lenhartkoo@foxmail.com

---

**Last Updated:** 2025-11-14  
**Version:** 1.0
