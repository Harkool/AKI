import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "AKI.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "AKI_processed.csv")


rename_dict = {
    # 基本信息 / Basic Information
    "ID": "id",
    "PatientID": "patient_id",
    "age": "age",
    "sex，男=0，女=1": "sex",  
    "民族，汉族=1，维吾尔族=2，哈萨克族=3，回族=4，其他=5": "ethnicity",
    
    # 利尿剂相关 / Diuretic-related
    "甘露醇 dose": "mannitol_dose",
    "利尿剂总剂量": "diuretic_total_dose", 
    "药品分类，呋塞米注射液=1，呋塞米片剂=2，托拉塞米注射液=3": "diuretic_type",
    "给药途径，静脉给药=1，肌肉注射=2，口服及其他=3": "admin_route",
    
    # 肾功能指标 / Kidney Function
    "用药前肌酐": "creatinine_pre",
    "用药后肌酐": "creatinine_post",
    "AKI差值": "aki_change",
    "AKI比值": "aki_ratio",
    "AKI class，未发生=0，1级=1，2级=2，3级=3": "aki_stage",  # 主要结局 / Primary outcome
    
    # 生命体征 / Vital Signs
    "收缩压": "sbp",
    "舒张压": "dbp",
    
    # 电解质 / Electrolytes
    "钙测定": "calcium",
    "氯测定": "chloride",
    "钠测定": "sodium",
    "钾测定": "potassium",
    
    # 血常规 / Blood Routine
    "白细胞": "wbc",
    "血小板": "platelet",
    "红细胞压积": "hematocrit",
    "血红蛋白": "hemoglobin",
    "血小板平均体积": "mpv",
    "血小板分布宽度": "pdw",
    "单核细胞百分比": "monocyte_pct",
    "淋巴细胞百分比": "lymphocyte_pct",
    "中性粒细胞百分比": "neutrophil_pct",
    "平均红细胞体积": "mcv",
    
    # 肝功能 / Liver Function
    "总胆红素": "total_bilirubin",
    "血清白蛋白": "albumin",
    "总胆汁酸": "total_bile_acid",
    "球蛋白": "globulin",
    "总蛋白": "total_protein",
    "直接胆红素": "direct_bilirubin",
    "天门冬氨酸氨基转移酶": "ast",
    "丙氨酸氨基转移酶": "alt",
    
    # 其他实验室指标 / Other Lab Values
    "尿素": "bun",
    
    # 合并用药 / Concomitant Medications
    "NSAID": "nsaid",
    "β拮抗剂": "beta_blocker",
    "抗心律失常药物": "antiarrhythmic",
    "钙阻滞剂": "calcium_blocker",
    "质子泵抑制剂": "ppi",
    "口服降糖药": "oral_hypoglycemic",
    "胰岛素": "insulin",
    "造影剂": "contrast_agent",
    "抗心绞痛药物": "antianginal",
    "ACEI": "acei",
    "ARB": "arb",
    "抗生素": "antibiotics",
    "免疫抑制剂": "immunosuppressant",
    "他汀": "statin",
    "氯吡格雷": "clopidogrel",
    "华法林": "warfarin",
    "地高辛": "digoxin",
    
    # 合并症 / Comorbidities
    "心脏手术": "cardiac_surgery",
    "高血压": "hypertension",
    "水肿": "edema",
    "脑损伤": "brain_injury",
    "脓毒症": "sepsis",
    "糖尿病": "diabetes",
    
    # 既往史 / Medical History
    "高血压史": "hypertension_history",
    "糖尿病史": "diabetes_history",
    "冠心病史": "cad_history",
    "慢性心衰史": "chf_history",
    "癌症史": "cancer_history"
}

def load_data(path=DATA_PATH):
    """加载数据并重命名 / Load and rename data"""
    print("="*70)
    print("步骤 1: 数据加载 / Step 1: Data Loading")
    print("="*70)
    print(f"从以下路径加载数据 / Loading from: {path}")
    
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"原始数据形状 / Original shape: {df.shape}")
    
    df = df.rename(columns=rename_dict)
    print(f"变量重命名完成 / Renaming completed: {len(rename_dict)} variables")
    
    return df

def data_quality_report(df):
    """数据质量报告 / Data quality report"""
    print("\n" + "="*70)
    print("数据质量报告 / Data Quality Report")
    print("="*70)
    
    # 缺失值统计
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        '变量': missing.index,
        '缺失数': missing.values,
        '缺失率(%)': missing_pct.values
    })
    missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失率(%)', ascending=False)
    
    print(f"\n有缺失的变量数 / Variables with missing: {len(missing_df)}")
    if len(missing_df) > 0:
        print(f"\n缺失率前10 / Top 10 by missing rate:")
        print(missing_df.head(10).to_string(index=False))
    
    # 重复值
    dup = df.duplicated(subset=['patient_id']).sum()
    print(f"\n重复患者ID / Duplicate IDs: {dup}")
    
    return missing_df


def remove_outliers_iqr(df, cols, factor=1.5):
    """IQR方法处理异常值 / Handle outliers using IQR"""
    print("\n使用IQR方法处理异常值 / Handling outliers (IQR method)")
    
    df_clean = df.copy()
    outlier_counts = {}
    
    for col in cols:
        if df_clean[col].dtype in [np.float64, np.int64]:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            
            outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
                df_clean[col] = np.clip(df_clean[col], lower, upper)
    
    print(f"含异常值的变量 / Variables with outliers: {len(outlier_counts)}")
    return df_clean


def impute_missing(df):
    """MICE插补 / MICE imputation"""
    print("\n使用MICE插补缺失值 / Imputing with MICE")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    impute_cols = [col for col in num_cols if col not in ['id', 'patient_id']]
    
    missing_before = df[impute_cols].isnull().sum().sum()
    
    imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])
    
    missing_after = df[impute_cols].isnull().sum().sum()
    print(f"插补前缺失 / Before: {missing_before}, 插补后 / After: {missing_after}")
    
    return df


def one_hot_encode(df, max_categories=20):
    """智能One-Hot编码 / Smart one-hot encoding"""
    print("\n执行One-Hot编码 / Performing one-hot encoding")
    
    df = df.copy()
    cat_cols = []
    
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                if not pd.to_numeric(df[col], errors='coerce').isnull().all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    continue
            except:
                pass
            
            if df[col].nunique() > max_categories:
                print(f"跳过 / Skip: {col} ({df[col].nunique()} categories)")
                continue
            
            cat_cols.append(col)
    
    if cat_cols:
        print(f"编码变量数 / Encoding {len(cat_cols)} variables")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df


def feature_engineering(df):
    """特征工程 / Feature engineering"""
    print("\n创建衍生变量 / Creating derived features")
    
    df = df.copy()
    
    # eGFR (MDRD)
    if all(col in df.columns for col in ['creatinine_pre', 'age', 'sex']):
        cr_mg_dl = df['creatinine_pre'] / 88.4
        df['egfr'] = (
            175 * (cr_mg_dl ** -1.154) * (df['age'] ** -0.203) * 
            np.where(df['sex'] == 1, 0.742, 1.0)
        )
        print("  ✓ egfr (估算GFR / Estimated GFR)")
    
    # CKD分期 / CKD staging
    if 'egfr' in df.columns:
        def get_ckd_stage(egfr):
            if egfr >= 90: return 1
            elif egfr >= 60: return 2
            elif egfr >= 30: return 3
            elif egfr >= 15: return 4
            else: return 5
        df['ckd_stage'] = df['egfr'].apply(get_ckd_stage)
        print("  ✓ ckd_stage (CKD分期)")
    
    # AKI二分类
    if 'aki_stage' in df.columns:
        df['aki_binary'] = (df['aki_stage'] > 0).astype(int)
        print("  ✓ aki_binary (AKI二分类)")
    
    # 剂量分组
    if 'diuretic_total_dose' in df.columns:
        df['dose_quartile'] = pd.qcut(
            df['diuretic_total_dose'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
        )
        p75 = df['diuretic_total_dose'].quantile(0.75)
        df['high_dose'] = (df['diuretic_total_dose'] > p75).astype(int)
        print("  ✓ dose_quartile, high_dose (剂量分组)")
    
    # 电解质紊乱
    if all(col in df.columns for col in ['sodium', 'potassium']):
        df['hyponatremia'] = (df['sodium'] < 135).astype(int)
        df['hypernatremia'] = (df['sodium'] > 145).astype(int)
        df['hypokalemia'] = (df['potassium'] < 3.5).astype(int)
        df['hyperkalemia'] = (df['potassium'] > 5.5).astype(int)
        df['electrolyte_disorder'] = (
            (df['hyponatremia'] | df['hypernatremia'] | 
             df['hypokalemia'] | df['hyperkalemia'])
        ).astype(int)
        print("  ✓ 电解质紊乱指标 / Electrolyte disorders")
    
    return df


def scale_numeric(df):
    """标准化 / Standardization"""
    print("\n标准化数值变量 / Standardizing numeric variables")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['id', 'patient_id']
    
    # 排除二分类变量
    for col in num_cols:
        if df[col].nunique() == 2 and set(df[col].unique()).issubset({0, 1}):
            exclude_cols.append(col)
    
    scale_cols = [col for col in num_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    print(f"标准化变量数 / Scaled variables: {len(scale_cols)}")
    return df


def main():
    """主流程 / Main pipeline"""
    print("\n" + "="*70)
    print("AKI数据预处理流程 / AKI Data Preprocessing Pipeline")
    print("="*70 + "\n")
    
    # 1. 加载
    df = load_data()
    
    # 2. 质量检查
    print("\n" + "="*70)
    print("步骤 2: 数据质量检查 / Step 2: Quality Check")
    print("="*70)
    data_quality_report(df)
    
    # 3. 异常值
    print("\n" + "="*70)
    print("步骤 3: 异常值处理 / Step 3: Outlier Handling")
    print("="*70)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df = remove_outliers_iqr(df, num_cols)
    
    # 4. 插补
    print("\n" + "="*70)
    print("步骤 4: 缺失值插补 / Step 4: Imputation")
    print("="*70)
    df = impute_missing(df)
    
    # 5. 编码
    print("\n" + "="*70)
    print("步骤 5: 分类编码 / Step 5: Encoding")
    print("="*70)
    df = one_hot_encode(df)
    
    # 6. 特征工程
    print("\n" + "="*70)
    print("步骤 6: 特征工程 / Step 6: Feature Engineering")
    print("="*70)
    df = feature_engineering(df)
    
    # 7. 标准化
    print("\n" + "="*70)
    print("步骤 7: 标准化 / Step 7: Standardization")
    print("="*70)
    df = scale_numeric(df)
    
    # 8. 保存
    print("\n" + "="*70)
    print("步骤 8: 保存 / Step 8: Save")
    print("="*70)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    
    print(f"\n✓ 处理完成！/ Processing Complete!")
    print(f"输出文件 / Output: {OUTPUT_PATH}")
    print(f"最终形状 / Final shape: {df.shape}")
    
    # 9. 统计摘要
    print("\n" + "="*70)
    print("最终统计 / Final Statistics")
    print("="*70)
    
    if 'aki_binary' in df.columns:
        aki_count = df['aki_binary'].sum()
        print(f"\nAKI病例 / AKI cases: {aki_count} ({aki_count/len(df)*100:.2f}%)")
    
    if 'aki_stage' in df.columns:
        print(f"\nAKI分级 / AKI stages:")
        for stage, count in df['aki_stage'].value_counts().sort_index().items():
            print(f"  Stage {int(stage)}: {count} ({count/len(df)*100:.2f}%)")
    
    print("\n" + "="*70)
    print("所有步骤完成！ / All steps completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()