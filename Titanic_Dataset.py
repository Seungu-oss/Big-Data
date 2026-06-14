import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# STEP 01. 데이터 준비 (Kaggle train.csv 로드)
# ==========================================
df = pd.read_csv('train.csv')

print(f"원본 데이터 Shape: {df.shape}")

# ==========================================
# 파생 변수 생성 (STEP 3-4)
# ==========================================
# 1. FamilySize (가족 수)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 2. IsAlone (혼자 탑승했는지 여부)
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 3. Title (이름에서 호칭 추출 - 가산점 포인트)
# 정규표현식을 사용하여 알파벳 뒤에 마침표(.)가 오는 문자열 추출
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 희귀 호칭은 'Rare'로 통합, 기혼/미혼 여성 호칭 통일
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 모델 학습에 방해되는 고유 식별자 및 결측치가 너무 많은 Cabin 제거
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

print(f"전처리 후 데이터 Shape: {df.shape}")
display(df.head())

# ==========================================
# STEP 02. 탐색적 데이터 분석 (EDA) - 시각화
# ==========================================
plt.figure(figsize=(15, 4))

# 1. 타겟 변수 시각화
plt.subplot(1, 3, 1)
sns.countplot(x='Survived', data=df)
plt.title('Target Distribution (Survived)')

# 2. 결측치 비율 시각화
plt.subplot(1, 3, 2)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')

# 3. 새로 만든 Title에 따른 생존율
plt.subplot(1, 3, 3)
sns.barplot(x='Title', y='Survived', data=df, errorbar=None)
plt.title('Survival Rate by Title')
plt.tight_layout()
plt.show()

# ==========================================
# STEP 03 & 04 & 05. ML Pipeline 및 실험 설정
# ==========================================
# 타겟 및 피처 분리
X = df.drop('Survived', axis=1)
y = df['Survived']

# 학습용/평가용 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 수치형 및 범주형 컬럼 명시 (Kaggle 데이터 기준)
num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone']
# Pclass는 숫자 형태지만 의미상 범주형이므로 cat_cols에 포함
cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title']

# 평가 함수
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred)
    }
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_test, y_proba)
    return metrics

# 실험 세팅 (과제 조건 반영)
experiments = {
    # Base: 범주형/결측치 제외 최소 데이터 활용
    'Base': Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[('num', 'passthrough', ['Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone'])],
            remainder='drop'
        )),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),

    # Exp-1: Mean Imputer / One-Hot / Standard Scaler / Feature Select X
    'Exp-1': Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),

    # Exp-2: Median Imputer / Ordinal(Label) / MinMax Scaler / Feature Select O (상위 6개)
    'Exp-2': Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', MinMaxScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat_cols)
        ])),
        ('feature_selection', SelectKBest(score_func=f_classif, k=6)),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),

    # Exp-3: Most Frequent / One-Hot / Robust Scaler / Feature Select O (상위 8개)
    'Exp-3': Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('scaler', RobustScaler())]), num_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
        ])),
        ('feature_selection', SelectKBest(score_func=f_classif, k=8)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
}

# ==========================================
# 모델 학습 및 실험 결과 출력
# ==========================================
results = []

for exp_name, pipeline in experiments.items():
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    metrics['Experiment'] = exp_name
    results.append(metrics)

# 데이터프레임으로 깔끔하게 출력
results_df = pd.DataFrame(results).set_index('Experiment')
display(results_df.round(4))
