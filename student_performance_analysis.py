
# -*- coding: utf-8 -*-
# Analisi del rendimento accademico tramite Machine Learning
# Codice ispirato alla tesi di Giuseppe Gaggiano

# =====================================================
# SEZIONE 1 - Importazione delle librerie necessarie
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# SEZIONE 2 - Caricamento e ispezione preliminare del dataset
# =====================================================
df = pd.read_csv("StudentPerformanceFactors.csv")
print("Preview del dataset:")
print(df.head())
print("Dimensioni:", df.shape)

# =====================================================
# SEZIONE 3 - Pulizia dei dati e gestione dei valori mancanti
# =====================================================
df.dropna(inplace=True)

# =====================================================
# SEZIONE 4 - Codifica delle variabili categoriche
# =====================================================
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =====================================================
# SEZIONE 5 - Analisi Esplorativa (EDA): Heatmap delle Correlazioni
# =====================================================
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap delle correlazioni")
plt.tight_layout()
plt.savefig("correlazione_heatmap.png")

# =====================================================
# SEZIONE 6 - Boxplot: Exam_Score rispetto a variabili categoriche
# =====================================================
boxplot_vars = [
    "Gender", "Parental_Education_Level", "School_Type",
    "Parental_Involvement", "Family_Income", "Distance_from_Home"
]
for var in boxplot_vars:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=var, y="Exam_Score", data=df)
    plt.title(f"Boxplot: Exam_Score vs {var}")
    plt.tight_layout()
    plt.savefig(f"boxplot_exam_score_vs_{var}.png")

# =====================================================
# SEZIONE 7 - Bubble Chart: relazione tra variabili numeriche
# =====================================================
# Bubble 1: Hours_Studied vs Exam_Score (dimensione: Tutoring_Sessions)
plt.figure(figsize=(10, 6))
plt.scatter(df["Hours_Studied"], df["Exam_Score"],
            s=df["Tutoring_Sessions"] * 10, alpha=0.5)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Bubble Chart: Hours Studied vs Exam Score (size = Tutoring Sessions)")
plt.tight_layout()
plt.savefig("bubble_hours_vs_score.png")

# Bubble 2: Attendance vs Exam_Score (dimensione: Sleep_Hours)
plt.figure(figsize=(10, 6))
plt.scatter(df["Attendance"], df["Exam_Score"],
            s=df["Sleep_Hours"] * 10, alpha=0.5, c="green")
plt.xlabel("Attendance")
plt.ylabel("Exam Score")
plt.title("Bubble Chart: Attendance vs Exam Score (size = Sleep Hours)")
plt.tight_layout()
plt.savefig("bubble_attendance_vs_score.png")

# =====================================================
# SEZIONE 8 - Separazione delle feature e del target
# =====================================================
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

# =====================================================
# SEZIONE 9 - Normalizzazione delle feature
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# SEZIONE 10 - Divisione del dataset in training e test set
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =====================================================
# SEZIONE 11 - Modello di Regressione Lineare
# =====================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# =====================================================
# SEZIONE 12 - Modello di Random Forest Regressor
# =====================================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# =====================================================
# SEZIONE 13 - Valutazione dei Modelli
# =====================================================
print("\nValutazione Modelli:")
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
