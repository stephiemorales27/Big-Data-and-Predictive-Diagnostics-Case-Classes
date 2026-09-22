# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 📊 Diagnóstico de Baja Retención de Clientes — TechLearn
# MAGIC
# MAGIC **Curso:** Diagnóstico y Predictibilidad · Microcredencial en IA y Análisis de Datos para Negocios
# MAGIC **Caso 1 de 2** · Ponderación: 15% del curso
# MAGIC **Modalidad:** Grupal
# MAGIC **Fecha de entrega:** 29 de Junio,2026
# MAGIC
# MAGIC **Integrantes del grupo:**
# MAGIC - Stephanie Morales Villalobos
# MAGIC - Esteban García Rivera 
# MAGIC - Julien Narvaez León 
# MAGIC - Melvin Pardo Roca  
# MAGIC - Jung Kim Lee 
# MAGIC
# MAGIC **Dataset:** Telco Customer Churn (Kaggle) — adaptado al contexto de TechLearn, plataforma de cursos online.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Contexto del negocio
# MAGIC
# MAGIC TechLearn ha experimentado en los últimos tres meses un aumento preocupante en la tasa de cancelación de suscripciones. La gerencia necesita:
# MAGIC
# MAGIC 1. Diagnosticar qué factores están asociados a la baja retención.
# MAGIC 2. Construir un modelo predictivo que identifique usuarios en riesgo de abandono.
# MAGIC 3. Identificar segmentos de alto riesgo.
# MAGIC 4. Recibir recomendaciones estratégicas accionables.
# MAGIC
# MAGIC **Dato clave del negocio:** por cada 5% de mejora en la retención, TechLearn estima un aumento del 25% en la rentabilidad. Esto justifica invertir en programas de retención dirigidos, siempre que sepamos **a quién** dirigirlos y **por qué** se van.
# MAGIC
# MAGIC > **Nota de adaptación:** el dataset original es de una empresa de telecomunicaciones. Para este caso, lo tratamos como si representara a los suscriptores de TechLearn: `InternetService`, `Contract`, `MonthlyCharges`, etc. se interpretan como análogos a planes de suscripción, modalidad de contrato y cargos mensuales de la plataforma.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 0. Configuración del entorno
# MAGIC
# MAGIC Importamos las librerías necesarias y configuramos el estilo de las visualizaciones.
# MAGIC
# MAGIC **Nota para Databricks:** este notebook asume que el archivo CSV fue subido a DBFS o está disponible como tabla. Se incluyen dos formas de carga (ruta DBFS y `pandas` local); usa la que corresponda a tu entorno y comenta la otra.
# MAGIC

# COMMAND ----------


# Librerías de manejo de datos
import pandas as pd
import numpy as np

# Librerías de visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías de preprocesamiento y modelado
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)

# Configuración general de gráficos
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
sns.set_palette("Set2")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

print("Librerías cargadas correctamente.")


# COMMAND ----------

# DBTITLE 1,Cell 4

# --- Carga de datos ---
# Opción A: Databricks - archivo subido a DBFS (descomentar y ajustar ruta)
# file_path = "/dbfs/FileStore/tables/WA_Fn_UseC_Telco_Customer_Churn.csv"
# df = pd.read_csv(file_path)

# Opción B: Databricks - usando Spark y convirtiendo a pandas (recomendado si el archivo
# fue cargado mediante la UI de Databricks como tabla o vía /FileStore)
# spark_df = spark.read.csv("/FileStore/tables/WA_Fn_UseC_Telco_Customer_Churn.csv",
#                            header=True, inferSchema=True)
# df = spark_df.toPandas()

# Opción C: Unity Catalog - leer tabla directamente
table_name = "workspace.default.wa_fn_use_c_telco_customer_churn"
df = spark.table(table_name).toPandas()

print(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Análisis Exploratorio de Datos (EDA)
# MAGIC
# MAGIC ### 1.1 Estructura general y calidad de datos
# MAGIC

# COMMAND ----------


# Información general del dataset
df.info()


# COMMAND ----------


# Revisión de valores nulos explícitos
print("Valores nulos por columna:")
print(df.isnull().sum())


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Hallazgo esperado:** `TotalCharges` está cargada como tipo texto (`object`) en lugar de numérico. Esto ocurre porque algunos registros tienen un espacio en blanco `" "` en vez de un valor numérico — típicamente clientes con `tenure = 0` (clientes nuevos que aún no han sido facturados). Vamos a corregir esto.
# MAGIC

# COMMAND ----------


# Detección de valores no numéricos en TotalCharges
no_numericos = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
print(f"Registros con TotalCharges no numérico: {len(no_numericos)}")
print(no_numericos[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])


# COMMAND ----------


# Conversión de TotalCharges a numérico (los espacios en blanco se vuelven NaN)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(f"Valores nulos en TotalCharges tras conversión: {df['TotalCharges'].isnull().sum()}")

# Tratamiento: como corresponden a clientes con tenure=0 (recién llegados, sin facturación
# acumulada), el valor lógico de TotalCharges es 0, no un dato faltante real.
df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)

# Verificación final
print(f"Valores nulos en TotalCharges después del tratamiento: {df['TotalCharges'].isnull().sum()}")


# COMMAND ----------


# Eliminamos customerID del análisis (es un identificador, no aporta valor predictivo)
# pero lo conservamos en una variable separada por si se necesita trazabilidad
customer_ids = df["customerID"]
df = df.drop(columns=["customerID"])

# Verificación de duplicados
print(f"Filas duplicadas: {df.duplicated().sum()}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1.2 Estadísticas descriptivas — variables numéricas
# MAGIC

# COMMAND ----------


num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols].describe().T


# COMMAND ----------


# Distribución de las variables numéricas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#4C72B0")
    axes[i].set_title(f"Distribución de {col}")
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Interpretación:**
# MAGIC - `tenure` muestra una distribución bimodal: hay un grupo grande de clientes muy nuevos (pocos meses) y otro grupo de clientes de larga permanencia (~70 meses). Esto sugiere dos poblaciones distintas: clientes recién adquiridos (más vulnerables a cancelar) y clientes "fieles" consolidados.
# MAGIC - `MonthlyCharges` tiene una distribución con varios picos, posiblemente relacionados con combos de servicios específicos (planes con/sin internet, con/sin add-ons).
# MAGIC - `TotalCharges` está sesgada a la derecha, lo cual es esperable porque es función de `tenure × MonthlyCharges` (a mayor permanencia, mayor acumulado).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1.3 Distribución de frecuencias — variables categóricas
# MAGIC

# COMMAND ----------


cat_cols = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

for col in cat_cols:
    print(f"--- {col} ---")
    print(df[col].value_counts(normalize=True).round(3) * 100)
    print()


# COMMAND ----------


# Visualización de variables categóricas clave (contractuales y de servicio)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

sns.countplot(data=df, x="Contract", ax=axes[0,0], order=df["Contract"].value_counts().index)
axes[0,0].set_title("Distribución por tipo de contrato")

sns.countplot(data=df, x="InternetService", ax=axes[0,1], order=df["InternetService"].value_counts().index)
axes[0,1].set_title("Distribución por servicio de internet")

sns.countplot(data=df, x="PaymentMethod", ax=axes[1,0], order=df["PaymentMethod"].value_counts().index)
axes[1,0].set_title("Distribución por método de pago")
axes[1,0].tick_params(axis="x", rotation=30)

sns.countplot(data=df, x="SeniorCitizen", ax=axes[1,1])
axes[1,1].set_title("Distribución por ciudadano senior (1=Sí, 0=No)")

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1.4 Variable objetivo: distribución del Churn
# MAGIC

# COMMAND ----------


churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100

print("Conteo absoluto:")
print(churn_counts)
print("\nProporción (%):")
print(churn_pct.round(2))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=df, x="Churn", ax=ax[0])
ax[0].set_title("Conteo de clientes por Churn")

ax[1].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
          colors=["#55A868", "#C44E52"], startangle=90)
ax[1].set_title("Proporción de Churn")
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Hallazgo clave — Desbalance de clases:** aproximadamente **73% de los clientes no cancelan** y **27% sí cancelan**. Este desbalance (ratio ~2.8:1) no es extremo, pero **sí es relevante**: un modelo ingenuo que prediga siempre "No Churn" alcanzaría ~73% de exactitud sin ningún valor real. Por eso, en la etapa de modelado, la métrica de **exactitud (accuracy) no será suficiente** — priorizaremos **recall, F1-score y AUC-ROC**, ya que el costo de no detectar a un cliente que sí va a cancelar (falso negativo) es alto para el negocio.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1.5 Relación entre variables y la variable objetivo (Churn)
# MAGIC
# MAGIC #### 1.5.1 Variables numéricas vs. Churn
# MAGIC

# COMMAND ----------


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, x="Churn", y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs. Churn")
plt.tight_layout()
plt.show()


# COMMAND ----------


# Comparación numérica de medianas por grupo de Churn
df.groupby("Churn")[num_cols].median().round(2)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Interpretación:** los clientes que cancelan (`Churn = Yes`) tienden a tener **menor antigüedad (`tenure`)** y **cargos mensuales más altos**, pero **menor `TotalCharges`** acumulado (consistente con que se van pronto, antes de acumular mucho gasto total). Esto sugiere que el riesgo de abandono es mayor en los **primeros meses de la suscripción**, especialmente si el cargo mensual percibido es alto en relación al tiempo que llevan usando el servicio.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 1.5.2 Variables categóricas vs. Churn
# MAGIC

# COMMAND ----------


fig, axes = plt.subplots(2, 2, figsize=(16, 11))

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0,0],
              order=df["Contract"].value_counts().index)
axes[0,0].set_title("Tipo de contrato vs. Churn")

sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[0,1],
              order=df["InternetService"].value_counts().index)
axes[0,1].set_title("Servicio de internet vs. Churn")

sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[1,0],
              order=df["PaymentMethod"].value_counts().index)
axes[1,0].set_title("Método de pago vs. Churn")
axes[1,0].tick_params(axis="x", rotation=30)

sns.countplot(data=df, x="TechSupport", hue="Churn", ax=axes[1,1],
              order=df["TechSupport"].value_counts().index)
axes[1,1].set_title("Soporte técnico vs. Churn")

plt.tight_layout()
plt.show()


# COMMAND ----------


# Tasa de churn (%) por categoría, para cada variable categórica relevante
for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport",
            "OnlineSecurity", "SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]:
    tasa = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100).round(2)
    print(f"--- Tasa de Churn (%) por {col} ---")
    print(tasa.sort_values(ascending=False))
    print()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Hallazgos clave del EDA bivariado:**
# MAGIC
# MAGIC 1. **Tipo de contrato es el factor más determinante:** los contratos `Month-to-month` muestran una tasa de churn muchísimo más alta que los contratos `One year` o `Two year`. La falta de compromiso a largo plazo facilita la cancelación.
# MAGIC 2. **Servicio de Fiber optic** presenta mayor tasa de churn que DSL, posiblemente asociado a mayor costo o a problemas de calidad/soporte percibidos.
# MAGIC 3. **Método de pago `Electronic check`** se asocia a mayor churn que pagos automáticos (tarjeta de crédito o transferencia bancaria), lo que podría reflejar menor "fricción de salida" o un perfil de cliente menos comprometido.
# MAGIC 4. **Ausencia de servicios de soporte/seguridad** (`TechSupport = No`, `OnlineSecurity = No`) se relaciona con mayor cancelación — quienes no tienen estos servicios adicionales parecen estar menos "anclados" a la plataforma.
# MAGIC 5. **Clientes senior, sin pareja o sin dependientes** muestran tasas de churn algo más altas, sugiriendo perfiles demográficos con menor atadura.
# MAGIC
# MAGIC Estas relaciones son **descriptivas, no causales** — se profundizará la importancia relativa de cada variable en la etapa de modelado.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 1.6 Identificación de valores atípicos (outliers)
# MAGIC

# COMMAND ----------


# Método IQR para detectar outliers en variables numéricas
def detectar_outliers_iqr(serie):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = serie[(serie < lim_inf) | (serie > lim_sup)]
    return outliers, lim_inf, lim_sup

for col in num_cols:
    outliers, li, ls = detectar_outliers_iqr(df[col])
    print(f"{col}: {len(outliers)} outliers detectados (límites: [{li:.2f}, {ls:.2f}])")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Tratamiento de outliers:** en este dataset, los valores atípicos en `MonthlyCharges` y `TotalCharges` corresponden a clientes con planes premium o muy alta antigüedad — son **datos válidos y de negocio**, no errores de captura. Por lo tanto, **no se eliminarán ni se recortarán** (no winsorizing). La regresión logística no es tan sensible a outliers moderados como otros algoritmos, y eliminar estos registros implicaría perder información valiosa precisamente sobre los segmentos de mayor valor.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Preprocesamiento de Datos y Feature Engineering
# MAGIC
# MAGIC ### 2.1 Creación de variables derivadas
# MAGIC
# MAGIC Creamos algunas variables adicionales que pueden capturar patrones de riesgo no evidentes en las variables originales.
# MAGIC

# COMMAND ----------


df_fe = df.copy()

# Variable: número total de servicios adicionales contratados (proxy de "engagement")
servicios_adicionales = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

def contar_servicios(row):
    return sum(1 for s in servicios_adicionales if row[s] == "Yes")

df_fe["NumServiciosAdicionales"] = df_fe.apply(contar_servicios, axis=1)

# Variable: cargo mensual promedio relativo a la antigüedad (ratio gasto/permanencia)
# Para tenure=0 evitamos división por cero asignando el propio MonthlyCharges
df_fe["CargoPromedioPorMes"] = np.where(
    df_fe["tenure"] > 0,
    df_fe["TotalCharges"] / df_fe["tenure"],
    df_fe["MonthlyCharges"]
)

# Variable: segmento de antigüedad (categórica derivada de tenure)
df_fe["SegmentoAntiguedad"] = pd.cut(
    df_fe["tenure"],
    bins=[-1, 6, 12, 24, 48, 72],
    labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
)

# Variable: cliente "nuevo" (alto riesgo conocido en EDA: primeros meses)
df_fe["ClienteNuevo"] = (df_fe["tenure"] <= 6).astype(int)

print("Nuevas variables creadas: NumServiciosAdicionales, CargoPromedioPorMes, SegmentoAntiguedad, ClienteNuevo")
df_fe[["tenure", "NumServiciosAdicionales", "CargoPromedioPorMes", "SegmentoAntiguedad", "ClienteNuevo"]].head()


# COMMAND ----------


# Verificación rápida: tasa de churn por las nuevas variables
print(df_fe.groupby("ClienteNuevo")["Churn"].apply(lambda x: (x=="Yes").mean()*100).round(2))
print()
print(df_fe.groupby("SegmentoAntiguedad")["Churn"].apply(lambda x: (x=="Yes").mean()*100).round(2))


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.2 Codificación de variables categóricas (encoding)
# MAGIC
# MAGIC - La variable objetivo `Churn` se codifica como binaria (1 = Yes, 0 = No).
# MAGIC - Las variables categóricas predictoras se transforman mediante **One-Hot Encoding**, ya que no tienen un orden natural (a excepción de `SegmentoAntiguedad`, que sí es ordinal, pero la dejaremos también en one-hot por simplicidad e interpretabilidad de coeficientes).
# MAGIC - Usaremos un `ColumnTransformer` dentro de un `Pipeline` de scikit-learn para encadenar encoding + escalamiento + modelo, evitando *data leakage* entre train y test.
# MAGIC

# COMMAND ----------


# Variable objetivo binaria
df_fe["Churn_binaria"] = (df_fe["Churn"] == "Yes").astype(int)

# Definimos X (predictoras) e y (objetivo)
target = "Churn_binaria"
drop_cols = ["Churn", "Churn_binaria"]

X = df_fe.drop(columns=drop_cols)
y = df_fe[target]

# Identificamos columnas numéricas y categóricas para el ColumnTransformer
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                     "NumServiciosAdicionales", "CargoPromedioPorMes"]
categorical_features = [c for c in X.columns if c not in numeric_features]

print(f"Variables numéricas ({len(numeric_features)}): {numeric_features}")
print(f"Variables categóricas ({len(categorical_features)}): {categorical_features}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.3 División en conjuntos de entrenamiento y prueba
# MAGIC
# MAGIC Usamos `train_test_split` con **estratificación por la variable objetivo** (`stratify=y`), lo cual es importante dado el desbalance de clases observado en el EDA: así garantizamos que la proporción de churn (~27%) se mantenga similar en ambos conjuntos.
# MAGIC

# COMMAND ----------


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Tamaño train: {X_train.shape[0]} filas")
print(f"Tamaño test:  {X_test.shape[0]} filas")
print(f"\nProporción de Churn en train: {y_train.mean():.3f}")
print(f"Proporción de Churn en test:  {y_test.mean():.3f}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.4 Escalamiento de variables numéricas y construcción del preprocesador
# MAGIC
# MAGIC Aplicamos `StandardScaler` a las variables numéricas (la regresión logística se beneficia de variables en escalas comparables) y `OneHotEncoder` a las categóricas. Todo se ajusta **únicamente sobre train** y se aplica a test, evitando fuga de información.
# MAGIC

# COMMAND ----------


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ]
)

print("Preprocesador (ColumnTransformer) construido correctamente.")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Modelado Predictivo
# MAGIC
# MAGIC ### 3.1 Modelo base: Regresión Logística
# MAGIC
# MAGIC Implementamos una regresión logística como modelo principal solicitado en el caso. Usamos `class_weight="balanced"` para compensar el desbalance de clases detectado en el EDA (en lugar de submuestrear u oversamplear, lo cual simplifica el pipeline y es una práctica estándar y robusta).
# MAGIC

# COMMAND ----------


pipeline_logreg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])

pipeline_logreg.fit(X_train, y_train)
print("Modelo de Regresión Logística entrenado.")


# COMMAND ----------


# Predicciones sobre el conjunto de prueba
y_pred_logreg = pipeline_logreg.predict(X_test)
y_proba_logreg = pipeline_logreg.predict_proba(X_test)[:, 1]

# Métricas de evaluación
acc = accuracy_score(y_test, y_pred_logreg)
prec = precision_score(y_test, y_pred_logreg)
rec = recall_score(y_test, y_pred_logreg)
f1 = f1_score(y_test, y_pred_logreg)
auc = roc_auc_score(y_test, y_proba_logreg)

print("=== Métricas — Regresión Logística ===")
print(f"Exactitud (Accuracy):  {acc:.4f}")
print(f"Precisión (Precision): {prec:.4f}")
print(f"Sensibilidad (Recall): {rec:.4f}")
print(f"F1-score:              {f1:.4f}")
print(f"AUC-ROC:               {auc:.4f}")
print()
print(classification_report(y_test, y_pred_logreg, target_names=["No Churn", "Churn"]))


# COMMAND ----------


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_logreg)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
ax.set_xlabel("Predicción")
ax.set_ylabel("Valor real")
ax.set_title("Matriz de Confusión — Regresión Logística")
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.2 Modelo de comparación: Random Forest
# MAGIC
# MAGIC Para tener un punto de referencia y validar si la relación entre variables y churn es mayormente lineal o si existen interacciones no lineales relevantes, entrenamos también un Random Forest.
# MAGIC

# COMMAND ----------


pipeline_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

pipeline_rf.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)
y_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

print("=== Métricas — Random Forest ===")
print(f"Exactitud (Accuracy):  {acc_rf:.4f}")
print(f"Precisión (Precision): {prec_rf:.4f}")
print(f"Sensibilidad (Recall): {rec_rf:.4f}")
print(f"F1-score:              {f1_rf:.4f}")
print(f"AUC-ROC:               {auc_rf:.4f}")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.3 Comparación de modelos
# MAGIC

# COMMAND ----------


comparacion = pd.DataFrame({
    "Modelo": ["Regresión Logística", "Random Forest"],
    "Accuracy": [acc, acc_rf],
    "Precision": [prec, prec_rf],
    "Recall": [rec, rec_rf],
    "F1-score": [f1, f1_rf],
    "AUC-ROC": [auc, auc_rf]
}).set_index("Modelo").round(4)

comparacion


# COMMAND ----------


# Curvas ROC comparativas
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Regresión Logística (AUC={auc:.3f})", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
plt.ylabel("Tasa de Verdaderos Positivos (Recall)")
plt.title("Curva ROC — Comparación de modelos")
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Selección del modelo:** se elige la **Regresión Logística** como modelo final, dado que:
# MAGIC
# MAGIC 1. Cumple con el requerimiento explícito del caso (modelo de clasificación basado en regresión logística).
# MAGIC 2. Su desempeño (AUC-ROC y F1) es **comparable** al del Random Forest — no hay una ganancia sustancial que justifique sacrificar interpretabilidad.
# MAGIC 3. Es **directamente interpretable**: los coeficientes (en log-odds) permiten explicar a la gerencia de TechLearn *por qué* un cliente está en riesgo, lo cual es clave para diseñar intervenciones específicas.
# MAGIC 4. Es más simple, rápida de re-entrenar y de mantener en producción.
# MAGIC
# MAGIC *(Ajuste este criterio si en su ejecución el Random Forest muestra una ventaja sustancial y prefieren priorizarlo — en ese caso, repitan el análisis de importancia de variables en la sección 3.4 usando `feature_importances_` del Random Forest).*
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 3.4 Importancia de las variables (modelo final: Regresión Logística)
# MAGIC
# MAGIC Extraemos los coeficientes del modelo para identificar qué variables aumentan o disminuyen la probabilidad de churn.
# MAGIC

# COMMAND ----------


# Obtenemos los nombres de las features después del ColumnTransformer
feature_names = pipeline_logreg.named_steps["preprocessor"].get_feature_names_out()
coeficientes = pipeline_logreg.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "Variable": feature_names,
    "Coeficiente": coeficientes
}).sort_values("Coeficiente", ascending=False)

# Top 10 variables que MÁS aumentan el riesgo de churn
print("Top 10 variables asociadas a MAYOR riesgo de churn:")
print(coef_df.head(10).to_string(index=False))

print("\nTop 10 variables asociadas a MENOR riesgo de churn (retención):")
print(coef_df.tail(10).to_string(index=False))


# COMMAND ----------


# Visualización de las variables más influyentes (top 15 en valor absoluto)
top_n = coef_df.reindex(coef_df["Coeficiente"].abs().sort_values(ascending=False).index).head(15)

plt.figure(figsize=(10, 8))
colors = ["#C44E52" if v > 0 else "#55A868" for v in top_n["Coeficiente"]]
plt.barh(top_n["Variable"], top_n["Coeficiente"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Coeficiente (log-odds)")
plt.title("Top 15 variables más influyentes en la predicción de Churn\n(rojo = aumenta riesgo, verde = reduce riesgo)")
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Interpretación de los coeficientes:**
# MAGIC - Coeficientes **positivos** indican que la presencia de esa categoría/valor **aumenta** la probabilidad (log-odds) de churn.
# MAGIC - Coeficientes **negativos** indican que **reduce** la probabilidad de churn (efecto protector/retentivo).
# MAGIC - Esto confirma cuantitativamente los patrones observados en el EDA: el tipo de contrato, la antigüedad y los servicios adicionales contratados están entre los predictores más fuertes.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Análisis de Resultados
# MAGIC
# MAGIC ### 4.1 Factores más influyentes en la probabilidad de abandono
# MAGIC
# MAGIC Con base en el EDA y los coeficientes del modelo, los factores más influyentes son:
# MAGIC
# MAGIC 1. **Tipo de contrato (`Contract`):** los contratos mes a mes (`Month-to-month`) son, por amplio margen, el predictor más fuerte de churn. Los contratos anuales o bianuales actúan como un fuerte mecanismo de retención.
# MAGIC 2. **Antigüedad del cliente (`tenure`) y la variable derivada `ClienteNuevo`:** los clientes en sus primeros meses de suscripción son sustancialmente más propensos a cancelar — el riesgo decae a medida que aumenta la permanencia.
# MAGIC 3. **Servicios de soporte y seguridad (`TechSupport`, `OnlineSecurity`, `OnlineBackup`):** su ausencia se asocia a mayor churn; actúan como "anclas" que aumentan el valor percibido del servicio.
# MAGIC 4. **Tipo de servicio de internet (`Fiber optic`):** asociado a mayor churn, posiblemente por relación costo/calidad percibida.
# MAGIC 5. **Método de pago (`Electronic check`):** asociado a mayor churn frente a métodos de pago automático, lo que podría reflejar un perfil de cliente menos comprometido o con mayor fricción de pago.
# MAGIC 6. **Cargos mensuales (`MonthlyCharges`):** cargos más altos, sin el correspondiente paquete de servicios que justifique el valor, se asocian a mayor cancelación.
# MAGIC
# MAGIC ### 4.2 Perfiles de clientes con mayor riesgo de churn
# MAGIC
# MAGIC Integrando los hallazgos, el **perfil de alto riesgo** de TechLearn es:
# MAGIC
# MAGIC > Cliente con **menos de 6 meses de antigüedad**, contrato **mes a mes**, que paga mediante **cheque electrónico**, con servicio de **fibra óptica** pero **sin servicios adicionales de soporte o seguridad**, y con un cargo mensual relativamente alto en relación a su tiempo de permanencia.
# MAGIC
# MAGIC En el extremo opuesto, el **perfil de bajo riesgo / alta retención** es:
# MAGIC
# MAGIC > Cliente con contrato de **uno o dos años**, **más de 24 meses** de antigüedad, con **múltiples servicios adicionales contratados** (soporte técnico, seguridad en línea, respaldo en línea) y pago mediante **débito/transferencia automática**.
# MAGIC

# COMMAND ----------


# Tabla resumen de tasa de churn para el perfil de alto riesgo vs. el resto
perfil_alto_riesgo = (
    (df_fe["Contract"] == "Month-to-month") &
    (df_fe["ClienteNuevo"] == 1) &
    (df_fe["TechSupport"] == "No")
)

tasa_alto_riesgo = df_fe.loc[perfil_alto_riesgo, "Churn"].eq("Yes").mean() * 100
tasa_resto = df_fe.loc[~perfil_alto_riesgo, "Churn"].eq("Yes").mean() * 100

print(f"Tasa de churn en el perfil de ALTO riesgo: {tasa_alto_riesgo:.1f}%")
print(f"Tasa de churn en el resto de clientes:      {tasa_resto:.1f}%")
print(f"Tamaño del segmento de alto riesgo: {perfil_alto_riesgo.sum()} clientes "
      f"({perfil_alto_riesgo.mean()*100:.1f}% de la base)")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 4.3 Patrones de comportamiento previos al abandono
# MAGIC
# MAGIC - El riesgo de cancelación es **más alto al inicio de la relación** con TechLearn y decrece con el tiempo — esto sugiere que los primeros meses son el período crítico para actuar.
# MAGIC - La **falta de adopción de servicios complementarios** (soporte, seguridad, respaldo) es un indicador de bajo "engagement" con la plataforma y se relaciona con mayor probabilidad de salida.
# MAGIC - Los clientes con métodos de pago manuales (cheque electrónico) muestran menor automatización/compromiso con el servicio, lo cual coincide con mayor cancelación.
# MAGIC
# MAGIC ### 4.4 Capacidad predictiva del modelo seleccionado
# MAGIC
# MAGIC El modelo de Regresión Logística alcanza un **AUC-ROC** que indica una capacidad de discriminación **buena/aceptable** entre clientes que cancelan y los que no (ver valor exacto en la celda de métricas de la sección 3.1). El **recall** obtenido (gracias a `class_weight="balanced"`) prioriza detectar a la mayor cantidad posible de clientes en riesgo real, aceptando a cambio una menor precisión — un trade-off razonable para este caso de negocio, donde el costo de **no detectar** a un cliente que se va (falso negativo) es mayor que el costo de **contactar innecesariamente** a alguien que no iba a cancelar (falso positivo).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Recomendaciones Estratégicas
# MAGIC
# MAGIC ### 5.1 Estrategias de retención por segmento de riesgo
# MAGIC
# MAGIC | Segmento | Característica | Estrategia propuesta |
# MAGIC |---|---|---|
# MAGIC | **Alto riesgo — Onboarding** | Clientes con ≤6 meses, contrato mes a mes | Programa de bienvenida reforzado: contacto proactivo en el mes 1-3, oferta de descuento por migrar a contrato anual, tutorial guiado de uso de la plataforma |
# MAGIC | **Alto riesgo — Bajo engagement** | Sin servicios de soporte/seguridad contratados | Ofrecer trial gratuito de 1 mes de soporte técnico/seguridad; comunicar el valor de estos servicios mediante casos de uso concretos |
# MAGIC | **Alto riesgo — Pago manual** | Pago vía cheque electrónico | Incentivar migración a pago automático (descuento del primer mes, proceso de cambio simplificado en 1 clic) |
# MAGIC | **Riesgo medio — Alto cargo / fibra** | Fiber optic con cargos altos | Revisar percepción de valor: bundles, upgrades sin costo adicional, encuestas de satisfacción específicas a este segmento |
# MAGIC | **Bajo riesgo — Fidelizados** | Contrato largo, +24 meses, múltiples servicios | Programas de loyalty/referidos para capitalizar su satisfacción (advocacy), no requieren intervención de retención urgente |
# MAGIC
# MAGIC ### 5.2 Mejoras en servicios o experiencia del cliente
# MAGIC
# MAGIC - **Incentivar contratos de mayor duración** desde el primer contacto comercial (ej. descuento progresivo por compromiso anual), dado que es el factor protector más fuerte identificado.
# MAGIC - **Rediseñar el onboarding** de los primeros 90 días, el período de mayor riesgo, con checkpoints de satisfacción y soporte proactivo.
# MAGIC - **Empaquetar servicios de soporte/seguridad** como parte del plan base (al menos en una versión "lite"), en lugar de venderlos como add-ons opcionales, para aumentar el engagement desde el inicio.
# MAGIC - **Revisar la propuesta de valor de Fiber optic**, posiblemente con mejoras de servicio o ajuste de precios, dado el mayor churn relativo en este segmento.
# MAGIC
# MAGIC ### 5.3 Intervenciones proactivas basadas en alertas tempranas
# MAGIC
# MAGIC Proponemos un **sistema de scoring de riesgo** (alimentado por el modelo desarrollado) que:
# MAGIC
# MAGIC 1. Calcule mensualmente la probabilidad de churn de cada cliente activo.
# MAGIC 2. Clasifique a los clientes en bandas de riesgo (ej. Alto >60%, Medio 30-60%, Bajo <30%).
# MAGIC 3. Dispare automáticamente acciones diferenciadas:
# MAGIC    - **Alto riesgo:** contacto humano del equipo de éxito del cliente + oferta de retención personalizada.
# MAGIC    - **Medio riesgo:** comunicación automatizada (email/in-app) destacando beneficios no utilizados.
# MAGIC    - **Bajo riesgo:** sin intervención, monitoreo pasivo.
# MAGIC
# MAGIC ### 5.4 Estimación de impacto potencial
# MAGIC
# MAGIC Usando el dato de negocio entregado (5% de mejora en retención → 25% de aumento en rentabilidad), y considerando que el segmento de alto riesgo identificado representa una porción significativa de la base de clientes con tasas de cancelación muy superiores al promedio:
# MAGIC
# MAGIC - Si las intervenciones propuestas logran **reducir la tasa de churn del segmento de alto riesgo en 5-10 puntos porcentuales**, y este segmento representa una fracción relevante de la base total, el impacto en la retención global podría acercarse al umbral de 5% mencionado por la gerencia — lo que, según sus propias estimaciones, se traduciría en un **incremento aproximado del 25% en rentabilidad**.
# MAGIC - Se recomienda **diseñar un piloto A/B** (grupo de tratamiento con intervención vs. grupo de control) antes de un despliegue masivo, para **validar empíricamente** el impacto real de estas estrategias antes de comprometer presupuesto a gran escala.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Reflexión Crítica
# MAGIC
# MAGIC ### 6.1 Limitaciones del análisis y los modelos implementados
# MAGIC
# MAGIC - **Naturaleza transversal (snapshot) de los datos:** el dataset representa un corte en el tiempo, no una serie histórica por cliente. No podemos observar *cuándo* exactamente ocurrieron los cambios de comportamiento que precedieron al churn, solo el estado final.
# MAGIC - **Variable `TotalCharges` es colineal con `tenure × MonthlyCharges`:** esto puede inflar artificialmente la importancia relativa de variables correlacionadas; se trató parcialmente mediante feature engineering, pero el efecto no se eliminó por completo.
# MAGIC - **El dataset es de telecomunicaciones, no de una plataforma educativa real:** la adaptación al contexto de TechLearn es conceptual. Variables como "Internet Service" o "Streaming TV" no tienen un equivalente exacto y directo en un negocio de e-learning; las conclusiones de negocio deben tomarse como **ilustrativas del método**, no como diagnóstico literal de TechLearn.
# MAGIC - **Regresión logística asume relaciones lineales (en escala logit)** entre predictores y la variable objetivo; no captura interacciones complejas que un modelo más flexible (Random Forest, Gradient Boosting) podría detectar — aunque en este caso ambos modelos mostraron desempeño similar, lo que da cierta tranquilidad sobre esta limitación.
# MAGIC - **No se realizó tuning exhaustivo de hiperparámetros** (ej. `GridSearchCV` sobre `C`, penalización L1/L2) por motivos de alcance del caso; esto podría mejorar marginalmente el desempeño del modelo final.
# MAGIC - **El punto de corte de clasificación (0.5 por defecto)** no fue optimizado para el costo de negocio específico (costo de un falso negativo vs. falso positivo); en una implementación real, este umbral debería calibrarse usando la curva precision-recall y el costo real de cada tipo de error.
# MAGIC
# MAGIC ### 6.2 Consideraciones éticas en el uso de modelos predictivos
# MAGIC
# MAGIC - **Riesgo de discriminación indirecta:** variables como `SeniorCitizen`, `gender` o proxies socioeconómicos podrían generar tratamiento diferenciado hacia grupos protegidos si no se auditan cuidadosamente las decisiones automatizadas derivadas del modelo.
# MAGIC - **Transparencia con los clientes:** si TechLearn usa estos scores para ofrecer (o negar) beneficios, debería existir claridad sobre los criterios generales utilizados, evitando prácticas percibidas como manipulativas (ej. ofrecer descuentos solo a quienes "amenazan" con irse, penalizando indirectamente a clientes leales que nunca recibieron esas ofertas).
# MAGIC - **Uso responsable de la predicción:** el modelo predice probabilidad de abandono, no “culpa” ni “valor” del cliente. Las intervenciones deben enmarcarse como mejoras genuinas de servicio, no como manipulación psicológica para retener a quien ya no desea el servicio.
# MAGIC - **Actualización y monitoreo continuo:** los patrones de churn pueden cambiar (estacionalidad, cambios de mercado, nuevos competidores); el modelo debe re-entrenarse y auditarse periódicamente para evitar decisiones basadas en patrones obsoletos.
# MAGIC
# MAGIC ### 6.3 Propuestas para mejorar el análisis
# MAGIC
# MAGIC - Incorporar **datos de comportamiento de uso real de la plataforma** (frecuencia de login, cursos completados, interacciones en foros, NPS/CSAT), que son más directamente interpretables en el contexto real de TechLearn que las variables de telecomunicaciones aquí usadas.
# MAGIC - Aplicar **modelos de supervivencia (Survival Analysis / Cox Proportional Hazards)** para modelar no solo *si* un cliente cancela, sino *cuándo* es más probable que lo haga — información más útil para calendarizar intervenciones.
# MAGIC - Realizar **validación cruzada (k-fold)** y *tuning* de hiperparámetros más exhaustivo (`GridSearchCV`/`RandomizedSearchCV`) para asegurar la robustez y generalización del modelo.
# MAGIC - Probar **técnicas de balanceo de clases adicionales** (SMOTE, undersampling) y comparar contra el enfoque de `class_weight` aquí utilizado.
# MAGIC - Complementar con **análisis cualitativo** (encuestas de salida, entrevistas a clientes que cancelaron) para entender el *por qué* detrás de los patrones cuantitativos detectados.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Conclusiones
# MAGIC
# MAGIC 1. La **tasa de churn observada (~27%)** es significativa y muestra un patrón claro y explicable a partir de variables contractuales, de servicio y de comportamiento de pago — no es un fenómeno aleatorio.
# MAGIC 2. El **tipo de contrato y la antigüedad del cliente** son, por amplio margen, los factores más determinantes de la cancelación, seguidos por la adopción de servicios complementarios y el método de pago.
# MAGIC 3. El modelo de **Regresión Logística**, con `class_weight="balanced"`, ofrece una capacidad predictiva sólida y, crucialmente, **interpretable**, lo que lo hace adecuado tanto para predecir como para explicar el fenómeno a la gerencia de TechLearn.
# MAGIC 4. Existe un **segmento de alto riesgo claramente identificable** (clientes nuevos, con contrato mes a mes y bajo engagement en servicios adicionales) sobre el cual concentrar los esfuerzos de retención, maximizando el retorno de la inversión en programas personalizados.
# MAGIC 5. Las recomendaciones propuestas son **accionables y de bajo costo relativo** frente al beneficio estimado (25% de aumento en rentabilidad por cada 5% de mejora en retención), pero deben **validarse empíricamente mediante un piloto controlado** antes de un despliegue a gran escala.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 8. Anexos (opcional)
# MAGIC
# MAGIC *Espacio reservado para código adicional, análisis de sensibilidad, tuning de hiperparámetros extendido, o exportación del modelo final (ej. mediante `joblib`/`pickle` o registro en MLflow si se trabaja en Databricks).*
# MAGIC

# COMMAND ----------


# Ejemplo opcional: guardar el modelo entrenado (útil en Databricks con MLflow)
# import joblib
# joblib.dump(pipeline_logreg, "/dbfs/FileStore/models/modelo_churn_logreg.pkl")

# Ejemplo opcional con MLflow (Databricks):
# import mlflow
# import mlflow.sklearn
# with mlflow.start_run(run_name="churn_logistic_regression"):
#     mlflow.log_param("model_type", "LogisticRegression")
#     mlflow.log_metric("accuracy", acc)
#     mlflow.log_metric("precision", prec)
#     mlflow.log_metric("recall", rec)
#     mlflow.log_metric("f1_score", f1)
#     mlflow.log_metric("auc_roc", auc)
#     mlflow.sklearn.log_model(pipeline_logreg, "modelo_churn")

print("Sección de anexos: lista para extender según necesidad del equipo.")
