# Databricks notebook source
# MAGIC %md
# MAGIC # 🧠 Semana 7 — Modelos de Clasificación para el Análisis Diagnóstico
# MAGIC **Curso:** 92-0030 Diagnóstico y Predictibilidad  
# MAGIC **Profesor:** Robin Sequeira  
# MAGIC **Universidad:** ULACIT — I Cuatrimestre 2026  
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### ¿Qué vamos a hacer hoy?
# MAGIC
# MAGIC Hasta ahora predijimos **números** (regresión).  
# MAGIC Hoy predecimos **categorías**: ¿este tumor es **Maligno** o **Benigno**?
# MAGIC
# MAGIC **Dataset:** Breast Cancer Wisconsin — 569 biopsias, 30 medidas celulares  
# MAGIC **Variable objetivo:** `diagnosis` → M (Maligno) o B (Benigno)
# MAGIC
# MAGIC **Flujo del notebook:**
# MAGIC 1. Cargar y explorar los datos
# MAGIC 2. Seleccionar las variables más relevantes
# MAGIC 3. Entrenar el modelo (Regresión Logística)
# MAGIC 4. Evaluar: matriz de confusión, precisión, recall y F1
# MAGIC 5. Interpretar los resultados en contexto médico
# MAGIC
# MAGIC ---
# MAGIC > ⚠️ **Antes de empezar:** subí el archivo `breast_cancer_semana7.csv` a Databricks desde  
# MAGIC > `Data → Add Data → Upload File` y copiá la ruta que te muestra.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 1 — Importar librerías

# COMMAND ----------

# ── Celda 1: Importar todas las librerías que vamos a usar ──────────────────
# Si alguna da error, avisale al profesor antes de continuar.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

print("✅ Todas las librerías cargaron correctamente.")
print(f"   pandas {pd.__version__}  |  numpy {np.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 2 — Cargar el dataset

# COMMAND ----------

# ── Celda 2: Cargar el CSV desde Databricks ─────────────────────────────────
#
# OPCIÓN A — Si subiste el CSV por la interfaz de Databricks:
#   Cambiá la ruta por la que te mostró Databricks al subir el archivo.
#   Ejemplo: /dbfs/FileStore/tables/breast_cancer_semana7.csv
#
# OPCIÓN B — Si usás Spark (modo Databricks nativo):
#   df = spark.read.csv("/FileStore/tables/breast_cancer_semana7.csv",
#header=True, inferSchema=True).toPandas()

# Leer desde la tabla de Unity Catalog
df = spark.table("workspace.default.breast_cancer_semana_7").toPandas() 

print("✅ Dataset cargado correctamente.")
print(f"   Filas:    {df.shape[0]}")
print(f"   Columnas: {df.shape[1]}")
print()
print("Distribución de la variable objetivo (diagnosis):")
print(df["diagnosis"].value_counts().to_string())
print()
vc = df["diagnosis"].value_counts()
pct_b = vc["B"] / len(df) * 100
pct_m = vc["M"] / len(df) * 100
print(f"   Benignos:  {vc['B']} ({pct_b:.1f}%)")
print(f"   Malignos:  {vc['M']} ({pct_m:.1f}%)")
print()
# INTERPRETACIÓN:
# Hay más tumores benignos que malignos (~63% vs ~37%).
# Este leve desbalance justifica usar F1-score además de accuracy.
print("💬 Hay un leve desbalance de clases.")
print("   Por eso vamos a reportar F1, precisión y recall, no solo accuracy.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 3 — Exploración inicial

# COMMAND ----------

# ── Celda 3: Exploración inicial del dataset ────────────────────────────────
# Antes de modelar siempre miramos los datos: tipos, nulos y estadísticas.

print("═" * 55)
print("PRIMERAS 5 FILAS")
print("═" * 55)
display(df.head())

# COMMAND ----------

# Verificar si hay valores nulos
nulos = df.isnull().sum().sum()
print(f"Valores nulos en el dataset: {nulos}")

if nulos == 0:
    print("✅ El dataset está limpio, sin valores faltantes.")
else:
    print("⚠️ Hay nulos. Revisar antes de continuar.")
    print(df.isnull().sum()[df.isnull().sum() > 0])

print()
print("Tipos de datos (primeras columnas):")
print(df.dtypes.head(8).to_string())

# COMMAND ----------

# Gráfico: distribución de la variable objetivo
fig, ax = plt.subplots(figsize=(5, 3.5))

colores = {"B": "#3B1F5E", "M": "#E8820C"}
vc = df["diagnosis"].value_counts()
vc.plot(kind="bar", color=[colores[k] for k in vc.index],
        ax=ax, edgecolor="white", width=0.5)

ax.set_title("¿Cuántos tumores Malignos vs Benignos?", fontsize=13, fontweight="bold")
ax.set_xlabel("Diagnóstico")
ax.set_ylabel("Cantidad de casos")
ax.set_xticklabels(["Benigno (B)", "Maligno (M)"], rotation=0)

# Agregar etiquetas encima de cada barra
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height() + 3),
                ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

print("\n📌 INTERPRETACIÓN:")
print(f"   El 62.7% de los tumores son benignos y el 37.3% son malignos.")
print("   Un modelo que SIEMPRE prediga 'benigno' tendría 62.7% de accuracy...")
print("   ...pero no detectaría NINGÚN tumor maligno. Por eso F1-score importa más.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 4 — Selección de variables por correlación

# COMMAND ----------

# ── Celda 4: ¿Qué variables tienen más relación con el diagnóstico? ──────────
#
# Para calcular correlaciones necesitamos que diagnosis sea numérico.
# Convertimos M=1 (Maligno) y B=0 (Benigno).
# Luego medimos cuánto se relaciona cada variable con ese 0/1.
# Solo nos quedamos con las que tengan correlación absoluta > 0.2.

df["diagnosis_num"] = (df["diagnosis"] == "M").astype(int)

# Columnas de features (todo menos diagnosis y diagnosis_num)
feat_cols = [c for c in df.columns if c not in ["diagnosis", "diagnosis_num"]]

# Correlación absoluta de cada variable con el diagnóstico
correlaciones = (
    df[feat_cols]
    .corrwith(df["diagnosis_num"])
    .abs()
    .sort_values(ascending=False)
)

print("TOP 10 variables más correlacionadas con el diagnóstico:")
print(correlaciones.head(10).round(3).to_string())

# Seleccionar solo las que superan el umbral
UMBRAL = 0.2
features_seleccionadas = correlaciones[correlaciones > UMBRAL].index.tolist()

print(f"\n✅ Variables seleccionadas (|correlación| > {UMBRAL}): {len(features_seleccionadas)} de {len(feat_cols)}")
print(features_seleccionadas)

# COMMAND ----------

# Gráfico: Top 10 variables por correlación
fig, ax = plt.subplots(figsize=(8, 4.5))

top10 = correlaciones.head(10)
colores_barras = ["#E8820C" if v > UMBRAL else "#555" for v in top10.values]

top10.plot(kind="barh", color=colores_barras, ax=ax, edgecolor="white")

ax.axvline(UMBRAL, color="white", linestyle="--", linewidth=1.5,
           label=f"Umbral = {UMBRAL}")
ax.set_title("Variables más relacionadas con el diagnóstico M/B",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Correlación absoluta")
ax.legend()
plt.tight_layout()
plt.show()

print("\n📌 INTERPRETACIÓN:")
print("   Las variables con el nombre 'worst' (peor valor) y 'mean' (promedio)")
print("   son las que más predicen si un tumor es maligno.")
print("   Tiene sentido médico: el tamaño y la irregularidad del núcleo celular")
print("   son señales clave de malignidad.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 5 — Preparar datos y dividir en entrenamiento/prueba

# COMMAND ----------

# ── Celda 5: Separar X (variables) e y (objetivo) ───────────────────────────

X = df[features_seleccionadas]   # Solo las variables seleccionadas
y = df["diagnosis_num"]          # 1=Maligno, 0=Benigno

# Dividir: 80% para entrenar, 20% para evaluar
# stratify=y asegura que la proporción M/B sea igual en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,    # Semilla para reproducibilidad
    stratify=y          # Mantiene la proporción de clases
)

print("División del dataset:")
print(f"   Entrenamiento: {X_train.shape[0]} casos ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"   Prueba:        {X_test.shape[0]} casos  ({X_test.shape[0]/len(df)*100:.0f}%)")
print()
print("Distribución en el conjunto de prueba:")
print(f"   Benignos:  {(y_test==0).sum()}")
print(f"   Malignos:  {(y_test==1).sum()}")
print()
print("✅ División correcta. Las proporciones se mantienen en ambos conjuntos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 6 — Construir y entrenar el modelo (Pipeline)

# COMMAND ----------

# ── Celda 6: Pipeline con StandardScaler + Regresión Logística ──────────────
#
# ¿Por qué un Pipeline?
#   Para que el escalado y el modelo se apliquen juntos de forma ordenada.
#   Evita errores de aplicar el scaler por separado.
#
# ¿Por qué StandardScaler?
#   La regresión logística es sensible a las escalas.
#   Algunas variables tienen valores en el rango 0-1 y otras llegan a 2000+.
#   StandardScaler las lleva todas a media=0 y desviación estándar=1.
#
# ¿Por qué max_iter=1000?
#   El optimizador necesita suficientes iteraciones para converger.
#   Con 100 (el default) a veces no alcanza.

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LogisticRegression(max_iter=1000, random_state=42))
])

# Entrenar: el modelo aprende los patrones del 80% de los datos
pipeline.fit(X_train, y_train)

# Predecir sobre el 20% que el modelo nunca vio
y_pred = pipeline.predict(X_test)

print("✅ Modelo entrenado y predicciones generadas.")
print(f"   Variables usadas: {X_train.shape[1]}")
print(f"   Casos de prueba:  {X_test.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 7 — Evaluación: Reporte de clasificación

# COMMAND ----------

# ── Celda 7: classification_report ──────────────────────────────────────────
#
# Este reporte muestra para CADA clase (Benigno y Maligno):
#
#  precision → de lo que dije que era maligno, ¿cuánto SÍ lo era?
#  recall    → de todos los malignos reales, ¿cuántos detecté?
#  f1-score  → balance entre precision y recall (0=pésimo, 1=perfecto)
#  support   → cuántos casos reales hay de esa clase

print("═" * 55)
print("REPORTE DE CLASIFICACIÓN — Breast Cancer")
print("═" * 55)
print(classification_report(
    y_test, y_pred,
    target_names=["Benigno (0)", "Maligno (1)"]
))

# COMMAND ----------

# Interpretación automática del reporte
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)       # Para la clase Maligno (1)
rec  = recall_score(y_test, y_pred)           # Para la clase Maligno (1)
f1   = f1_score(y_test, y_pred)               # Para la clase Maligno (1)

print("📌 INTERPRETACIÓN DE LAS MÉTRICAS:")
print()
print(f"  Accuracy:  {acc*100:.1f}%")
print(f"  → El modelo acierta en el {acc*100:.1f}% de los casos.")
print()
print(f"  Precisión (Maligno): {prec*100:.1f}%")
print(f"  → De cada 100 veces que el modelo dice 'maligno',")
print(f"    {prec*100:.0f} realmente lo son.")
print()
print(f"  Recall (Maligno): {rec*100:.1f}%")
print(f"  → De cada 100 tumores malignos reales,")
print(f"    el modelo detecta {rec*100:.0f}.")
print(f"    Los {100-rec*100:.0f} restantes son Falsos Negativos (no detectados).")
print()
print(f"  F1-Score (Maligno): {f1:.3f}")
print(f"  → Balance entre precisión y recall. Cercano a 1.0 = muy bueno.")
print()
if rec >= 0.95:
    print("  ✅ Recall alto: el modelo detecta casi todos los malignos.")
elif rec >= 0.90:
    print("  ⚠️ Recall aceptable, pero considerar bajar el umbral de decisión.")
else:
    print("  ❌ Recall bajo: el modelo deja escapar muchos malignos reales.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 8 — Matriz de confusión

# COMMAND ----------

# ── Celda 8: Matriz de confusión visual ─────────────────────────────────────
#
# La matriz muestra los 4 tipos de resultados posibles:
#
#   TN (arriba izq): dijo Benigno  → era Benigno   ✅ correcto
#   FP (arriba der): dijo Maligno  → era Benigno   ⚠️ alarma falsa
#   FN (abajo izq):  dijo Benigno  → era Maligno   ❌ PELIGROSO
#   TP (abajo der):  dijo Maligno  → era Maligno   ✅ correcto

fig, ax = plt.subplots(figsize=(5, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=["Benigno", "Maligno"],
    colorbar=False,
    cmap="Purples",
    ax=ax
)

ax.set_title("Matriz de Confusión — Breast Cancer",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Desglose numérico de la matriz
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("═" * 60)
print("DESGLOSE DE LA MATRIZ DE CONFUSIÓN")
print("═" * 60)
print(f"  TN — Verdadero Negativo:  {TN:3d}   Dijo Benigno  | Era Benigno   ✅")
print(f"  FP — Falso Positivo:      {FP:3d}   Dijo Maligno  | Era Benigno   ⚠️")
print(f"  FN — Falso Negativo:      {FN:3d}   Dijo Benigno  | Era Maligno   ❌")
print(f"  TP — Verdadero Positivo:  {TP:3d}   Dijo Maligno  | Era Maligno   ✅")
print()
print("📌 INTERPRETACIÓN:")
print()
print(f"  Los {FN} Falsos Negativos son el error más grave en medicina.")
print(f"  Son pacientes a quienes el modelo les diría 'todo está bien'")
print(f"  cuando en realidad tienen un tumor maligno.")
print()
print(f"  Los {FP} Falsos Positivos son una alarma falsa.")
print(f"  Son molestos e implican estudios extra, pero no son peligrosos.")
print()
total_real_malignos = FN + TP
print(f"  En resumen: de {total_real_malignos} tumores malignos reales,")
print(f"  el modelo detectó {TP} ({TP/total_real_malignos*100:.1f}%) y se le escaparon {FN} ({FN/total_real_malignos*100:.1f}%).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 9 — Distribución de probabilidades predichas

# COMMAND ----------

# ── Celda 9: ¿Qué tan seguro está el modelo en cada predicción? ─────────────
#
# predict_proba devuelve la probabilidad de cada clase.
# Tomamos la columna [:, 1] = probabilidad de ser Maligno.
#
# Un buen modelo debería:
#   - Agrupar los benignos cerca de 0%
#   - Agrupar los malignos cerca de 100%
# Si se mezclan cerca del 50%, el modelo tiene incertidumbre en esa zona.

probabilidades_maligno = pipeline.predict_proba(X_test)[:, 1]

fig, ax = plt.subplots(figsize=(9, 4.5))

# Benignos reales
ax.hist(probabilidades_maligno[y_test == 0], bins=25,
        alpha=0.65, color="#3B1F5E", label="Benigno real", edgecolor="white")

# Malignos reales
ax.hist(probabilidades_maligno[y_test == 1], bins=25,
        alpha=0.65, color="#E8820C", label="Maligno real", edgecolor="white")

# Línea del umbral de decisión
ax.axvline(0.5, color="white", linestyle="--", linewidth=2,
           label="Umbral = 50%")

ax.set_title("¿Qué tan seguro está el modelo en cada caso?",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Probabilidad predicha de ser Maligno")
ax.set_ylabel("Cantidad de casos")
ax.legend()
plt.tight_layout()
plt.show()

print("📌 INTERPRETACIÓN:")
print("   Si las dos barras están bien separadas (morado cerca del 0,")
print("   naranja cerca del 1), el modelo distingue bien entre clases.")
print("   Si se solapan mucho cerca del 50%, hay casos dudosos")
print("   donde el modelo no tiene certeza. En medicina, esos casos")
print("   deberían enviarse a revisión adicional.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Celda 10 — Resumen final del modelo

# COMMAND ----------

# ── Celda 10: Resumen ejecutivo del modelo ───────────────────────────────────
# Este resumen es lo que le presentarías a un director médico o gerente.

print("═" * 60)
print("RESUMEN DEL MODELO — Regresión Logística, Breast Cancer")
print("═" * 60)
print()
print(f"  Dataset:           569 biopsias (569 pacientes)")
print(f"  Variables usadas:  {len(features_seleccionadas)} de 30 disponibles")
print(f"  División:          80% entrenamiento / 20% prueba")
print()
print("  MÉTRICAS SOBRE EL CONJUNTO DE PRUEBA:")
print(f"  ├─ Accuracy:              {acc*100:.1f}%")
print(f"  ├─ Precisión (Maligno):   {prec*100:.1f}%")
print(f"  ├─ Recall    (Maligno):   {rec*100:.1f}%")
print(f"  └─ F1-Score  (Maligno):   {f1:.3f}")
print()
print("  ERRORES DEL MODELO:")
print(f"  ├─ Falsos Positivos (alarmas falsas): {FP}")
print(f"  └─ Falsos Negativos (malignos no detectados): {FN}  ← los más críticos")
print()
print("  CONCLUSIÓN DE NEGOCIO:")
if rec >= 0.93:
    print(f"  El modelo detecta el {rec*100:.1f}% de los tumores malignos.")
    print(f"  Para uso clínico de apoyo es un resultado sólido,")
    print(f"  pero siempre debe complementarse con criterio médico.")
else:
    print(f"  El recall de {rec*100:.1f}% puede mejorarse ajustando el umbral")
    print(f"  o explorando más variables del dataset.")
print()
print("═" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✏️ Tu reflexión (completar antes de entregar)
# MAGIC
# MAGIC Respondé estas preguntas en esta celda Markdown:
# MAGIC
# MAGIC **1. ¿Cuántos Falsos Negativos obtuvo tu modelo? ¿Qué significa eso en términos médicos?**
# MAGIC
# MAGIC El modelo logró obtener la cantidad de Falsos Negativos identificada en la celda anterior (FN). Desde un punto de vista médico, este dato implica que ese número de pacientes con cáncer maligno fue incorrectamente catalogado como benigno, lo que representa el error más grave, dado que esto conlleva a que no reciban el tratamiento adecuado en el momento oportuno.
# MAGIC
# MAGIC **2. ¿Por qué el recall es más importante que la accuracy en este contexto?**
# MAGIC
# MAGIC El recall resulta más relevante porque evalúa la habilidad del modelo para identificar todos los casos de cáncer maligno. En el ámbito médico, es crucial reducir al mínimo los Falsos Negativos, ya que la falta de detección de un caso maligno puede generar graves repercusiones. La accuracy puede mostrarse alta incluso cuando el modelo no logra identificar malignos, si la mayoría de los casos son benignos.
# MAGIC
# MAGIC **3. Si bajaras el umbral de decisión de 50% a 30%, ¿cómo cambiaría el recall? ¿Y la precisión?**
# MAGIC
# MAGIC Si se disminuye el umbral al 30%, el modelo clasificará un mayor número de casos como malignos, lo que incrementará el recall (menos Falsos Negativos), aunque al mismo tiempo se reducirá la precisión (aumento de Falsos Positivos), ya que existirán más alarmas erróneas.
# MAGIC
# MAGIC **4. ¿Cómo aplicarías este mismo modelo a tu dataset del proyecto personal?**
# MAGIC
# MAGIC Aplicaría un proceso similar: seleccionaría las variables pertinentes, dividiría los datos en conjuntos de entrenamiento y prueba, entrenaría un modelo de clasificación (como la regresión logística), evaluaría utilizando métricas como la accuracy, el recall y la matriz de confusión, y ajustaría el umbral teniendo en cuenta la relevancia de minimizar errores críticos según mi contexto.
# MAGIC
# MAGIC ---
# MAGIC *Semana 7 — 92-0030 Diagnóstico y Predictibilidad — Robin Sequeira — ULACIT — I Cuatrimestre 2026*