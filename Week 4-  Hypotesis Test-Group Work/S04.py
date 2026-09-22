# Databricks notebook source
# ---------------------------------------------------------------
# BLOQUE 1 — Carga y primera mirada
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Cargamos el Superstore como base de demostración.
# En la Parte 2 van a reemplazar esto con su propio dataset.
pdf = spark.table("workspace.default.global_superstore").toPandas()

df = pdf.copy()

print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}")
print("\nColumnas disponibles:")
print(df.columns.tolist())

# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE 5 — Correlación entre variables
# ---------------------------------------------------------------
# La correlación mide qué tan relacionadas están dos variables.
# Rango: -1 (inversa perfecta) a +1 (directa perfecta).
# 0 = sin relación lineal.
#
# En el Módulo 3 usaron esto para seleccionar variables.
# Hoy entienden por qué ese criterio tiene sentido estadístico.

corr = df.select_dtypes(include='number').corr()

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, ax=ax)
ax.set_title('Correlaciones — Superstore DS')
plt.tight_layout()
plt.show()

# Hallazgos clave:
# Sales  - Profit:   r ≈ +0.48 → positiva moderada
# Discount - Profit: r ≈ -0.22 → negativa débil (más descuento, menos ganancia)
# Discount - Sales:  r ≈ -0.03 → prácticamente nula


# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE 6 — Prueba de hipótesis: ¿la diferencia es real?
# ---------------------------------------------------------------
# H0: no hay diferencia entre grupos (lo que asumimos por defecto)
# H1: sí hay diferencia real (lo que queremos demostrar)
#
# El valor p: si p < 0.05 → rechazamos H0 → diferencia probablemente real.
# IMPORTANTE: p < 0.05 no dice que la diferencia sea grande o importante.

g1 = df[df['Category'] == 'Technology']['Profit']
g2 = df[df['Category'] == 'Furniture']['Profit']

t_stat, p_val = stats.ttest_ind(g1, g2)

print("Prueba t — Technology vs Furniture (Profit)")
print(f"  Media Technology: ${g1.mean():.2f}")
print(f"  Media Furniture:  ${g2.mean():.2f}")
print(f"  Diferencia:       ${g1.mean() - g2.mean():.2f}")
print(f"  Valor p:          {p_val:.4f}")
if p_val < 0.05:
    print("  → Diferencia estadísticamente significativa.")
else:
    print("  → No hay evidencia suficiente de diferencia.")


# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE A — Cargar su dataset
# ---------------------------------------------------------------
# Reemplacen esto con la carga de su propio dataset.
# Si lo tienen en Databricks:
#   mi_df = spark.table("workspace.default.SU_TABLA").toPandas()
# Si lo tienen como CSV:
#   mi_df = pd.read_csv("/ruta/a/su/archivo.csv")

# Ejemplo genérico — cambien por su dataset real:
mi_df = spark.table("workspace.default.global_superstore").toPandas()
# o:
# mi_df = pd.read_csv("/dbfs/FileStore/su_archivo.csv")

print(f"Dataset: {mi_df.shape[0]:,} filas, {mi_df.shape[1]} columnas")
print("\nmi_df.shape:")
print(mi_df.shape)
print("\nmi_df.dtypes:")
print(mi_df.dtypes)
print("\nPorcentaje de nulos por columna:")
print(mi_df.isna().mean().sort_values(ascending=False) * 100)
print("\nFilas duplicadas:")
print(mi_df.duplicated().sum())
print("\nValores únicos por columna:")
print(mi_df.nunique().sort_values(ascending=False))
print("\nColumnas:")
print(mi_df.columns.tolist())
print("\nPrimeras filas:")
mi_df.head()

# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE B — Estadística descriptiva de su dataset
# ---------------------------------------------------------------

# Resumen general
print("=== Resumen estadístico ===")
print(mi_df.describe().round(2))

# Identifiquen sus columnas numéricas clave
# (las que son relevantes para su problema de negocio)
COLUMNA_OBJETIVO = "Profit"   # cambia esto si tu variable objetivo es otra

print(f"\n=== Análisis de {COLUMNA_OBJETIVO} ===")
print(f"Media:    {mi_df[COLUMNA_OBJETIVO].mean():.2f}")
print(f"Mediana:  {mi_df[COLUMNA_OBJETIVO].median():.2f}")
print(f"Std:      {mi_df[COLUMNA_OBJETIVO].std():.2f}")
print(f"Sesgo:    {mi_df[COLUMNA_OBJETIVO].skew():.2f}")

# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE C — Distribución de la variable principal
# ---------------------------------------------------------------
# Grafiquen la distribución de su variable más importante.

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(mi_df[COLUMNA_OBJETIVO].dropna(), bins=40, color='#3B1F5E', edgecolor='white')
axes[0].axvline(mi_df[COLUMNA_OBJETIVO].mean(),   color='#E8820C', lw=2, ls='--', label='Media')
axes[0].axvline(mi_df[COLUMNA_OBJETIVO].median(), color='white',   lw=2, ls='-',  label='Mediana')
axes[0].set_title(f'Distribución de {COLUMNA_OBJETIVO}')
axes[0].legend()

# Boxplot para visualizar outliers
axes[1].boxplot(mi_df[COLUMNA_OBJETIVO].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor='#3B1F5E', color='white'),
                medianprops=dict(color='#E8820C', linewidth=2))
axes[1].set_title(f'Boxplot — {COLUMNA_OBJETIVO}')

plt.suptitle(f'Análisis de distribución — {COLUMNA_OBJETIVO}', fontsize=13)
plt.tight_layout()
plt.show()


# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE D — Correlaciones en su dataset
# ---------------------------------------------------------------
# ¿Qué variables numéricas están relacionadas entre sí?
# ¿Cuál tiene mayor relación con su variable objetivo?

import matplotlib.pyplot as plt
import seaborn as sns

numericas = mi_df.select_dtypes(include='number')

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(numericas.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax)
ax.set_title(f'Correlaciones — {mi_df.shape[0]:,} registros')
plt.tight_layout()
plt.show()

# Tabla de correlación con la variable objetivo
print(f"\nCorrelación con {COLUMNA_OBJETIVO}:")
corr_obj = numericas.corr()[COLUMNA_OBJETIVO].drop(COLUMNA_OBJETIVO)
print(corr_obj.sort_values(key=abs, ascending=False).round(3))


# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE E — Análisis por segmento (si aplica)
# ---------------------------------------------------------------
# Si su dataset tiene una columna categórica relevante
# (tipo de cliente, categoría de producto, región, etc.)
# úsenla para ver si el comportamiento cambia entre grupos.

COLUMNA_SEGMENTO = "Category"   # cambia esto

print(f"=== {COLUMNA_OBJETIVO} por {COLUMNA_SEGMENTO} ===")
print(mi_df.groupby(COLUMNA_SEGMENTO)[COLUMNA_OBJETIVO]
      .agg(['mean', 'median', 'std', 'count'])
      .round(2))

# Visualización comparativa
fig, ax = plt.subplots(figsize=(10, 4))
grupos = [mi_df[mi_df[COLUMNA_SEGMENTO] == g][COLUMNA_OBJETIVO].dropna()
          for g in mi_df[COLUMNA_SEGMENTO].unique()]
etiquetas = mi_df[COLUMNA_SEGMENTO].unique().tolist()
ax.boxplot(grupos, tick_labels=etiquetas, patch_artist=True)
ax.set_title(f'{COLUMNA_OBJETIVO} por {COLUMNA_SEGMENTO}')
ax.set_ylabel(COLUMNA_OBJETIVO)
plt.tight_layout()
plt.show()




# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE E — Análisis por segmento (si aplica)
# ---------------------------------------------------------------
# Si su dataset tiene una columna categórica relevante
# (tipo de cliente, categoría de producto, región, etc.)
# úsenla para ver si el comportamiento cambia entre grupos.

COLUMNA_SEGMENTO = "Category"   # cambia esto

print(f"=== {COLUMNA_OBJETIVO} por {COLUMNA_SEGMENTO} ===")
display(mi_df.groupby(COLUMNA_SEGMENTO)[COLUMNA_OBJETIVO]
        .agg(['mean', 'median', 'std', 'count'])
        .round(2))

# Visualización comparativa
fig, ax = plt.subplots(figsize=(10, 4))
grupos = [mi_df[mi_df[COLUMNA_SEGMENTO] == g][COLUMNA_OBJETIVO].dropna()
          for g in mi_df[COLUMNA_SEGMENTO].unique()]
etiquetas = mi_df[COLUMNA_SEGMENTO].unique().tolist()
ax.boxplot(grupos, tick_labels=etiquetas, patch_artist=True)
ax.set_title(f'{COLUMNA_OBJETIVO} por {COLUMNA_SEGMENTO}')
ax.set_ylabel(COLUMNA_OBJETIVO)
plt.tight_layout()
plt.show()




# COMMAND ----------

# ---------------------------------------------------------------
# BLOQUE F — Prueba de hipótesis sobre sus datos
# ---------------------------------------------------------------
# Elijan dos grupos de su dataset y prueben si la diferencia
# en la variable objetivo es estadísticamente significativa.
#
# Ejemplo: ¿el grupo A tiene un valor promedio diferente al grupo B?

from scipy import stats

GRUPO_A = "Technology"   # cambia esto
GRUPO_B = "Furniture"   # cambia esto

datos_A = mi_df[mi_df[COLUMNA_SEGMENTO] == GRUPO_A][COLUMNA_OBJETIVO].dropna()
datos_B = mi_df[mi_df[COLUMNA_SEGMENTO] == GRUPO_B][COLUMNA_OBJETIVO].dropna()

t_stat, p_val = stats.ttest_ind(datos_A, datos_B)

print(f"Prueba t — {GRUPO_A} vs {GRUPO_B}")
print(f"  Media {GRUPO_A}: {datos_A.mean():.2f}")
print(f"  Media {GRUPO_B}: {datos_B.mean():.2f}")
print(f"  Diferencia:      {datos_A.mean() - datos_B.mean():.2f}")
print(f"  Valor p:         {p_val:.4f}")
print()
if p_val < 0.05:
    print("  → Diferencia estadísticamente significativa (p < 0.05).")
    print("  → Ahora la pregunta importante: ¿es relevante para su negocio?")
else:
    print("  → No hay evidencia suficiente de diferencia real.")
    print("  → Eso también es información: los grupos se comportan igual.")


# COMMAND ----------

# MAGIC %md BLOQUE A — Cargar dataset
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿Cuántas filas y columnas tiene el dataset? 51.290 filas y 2 columnas.
# MAGIC
# MAGIC ¿Qué columnas contiene? Category, City, Country, Customer ID, Customer Name, Discount, Market, ji_lu-shu, Order Date, Order ID, Order Priority, Product ID, Product Name, Profit, Quantity, Region, Row ID, Sales, Segment, Ship Date, Ship Mode, Shipping Cost, State, Sub-Category, Year, Market2, weeknum.
# MAGIC
# MAGIC ¿Las primeras filas se ven correctas? Si.
# MAGIC
# MAGIC ¿El dataset cargó bien o hay problemas visibles desde el inicio? No se ven problemas.
# MAGIC
# MAGIC
# MAGIC
# MAGIC BLOQUE B — Estadística descriptiva
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿Cuáles son las variables numéricas clave para el problema de negocio? Las variables numéricas clave son: Sales, Profit, Discount, Quantity, Shipping Cost.
# MAGIC
# MAGIC ¿Cuál es la variable objetivo principal? Análisis del "Profit".
# MAGIC
# MAGIC ¿La media y la mediana son similares? No.
# MAGIC
# MAGIC Si la media y la mediana son diferentes, ¿hay sesgo? Sí.
# MAGIC
# MAGIC ¿La desviación estándar es alta? Sí.
# MAGIC
# MAGIC ¿Hay valores extremos que puedan afectar el análisis? Sí.
# MAGIC
# MAGIC ¿La variable objetivo parece estable o muy dispersa? Muy dispersa.
# MAGIC
# MAGIC
# MAGIC BLOQUE C — Distribución de la variable principal
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿La distribución se parece a una campana normal? No, la mayoría concentrado alrededor del cero.
# MAGIC
# MAGIC ¿Tiene cola larga? Sí.
# MAGIC
# MAGIC ¿La cola está hacia la derecha o hacia la izquierda? Derecha, sesgo positivo.
# MAGIC
# MAGIC ¿La media está lejos de la mediana? Sí  desde el punto de vista de que la media es 28.61 y la mediana es 9.24, en el gráfico no es fácil de distinguir.
# MAGIC
# MAGIC ¿Existen outliers visibles en el boxplot? Sí, hacias los extremos.
# MAGIC
# MAGIC ¿La variable objetivo necesita tratamiento especial antes de modelar? Sí, vea la respuesta anterior.
# MAGIC
# MAGIC
# MAGIC
# MAGIC BLOQUE D — Correlaciones
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿Qué variables numéricas están relacionadas entre sí? Sales vs Shipping Cost, Profit vs Sales, Profit vs Shipping Cost, Profit vs Discount. Sales vs Quantity, Shipping Cost vs Quantity.
# MAGIC
# MAGIC ¿Qué variable tiene mayor relación con la variable objetivo? Sales.
# MAGIC
# MAGIC ¿Hay correlaciones fuertes, por ejemplo mayores a 0.5 o menores a -0.5? Sí, Sales vs Shipping Cost = 0.77.
# MAGIC
# MAGIC ¿Hay alguna relación inesperada? No.
# MAGIC
# MAGIC ¿Qué variable tiene más potencial como predictora? Sales.
# MAGIC
# MAGIC ¿Hay variables redundantes que miden casi lo mismo? No visiblemente.
# MAGIC
# MAGIC
# MAGIC BLOQUE E — Análisis por segmento
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿Qué columna categórica es relevante para segmentar el análisis? Category.
# MAGIC
# MAGIC ¿El comportamiento de la variable objetivo cambia entre grupos? Sí.
# MAGIC
# MAGIC ¿Qué grupo tiene mayor media? Technology.
# MAGIC
# MAGIC ¿Qué grupo tiene mayor mediana? Technology.
# MAGIC
# MAGIC ¿Qué grupo tiene mayor dispersión? Technology.
# MAGIC
# MAGIC ¿Hay segmentos con más riesgo, pérdida, variabilidad o valor? Riesgo = Office Supplies, Pérdidas = Office Supplies, Variabilidad = Technology.
# MAGIC
# MAGIC ¿Los grupos se comportan igual o hay diferencias claras? No tienen comportamiento similar.
# MAGIC
# MAGIC
# MAGIC BLOQUE F — Prueba de hipótesis
# MAGIC
# MAGIC Preguntas:
# MAGIC
# MAGIC ¿Qué dos grupos se van a comparar? Technology y Office Supplies.
# MAGIC
# MAGIC ¿La diferencia entre los grupos es estadísticamente significativa? Sí
# MAGIC
# MAGIC ¿El valor p es menor que 0.05? Sí.
# MAGIC
# MAGIC Si la diferencia es significativa, ¿también es relevante para el negocio? Sí.
# MAGIC
# MAGIC Si no hay diferencia significativa, ¿qué implica eso? Que no tienen relevancia/evidencia estadísticamente hablando.
# MAGIC
# MAGIC ¿La comparación ayuda a tomar una decisión real? Sí claramente.
# MAGIC
# MAGIC REFLEXIÓN PARA EL PORTAFOLIO I
# MAGIC
# MAGIC Preguntas finales:
# MAGIC
# MAGIC ¿Qué aprendió el grupo sobre sus propios datos? Que existe mucha dispersión.
# MAGIC
# MAGIC ¿Hubo algo inesperado? La dispersion alta que existe en el dataset, negativa así como positiva.
# MAGIC
# MAGIC ¿Qué diferencias notaron entre trabajar con Superstore y trabajar con su dataset real? Son bastante similares en context, pero el dataset que escogimos tiene mucha más dispersidad.
# MAGIC
# MAGIC ¿Cómo conecta este análisis estadístico con el pipeline del Módulo 3? como prepara el dataset para el análisis de una forma correcta.
# MAGIC
# MAGIC ¿Qué ajuste harían al diseño del proyecto con base en lo descubierto? Remover "Outliers" por segment (category), analizarlos por los "quartiles" y verificar como afecta la rentabilidad.
# MAGIC
# MAGIC Las 5 preguntas fuertes para la presentación
# MAGIC
# MAGIC Estas son las que yo usaría:
# MAGIC
# MAGIC ¿Cuál es la variable principal del proyecto y qué descubrimos sobre su comportamiento? La variable principal es "Profit", la cuál está muy dispersa.
# MAGIC
# MAGIC ¿La distribución de esa variable es normal, sesgada o tiene outliers importantes? No es normal y dispersa, sesgado hacia la derecho.
# MAGIC
# MAGIC ¿Qué variables parecen tener mayor relación con la variable objetivo? Shipping cost y en discount en menor medida que shipping cost.
# MAGIC
# MAGIC ¿El comportamiento cambia entre segmentos del dataset?  Sí cambian por categoría.	
# MAGIC
# MAGIC ¿Qué decisión o ajuste haríamos al proyecto después de este análisis? Remover los "outliers", existen cierta categorías con valores extremos.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC