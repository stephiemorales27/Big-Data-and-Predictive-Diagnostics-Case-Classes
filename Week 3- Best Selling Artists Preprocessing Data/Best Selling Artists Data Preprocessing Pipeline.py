# Databricks notebook source
# DBTITLE 1,Data preprocessing pipeline for best_selling_artists dataset
# CELDA 1 — Punto de partida: cargar el dataset
# (Si ya tienes df_clean del Módulo 2, puedes saltar esta celda)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Carga desde la tabla Spark (misma fuente que Módulos 1 y 2)
df_raw = spark.table("workspace.default.best_selling_artists").toPandas().copy()

# Limpieza de datos: convertir TCU y Sales de texto a numérico
def clean_million_string(value):
    """Convierte strings como '290.4 million' a float"""
    if pd.isna(value):
        return np.nan
    return float(str(value).replace(' million', '').strip())

def clean_sales_string(value):
    """Convierte valores de Sales a float, manejando rangos y concatenaciones"""
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # Caso 1: Rango con guión "90–100 million" -> promedio
    if '–' in value_str or '-' in value_str:
        value_str = value_str.replace('–', '-')
        parts = value_str.replace(' million', '').split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            pass
    
    # Caso 2: Valores concatenados "600 million500 million" -> tomar el primero
    if value_str.count('million') > 1:
        # Extraer el primer número antes del primer 'million'
        first_value = value_str.split('million')[0].strip()
        return float(first_value)
    
    # Caso 3: Formato simple "500 million"
    return float(value_str.replace(' million', '').strip())

df_clean = df_raw.copy()
df_clean['TCU'] = df_clean['TCU'].apply(clean_million_string)
df_clean['Sales'] = df_clean['Sales'].apply(clean_sales_string)

print("Dataset cargado y limpiado correctamente.")
print("Shape:", df_clean.shape)
print("Columnas:", df_clean.columns.tolist())
print("\nTipos de datos después de limpieza:")
print(df_clean.dtypes)


# ============================================================
# CELDA 2 — Clasificar las columnas del Superstore
# (Decisión humana: qué tipo es cada columna)
# ============================================================

# Columnas que usaremos como predictoras
# Excluimos 'Artist' por alta cardinalidad (121 artistas únicos)
columnas_excluir = ["Artist"]

# Variable objetivo
target = "Sales"

# Columnas numéricas (irán con StandardScaler)
num_cols = ["Year", "TCU"]

# Columnas categóricas (irán con OneHotEncoder)
cat_cols = ["Country", "period_active", "Genre"]

print("Variable objetivo:", target)
print("\nColumnas numéricas:", num_cols)
print("\nColumnas categóricas:", cat_cols)


# ============================================================
# CELDA 3 — Preparar X (predictoras) e y (objetivo)
# ============================================================

X = df_clean[num_cols + cat_cols].copy()
y = df_clean[target].copy()

print("Shape de X:", X.shape)
print("Shape de y:", y.shape)
print("\nPrimeras filas de X:")
X.head()


# ============================================================
# CELDA 4 — Crear los transformadores individuales
# ============================================================

# Para columnas numéricas: estandariza a media=0, desv. estándar=1
num_transformer = StandardScaler()

# Para columnas categóricas: genera una columna binaria por categoría
# handle_unknown='ignore' evita errores si en producción aparece una categoría nueva
cat_transformer = OneHotEncoder(handle_unknown="ignore")

print("Transformadores creados:")
print("  Numéricas  →", num_transformer)
print("  Categóricas →", cat_transformer)


# ============================================================
# CELDA 5 — Construir el ColumnTransformer
# Aplica el transformador correcto a cada grupo de columnas
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),   # StandardScaler a columnas numéricas
        ("cat", cat_transformer, cat_cols)    # OneHotEncoder a columnas categóricas
    ],
    remainder="drop"    # descarta cualquier otra columna
)

print("ColumnTransformer configurado correctamente.")
print("Grupos:")
print("  Numéricas:", num_cols)
print("  Categóricas:", cat_cols)


# ============================================================
# CELDA 6 — Construir el Pipeline completo
# ============================================================

pipe = Pipeline(steps=[
    ("prep", preprocessor)
])

print("Pipeline creado.")
print("Pasos:", [step[0] for step in pipe.steps])


# ============================================================
# CELDA 7 — CORRECTO: Dividir PRIMERO, luego ajustar el Pipeline
# Esto evita el data leakage
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("División completada:")
print("  X_train:", X_train.shape)
print("  X_test:", X_test.shape)
print("  y_train:", y_train.shape)
print("  y_test:", y_test.shape)


# ============================================================
# CELDA 8 — Ajustar y transformar con el Pipeline
# fit SOLO con train, luego transform ambos conjuntos
# ============================================================

# El Pipeline aprende solo con los datos de entrenamiento
X_train_t = pipe.fit_transform(X_train)

# Luego transforma el test usando lo que aprendió (sin verlo de nuevo)
X_test_t  = pipe.transform(X_test)

print("Pipeline aplicado correctamente.")
print("Shape X_train transformado:", X_train_t.shape)
print("Shape X_test transformado: ", X_test_t.shape)

# Todas las columnas son ahora numéricas
print("\nTipo de datos:", type(X_train_t))


# ============================================================
# CELDA 9 — ¿Qué columnas generó el OneHotEncoder?
# ============================================================

# Nombres de las columnas generadas
cat_feature_names = pipe.named_steps["prep"] \
    .named_transformers_["cat"] \
    .get_feature_names_out(cat_cols)

all_feature_names = num_cols + list(cat_feature_names)

print(f"Total columnas transformadas: {len(all_feature_names)}")
print("\nColumnas numéricas (sin cambio de nombre):", num_cols)
print("\nColumnas generadas por OneHotEncoder:")
for name in cat_feature_names:
    print(" ", name)

# ============================================================
# CELDA 10 — Selección de características: correlación con Sales
# ============================================================

# Calculamos correlación de Pearson de las variables numéricas con Sales
# (antes de transformar, sobre el dataset original para legibilidad)

corr_con_sales = df_clean[num_cols + [target]].corr()[target].drop(target)
corr_ordenada  = corr_con_sales.abs().sort_values(ascending=False)

print("Correlación absoluta de las numéricas con Sales:\n")
for col, val in corr_ordenada.items():
    estado = "✓ INCLUIR" if val > 0.2 else "✗ descartable"
    print(f"  {col:20s}: {corr_con_sales[col]:+.3f}  ({val:.3f})  {estado}")

print("\nUmbral aplicado: correlación absoluta > 0.2")


# ============================================================
# CELDA 11 — Visualización: heatmap de correlaciones
# ============================================================

import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 5))

corr_matrix = df_clean[num_cols + [target]].corr()

sns.heatmap(
    corr_matrix,
    annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5,
    ax=ax
)

ax.set_title("Mapa de correlaciones — Variables numéricas del Superstore DS", fontsize=13)
plt.tight_layout()
plt.show()


# ============================================================
# CELDA 12 — Filtrar columnas por umbral de correlación
# ============================================================

UMBRAL = 0.2

# Columnas que superan el umbral
cols_seleccionadas = corr_con_sales[corr_con_sales.abs() > UMBRAL].index.tolist()
cols_descartadas   = corr_con_sales[corr_con_sales.abs() <= UMBRAL].index.tolist()

print(f"Columnas que pasan el filtro (|correlación| > {UMBRAL}):")
for col in cols_seleccionadas:
    print(f"  {col}: {corr_con_sales[col]:+.3f}")

print(f"\nColumnas descartables (|correlación| <= {UMBRAL}):")
for col in cols_descartadas:
    print(f"  {col}: {corr_con_sales[col]:+.3f}")


# ============================================================
# CELDA 13 — Pipeline con solo las columnas seleccionadas
# Reentrenamos con las columnas que pasaron el filtro de correlación
# ============================================================

# Actualizamos las listas
num_cols_sel = [col for col in cols_seleccionadas]
# (las categóricas se mantienen; el filtro de correlación se aplica a numéricas)

print("Columnas numéricas seleccionadas:", num_cols_sel)
print("Columnas categóricas (sin cambio):", cat_cols)

# Pipeline con columnas filtradas
preprocessor_sel = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_sel),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="drop"
)

pipe_sel = Pipeline(steps=[("prep", preprocessor_sel)])

X_sel = df_clean[num_cols_sel + cat_cols].copy()

X_train_s, X_test_s, _, _ = train_test_split(X_sel, y, test_size=0.2, random_state=42)

X_train_final = pipe_sel.fit_transform(X_train_s)
X_test_final  = pipe_sel.transform(X_test_s)

print("\nShape final (columnas seleccionadas):", X_train_final.shape)
print("Listo para entrenar el modelo en módulos siguientes.")


# ============================================================
# CELDA 14 — Resumen del Módulo 3
# ============================================================

print("=" * 55)
print("  RESUMEN DEL PREPROCESAMIENTO — Módulo 3")
print("=" * 55)
print()
print(f"Dataset original:         {df_clean.shape}")
print(f"Columnas numéricas usadas:  {num_cols}")
print(f"Columnas categóricas usadas: {cat_cols}")
print()
print(f"Después del Pipeline completo:")
print(f"  X_train transformado:   {X_train_t.shape}")
print(f"  X_test transformado:    {X_test_t.shape}")
print()
print(f"Después de selección por correlación (>{UMBRAL}):")
print(f"  X_train final:          {X_train_final.shape}")
print(f"  X_test final:           {X_test_final.shape}")
print()
print("Próximo paso: Módulo 4 — Fundamentos estadísticos")
print("  Usaremos este mismo dataset para estadística descriptiva,")
print("  distribuciones y heatmaps de correlación con seaborn.")
print("=" * 55)


# COMMAND ----------

# MAGIC %md
# MAGIC Metacognitivo 1: 
# MAGIC
# MAGIC **¿Qué aprendí esta semana?**  
# MAGIC Aprendí a limpiar y transformar datos usando pandas y scikit-learn, a construir pipelines de preprocesamiento y a seleccionar variables relevantes mediante correlación.
# MAGIC
# MAGIC **¿Qué me costó más?**  
# MAGIC Me costó entender cómo evitar el data leakage y cómo interpretar correctamente los resultados de la correlación para seleccionar variables.
# MAGIC
# MAGIC **¿Dónde lo aplicaría?**  
# MAGIC Lo aplicaría en cualquier proyecto de ciencia de datos donde sea necesario preparar datos antes de entrenar modelos, especialmente en contextos donde la calidad del preprocesamiento impacta el desempeño del modelo.