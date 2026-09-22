# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 6 — Regresión Lineal con Insurance
# MAGIC **Diagnóstico y Predictibilidad | 92-0030 | Prof. Robin Sequeira**
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Antes de ejecutar cualquier celda, respondan esto en grupo:
# MAGIC
# MAGIC > *¿Pueden predecir cuánto va a gastar una persona en salud solo con saber su edad, peso y si fuma?*
# MAGIC
# MAGIC Anoten su hipótesis aquí abajo. Al final del notebook vamos a comparar.
# MAGIC
# MAGIC **Nuestra hipótesis:** _(escriban aquí)_

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 1 — Cargar herramientas y datos
# MAGIC Ejecuten esta celda. Si aparece ✅, estamos listos para empezar.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargamos el dataset desde Databricks
df_raw = spark.table('workspace.default.insurance').toPandas()
df = df_raw.copy()

print(f'✅ Dataset cargado: {df.shape[0]} personas, {df.shape[1]} variables')
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 2 — Explorar los datos
# MAGIC
# MAGIC Antes de construir cualquier modelo, necesitamos entender qué hay en el dataset.
# MAGIC
# MAGIC **Ejecuten y respondan:**

# COMMAND ----------

# ¿Hay nulos o duplicados?
print(f'Nulos: {df.isna().sum().sum()}  |  Duplicados: {df.duplicated().sum()}')
print()
# ¿Cómo están distribuidos los costos médicos?
print('Estadísticos de charges (costo médico en USD):')
print(df['charges'].describe().round(2).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC **¿Qué notan?** _(completen)_
# MAGIC - El costo mínimo es: 2107.24___
# MAGIC - El costo máximo es:  69931.71
# MAGIC - El promedio es: 19605.36___
# MAGIC - ¿La mediana es muy diferente al promedio? ¿Qué sugiere eso?
# MAGIC El promedio es 19605.36, bastante más alto que la mediana (12835.09).
# MAGIC

# COMMAND ----------

# Veamos la distribución de costos separada por fumador y no fumador
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['charges'], bins=30, color='#E8820C', edgecolor='white')
axes[0].set_title('Distribución de Charges (todos)')
axes[0].set_xlabel('Costo médico (USD)')
axes[0].set_ylabel('Personas')

fuma    = df[df['smoker'] == 'yes']['charges']
no_fuma = df[df['smoker'] == 'no']['charges']
axes[1].hist(fuma,    bins=20, alpha=0.7, color='#E8820C', label='Fumador',    edgecolor='white')
axes[1].hist(no_fuma, bins=20, alpha=0.7, color='#3B1F5E', label='No fumador', edgecolor='white')
axes[1].set_title('Charges por condición de fumador')
axes[1].set_xlabel('Costo médico (USD)')
axes[1].set_ylabel('Personas')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f'Promedio fumadores:    ${fuma.mean():,.0f}')
print(f'Promedio no fumadores: ${no_fuma.mean():,.0f}')
print(f'Diferencia:            ${fuma.mean() - no_fuma.mean():,.0f}')

# COMMAND ----------

# MAGIC %md
# MAGIC **Discutan en grupo:**
# MAGIC - ¿Cuánto más gasta en promedio un fumador?
# MAGIC Esto significa que, en promedio, un fumador gasta 40,711 USD más en costos médicos que un no fumador. 
# MAGIC - ¿Creen que `smoker` va a ser la variable más importante del modelo?
# MAGIC
# MAGIC El hecho de fumar parece tener un impacto directo y fuerte en los gastos médicos, lo que lo convierte en una variable predictiva muy relevante.
# MAGIC
# MAGIC Sin embargo, no necesariamente será la única más importante: edad, índice de masa corporal (BMI), y condiciones médicas previas también pueden influir bastante.
# MAGIC - ¿Por qué la distribución del gráfico izquierdo tiene dos picos?
# MAGIC La presencia de dos picos (bimodalidad) sugiere que dentro del grupo de fumadores hay dos subgrupos distintos:
# MAGIC
# MAGIC Un grupo con costos médicos relativamente moderados.
# MAGIC
# MAGIC Otro grupo con costos médicos muy altos (probablemente fumadores con enfermedades graves como cáncer, problemas cardíacos o respiratorios).
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 3 — Preparar los datos para el modelo
# MAGIC
# MAGIC El modelo necesita:
# MAGIC - **y** → lo que queremos predecir (`charges`)
# MAGIC - **X** → todo lo demás (las variables que usa para predecir)
# MAGIC
# MAGIC También separamos las columnas por tipo porque cada tipo necesita un tratamiento diferente.

# COMMAND ----------

y = df['charges']
X = df.drop(columns=['charges'])

# Numéricas: se escalan para que estén en la misma unidad
num_cols = ['age', 'bmi', 'children']

# Categóricas: se convierten en columnas de 0 y 1
# (el modelo no puede leer 'male' o 'female', pero sí puede leer 1 y 0)
cat_cols = ['sex', 'smoker', 'region']

print(f'Variable a predecir: charges')
print(f'Numéricas:   {num_cols}')
print(f'Categóricas: {cat_cols}')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 4 — Dividir en entrenamiento y prueba
# MAGIC
# MAGIC **Regla importante:** dividimos ANTES de transformar nada.  
# MAGIC Si transformamos primero, el modelo vería datos que no debería → **data leakage** (fuga de datos).

# COMMAND ----------

# 80% para que el modelo aprenda, 20% para evaluar si realmente aprendió
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'El modelo va a aprender con: {X_train.shape[0]} personas')
print(f'Lo vamos a evaluar con:      {X_test.shape[0]} personas que nunca vio')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 5 — Construir y entrenar el modelo
# MAGIC
# MAGIC Usamos un **Pipeline** que hace todo en orden automáticamente:
# MAGIC 1. Escala los números
# MAGIC 2. Convierte el texto en 0s y 1s
# MAGIC 3. Entrena la regresión lineal

# COMMAND ----------

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(),                       num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
])

pipe_lr = Pipeline(steps=[
    ('prep',  preprocessor),
    ('model', LinearRegression()),
])

# El modelo aprende solo con los datos de entrenamiento
pipe_lr.fit(X_train, y_train)
print('✅ Modelo entrenado. Ahora vamos a ver qué tan bien predice.')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 6 — ¿Qué tan bien predice el modelo?
# MAGIC
# MAGIC Evaluamos sobre los datos de **prueba** (los que el modelo nunca vio).
# MAGIC
# MAGIC - **MAE** → error promedio en dólares
# MAGIC - **RMSE** → castiga más los errores grandes
# MAGIC - **R²** → qué proporción de la variación en costos logra explicar el modelo (de 0 a 1)

# COMMAND ----------

y_pred_lr = pipe_lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)

print('REGRESIÓN LINEAL — resultados sobre datos de prueba:')
print(f'  MAE:  ${mae_lr:,.0f}   → en promedio nos equivocamos este monto por persona')
print(f'  RMSE: ${rmse_lr:,.0f}  → penaliza más los errores grandes')
print(f'  R²:   {r2_lr:.3f}    → el modelo explica el {r2_lr*100:.1f}% de la variación en costos')

# COMMAND ----------

# MAGIC %md
# MAGIC **Discutan en grupo:**
# MAGIC - Si el MAE es $4,500 y una prima mensual promedio es $350 ($4,200/año), ¿ese error es aceptable para una aseguradora?
# MAGIC Si el MAE es $4,500 y la prima anual promedio es $4,200, significa que el error esperado por persona es mayor que lo que se cobra en promedio por año.
# MAGIC
# MAGIC Para una aseguradora, esto puede ser problemático, porque el modelo estaría equivocándose más de lo que representa la prima anual típica.
# MAGIC
# MAGIC En términos prácticos: el modelo podría subestimar o sobreestimar costos de manera que afecte la rentabilidad o la equidad en las primas.
# MAGIC - ¿Qué significa un R²=0.75 en términos del negocio?
# MAGIC Un R² de 0.75 significa que el modelo explica el 75% de la variación en los costos médicos.
# MAGIC
# MAGIC En negocio: el modelo captura gran parte de las diferencias entre personas (edad, hábitos, etc.), pero aún queda un 25% de variación sin explicar.
# MAGIC
# MAGIC Esto implica que el modelo es útil y bastante predictivo, pero no perfecto: habrá casos donde los costos reales difieran bastante de lo estimado.
# MAGIC
# MAGIC - ¿Esperaban un número más alto o más bajo?
# MAGIC
# MAGIC En los modelos de negocio basados en información médica, un R² de 0.75 es generalmente visto como adecuado, dado que los gastos en salud son extremadamente variables y complicados de anticipar.
# MAGIC
# MAGIC Si contaban previamente con un modelo de menor exactitud, este nuevo valor podría ser superior a lo anticipado.
# MAGIC
# MAGIC Si la meta era alcanzar un nivel casi ideal (por ejemplo, 0.90), entonces podría parecer inferior a lo anhelado.
# MAGIC En términos generales, se considera un resultado bastante aceptable, aunque se persigue constantemente una mejora mediante la incorporación de más variables o el uso de técnicas más sofisticadas.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 7 — Ver dónde se equivoca el modelo

# COMMAND ----------

# Residual = lo real - lo que predijo el modelo
# Si el modelo es bueno, los errores deben ser aleatorios alrededor del cero
residuales = y_test - y_pred_lr

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ¿Los errores tienen algún patrón?
axes[0].scatter(y_pred_lr, residuales, alpha=0.4, color='#E8820C', s=20)
axes[0].axhline(0, color='#3B1F5E', linestyle='--', linewidth=2)
axes[0].set_title('¿Dónde se equivoca el modelo?')
axes[0].set_xlabel('Lo que predijo (USD)')
axes[0].set_ylabel('Error cometido (USD)')

# ¿Qué tan cerca están las predicciones de la realidad?
axes[1].scatter(y_test, y_pred_lr, alpha=0.4, color='#3B1F5E', s=20)
axes[1].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color='#E8820C', linestyle='--', linewidth=2, label='Predicción perfecta')
axes[1].set_title('Real vs Predicho')
axes[1].set_xlabel('Costo real (USD)')
axes[1].set_ylabel('Costo predicho (USD)')
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observen el gráfico izquierdo:**
# MAGIC - ¿Los puntos están distribuidos aleatoriamente alrededor del cero?
# MAGIC  Los puntos no parecen estar distribuidos aleatoriamente alrededor del cero porque se observa como un patrón en forma de abanico que se abre a la derecha. 
# MAGIC Cuando los gastos estimados son bajos o medianos, los fallos permanecen relativamente reducidos y próximos a cero. 
# MAGIC Esto significa que a media los costos previchos aumentan, los errores son más grandes y dispersos. 
# MAGIC - ¿O forman un abanico que se abre hacia la derecha?
# MAGIC   - Si es así → el modelo comete errores más grandes con los costos más altos (típico en fumadores) 
# MAGIC   
# MAGIC   En realidad, configuran un espectro que se despliega hacia la derecha: a medida que se incrementan los gastos estimados, los desaciertos suelen ser también más significativos.
# MAGIC
# MAGIC Esto sugiere que el modelo se equivoca más con los costos altos, lo cual es típico en fumadores, porque sus gastos médicos pueden variar mucho y tener valores extremos.
# MAGIC
# MAGIC **Observen el gráfico derecho:**
# MAGIC - Los puntos pegados a la línea naranja = predicciones correctas
# MAGIC - Los puntos alejados = casos donde el modelo falló más  
# MAGIC - ¿En qué rango de precios falla más?
# MAGIC Se puede notar que el modelo presenta un mayor número de fallos en el segmento de precios altos (superior a aproximadamente 40,000 USD). Esto concuerda con lo que se evidenció en el gráfico de la izquierda: las grandes discrepancias se agrupan en los costos elevados, que a menudo están vinculados a pacientes fumadores que padecen enfermedades graves o que requieren tratamientos de alto costo.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 8 — ¿Podemos mejorar con Ridge?
# MAGIC
# MAGIC Agregamos **PolynomialFeatures(degree=2)** para que el modelo capture relaciones más complejas (por ejemplo, que el efecto de fumar sea diferente según la edad).  
# MAGIC **Ridge** evita que el modelo se sobreajuste al tener tantas variables nuevas.

# COMMAND ----------

pipe_ridge = Pipeline(steps=[
    ('prep',  preprocessor),
    ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0)),
])

pipe_ridge.fit(X_train, y_train)
y_pred_ridge = pipe_ridge.predict(X_test)

mae_ridge  = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge   = r2_score(y_test, y_pred_ridge)

print('RIDGE (degree=2):')
print(f'  MAE:  ${mae_ridge:,.0f}')
print(f'  RMSE: ${rmse_ridge:,.0f}')
print(f'  R²:   {r2_ridge:.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Paso 9 — Comparación final

# COMMAND ----------

comparacion = pd.DataFrame({
    'Modelo':     ['Regresión Lineal', 'Ridge (degree=2)'],
    'MAE (USD)':  [round(mae_lr, 0),   round(mae_ridge, 0)],
    'RMSE (USD)': [round(rmse_lr, 0),  round(rmse_ridge, 0)],
    'R²':         [round(r2_lr, 3),    round(r2_ridge, 3)],
})
print(comparacion.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC **Para la presentación grupal, deben responder con los números reales:**
# MAGIC
# MAGIC 1. ¿El MAE de la regresión lineal es aceptable para una aseguradora?
# MAGIC El MAE de $2,434 indica que, de forma promedio, el modelo presenta un desfase de esa cifra en los gastos médicos por individuo.
# MAGIC
# MAGIC Al contrastarlo con una prima anual promedio de $4,200, el margen de error constituye más de la mitad de lo que se recibe del cliente durante un año.
# MAGIC
# MAGIC Para las compañías de seguros, este grado de error puede ser perjudicial, ya que influye en la habilidad de establecer primas adecuadas y de preservar la rentabilidad. No sería visto como aceptable sin ajustes o segmentación adicional.
# MAGIC
# MAGIC 2. ¿Ridge mejoró el R²? ¿Cuánto?
# MAGIC Regresión lineal: R² = 0.962
# MAGIC
# MAGIC Ridge (degree=2): R² = 0.986
# MAGIC
# MAGIC La mejora es de 0.024 puntos (2.4%).
# MAGIC  Aunque parece pequeña, en términos de negocio es significativa porque su modelo explica mejor la variación en los costos y reduce la incertidumbre.
# MAGIC
# MAGIC 3. ¿Qué muestra el gráfico de residuales sobre dónde falla el modelo?
# MAGIC Los residuales muestran un abanico hacia la derecha: los errores son más grandes en los costos altos.
# MAGIC
# MAGIC Esto confirma que el modelo falla más en los casos extremos, típicamente fumadores con gastos médicos elevados.
# MAGIC
# MAGIC  En rangos bajos y medios el modelo predice bien, pero en los altos se amplifican los errores.
# MAGIC
# MAGIC 4. Vuelvan a su hipótesis del inicio: ¿los datos la confirmaron?
# MAGIC
# MAGIC La hipótesis inicial era que “smoker” sería una variable clave y que los fumadores tendrían costos mucho más altos.
# MAGIC
# MAGIC Los datos lo confirmaron: el promedio de fumadores es $52,405 frente a $11,694 en no fumadores, una diferencia de $40,711.
# MAGIC
# MAGIC Esto valida la hipótesis: fumar es efectivamente una de las variables más importantes para explicar los costos médicos.
# MAGIC ---
# MAGIC *Próximo módulo: clasificación binaria con el dataset Breast Cancer.*