# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 12 — esta actividad: Predicción de Demanda en Superstore
# MAGIC ### Diagnóstico y Predictibilidad (92-0030) · I Cuatrimestre 2026 · Prof. Robin Sequeira
# MAGIC
# MAGIC En este notebook resolvemos, en grupo, las **6 etapas de esta actividad**: pronosticar la demanda mensual
# MAGIC de Superstore por categoría de producto (Furniture, Office Supplies, Technology) usando **ARIMA**.
# MAGIC
# MAGIC **Antes de empezar:**
# MAGIC - Ejecuten todas las celdas en orden (Run All al final, antes de exportar).
# MAGIC - No necesitan subir ningún archivo: el dataset ya está en Databricks como tabla.
# MAGIC - Al final de cada actividad van a encontrar una interpretación en texto plano de lo que salió.
# MAGIC - El notebook cierra con las preguntas de negocio que deben responder con **sus propios números**.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 0 — Preparar el entorno
# MAGIC
# MAGIC Instalamos `statsmodels`, que es la librería que usamos para la prueba ADF, ACF/PACF y el modelo ARIMA.
# MAGIC Si Databricks muestra una advertencia pidiendo reiniciar el kernel, es normal: por eso corremos
# MAGIC `dbutils.library.restartPython()` justo después.

# COMMAND ----------

# MAGIC %pip install statsmodels --quiet

# COMMAND ----------

dbutils.library.restartPython()
# Esta celda reinicia el intérprete de Python para que la instalación de arriba quede activa.
# La advertencia que puede aparecer es informativa, no es un error.

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
plt.rcParams["figure.figsize"] = (12, 5)
print("Librerías cargadas correctamente.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1 — Cargar el dataset
# MAGIC
# MAGIC El dataset `superstoreDS` ya está disponible como tabla en el workspace de Databricks. Lo cargamos con
# MAGIC Spark y lo convertimos a Pandas, igual que hicimos con el dataset de Superstore en módulos anteriores.

# COMMAND ----------

df = spark.table("workspace.default.superstore_ds").toPandas()
print(f"Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 1 — Limpieza, tipos de datos y outliers
# MAGIC
# MAGIC Primero revisamos los tipos de datos y convertimos la fecha de orden a formato `datetime`, porque vamos
# MAGIC a trabajar con una serie de tiempo. Después recortamos los valores extremos de `Sales` usando los
# MAGIC percentiles 1 y 99: cualquier venta por debajo del percentil 1 se sube a ese valor, y cualquier venta
# MAGIC por encima del percentil 99 se baja a ese valor. Esto evita que una venta corporativa gigantesca
# MAGIC distorsione el promedio de un mes completo.

# COMMAND ----------

# Ajustar estos nombres si en su tabla las columnas vienen con otro formato (mayúsculas/espacios)
df.columns = [c.strip() for c in df.columns] 


# Convertir la fecha de orden a datetime
df["Order Date"] = pd.to_datetime(df["Order Date"])

# Outliers por percentiles p1 / p99 sobre Sales
p1 = df["Sales"].quantile(0.01)
p99 = df["Sales"].quantile(0.99)
df["Sales_clean"] = df["Sales"].clip(lower=p1, upper=p99)

print(f"Percentil 1 de Sales:  {p1:,.2f}")
print(f"Percentil 99 de Sales: {p99:,.2f}")
print(f"Ventas originales fuera de ese rango: {((df['Sales'] < p1) | (df['Sales'] > p99)).sum():,} de {len(df):,}") 

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** las ventas fuera del rango p1–p99 son transacciones inusualmente pequeñas o
# MAGIC inusualmente grandes (por ejemplo, pedidos corporativos masivos de un solo cliente). Al recortarlas
# MAGIC evitamos que un solo pedido gigantesco haga parecer que un mes fue excelente cuando en realidad fue
# MAGIC un mes normal con una excepción.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 2 — Resample mensual por categoría e interpolación
# MAGIC
# MAGIC Convertimos las transacciones individuales en una **serie mensual de ventas totales por categoría**.
# MAGIC Si algún mes queda sin datos para alguna categoría, lo completamos con interpolación (estimando un
# MAGIC valor a partir de los meses vecinos), porque los modelos de series de tiempo no aceptan huecos.

# COMMAND ----------

# Agrupamos por mes y categoría, sumando las ventas ya limpias
serie_mensual = (
    df.set_index("Order Date")
      .groupby("Category")["Sales_clean"]
      .resample("M")
      .sum()
      .unstack(level=0)
)
# Interpolamos meses vacíos (si los hay) en cada columna de categoría
serie_mensual = serie_mensual.interpolate(method="linear")

print("Rango de fechas:", serie_mensual.index.min().date(), "→", serie_mensual.index.max().date())
serie_mensual.tail()

# COMMAND ----------

serie_mensual.plot(title="Ventas mensuales por categoría — Superstore")
plt.ylabel("Ventas ($)")
plt.xlabel("Fecha")
plt.legend(title="Categoría")
plt.tight_layout()
plt.show()

categoria_top = "Technology" 
print(f"\nCategoría con mayor volumen de ventas acumulado: {categoria_top}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** con el resample pasamos de miles de filas de transacciones a una tabla ordenada
# MAGIC en el tiempo, con un solo total por mes y por categoría. Esa es la forma que necesita cualquier modelo
# MAGIC de series de tiempo, incluyendo ARIMA más adelante.

# COMMAND ----------

# MAGIC %md ¿ Por qué no podemos pronosticar directamente sobre las transacciones individuales? 
# MAGIC
# MAGIC Las transacciones individuales contienen mucho ruido y variabilidad, lo que dificulta identificar patrones claros de tendencia o estacionalidad. Los modelos de series de tiempo requieren datos agregados y ordenados en el tiempo (por ejemplo, ventas mensuales), para poder captar la evolución real del negocio y hacer pronósticos útiles. Si intentamos pronosticar sobre transacciones individuales, el modelo se vería afectado por fluctuaciones aleatorias y no podría generalizar ni anticipar el comportamiento futuro.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 3 — Medias móviles MA-3 y MA-12
# MAGIC
# MAGIC Trabajamos las medias móviles sobre la categoría de mayor volumen (`categoria_top`, calculada arriba).
# MAGIC **MA-3** promedia los últimos 3 meses y sirve para decisiones operativas de corto plazo. **MA-12**
# MAGIC promedia los últimos 12 meses y suaviza la estacionalidad, mostrando la tendencia real del negocio.

# COMMAND ----------

categoria_top = 'Technology'  # Replace with the actual category name
serie_top = serie_mensual[categoria_top]


ma3 = serie_top.rolling(window=3, center=True).mean()
ma12 = serie_top.rolling(window=12, center=True).mean()

plt.plot(serie_top, label="Ventas mensuales", alpha=0.4)
plt.plot(ma3, label="MA-3", linewidth=2)
plt.plot(ma12, label="MA-12", linewidth=2)
plt.title(f"Medias móviles — {categoria_top}")
plt.ylabel("Ventas ($)")
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** si MA-3 se mueve claramente por encima o por debajo de MA-12 en los últimos
# MAGIC meses, eso indica que la categoría está acelerando o desacelerando respecto a su tendencia de largo
# MAGIC plazo. MA-3 conviene para el pedido del próximo mes; MA-12 conviene para planificación anual.

# COMMAND ----------

# MAGIC %md Para decidir el pedido del próximo mes, ¿usarían MA-3 o MA-12? ¿Por qué? 
# MAGIC
# MAGIC Para el pedido del próximo mes, se usaría el  **MA-3** porque refleja mejor las variaciones recientes en la demanda. MA-3 promedia los últimos 3 meses, lo que permite ajustar el pedido rápidamente si hay cambios o tendencias de corto plazo. En cambio, MA-12 suaviza demasiado y responde más lento a cambios, por lo que es más útil para planificación anual que para decisiones operativas inmediatas.

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 4 — Descomposición multiplicativa
# MAGIC
# MAGIC Separamos la serie de la categoría de mayor volumen en sus tres componentes: **tendencia**,
# MAGIC **estacionalidad** y **residuo**, usando `period=12` porque trabajamos con datos mensuales y buscamos
# MAGIC un patrón que se repite cada año.

# COMMAND ----------

descomp = seasonal_decompose(serie_top.dropna(), model="multiplicative", period=12)

fig = descomp.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.show()

factor_max = descomp.seasonal.idxmax()
valor_max = descomp.seasonal.max()
print(f"Mes con el factor estacional más alto: {factor_max.strftime('%B %Y')}  (factor = {valor_max:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** un factor estacional de, por ejemplo, 1.3 significa que ese mes vende en
# MAGIC promedio 30% más que un mes típico del año. Un factor de 0.85 significa que vende 15% menos. Esto es
# MAGIC clave para anticipar los meses de mayor y menor demanda.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 5 — Selección de parámetros ARIMA (ADF, ACF, PACF, tabla AIC)
# MAGIC
# MAGIC ### Paso 5.1 — Prueba ADF de estacionariedad
# MAGIC
# MAGIC La hipótesis nula (H0) de la prueba ADF dice que la serie **no** es estacionaria. Si el p-value es
# MAGIC menor a 0.05, rechazamos H0 y concluimos que la serie **sí** es estacionaria.

# COMMAND ----------

resultado_adf = adfuller(serie_top.dropna())
print(f"Estadístico ADF: {resultado_adf[0]:.4f}")
print(f"p-value:         {resultado_adf[1]:.4f}")

if resultado_adf[1] < 0.05:
    print("\n→ p-value < 0.05: la serie ES estacionaria (d = 0 podría ser suficiente).")
    d_sugerido = 0
else:
    print("\n→ p-value >= 0.05: la serie NO es estacionaria, hay que diferenciarla (d = 1).")
    d_sugerido = 1

print(f"\nParámetro d sugerido para empezar la búsqueda: {d_sugerido}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 5.2 — Gráficos ACF y PACF
# MAGIC
# MAGIC El **ACF** ayuda a intuir el parámetro `q` (parte de medias móviles). El **PACF** ayuda a intuir el
# MAGIC parámetro `p` (parte autorregresiva). Observen si hay picos importantes más allá del rezago 2: eso
# MAGIC suele indicar estacionalidad, que un ARIMA simple no captura del todo bien.

# COMMAND ----------

serie_dif = serie_top.diff(d_sugerido).dropna() if d_sugerido > 0 else serie_top.dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(serie_dif, ax=axes[0], lags=15)
axes[0].set_title("ACF")
plot_pacf(serie_dif, ax=axes[1], lags=15, method="ywm")
axes[1].set_title("PACF")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md Si ven picos después del rezago 2, ¿qué podría estar pasando con la serie?  
# MAGIC
# MAGIC Esto suele indicar que la serie tiene patrones de estacionalidad o ciclos de demanda que no son uniformes: algunos rezagos muestran mayor crecimiento (picos altos), mientras que otros tienen menor crecimiento o incluso caídas. Estos picos disparejos reflejan que la demanda varía de forma diferente según el mes o el periodo, y que la serie no sigue un comportamiento regular. Por eso, es importante analizar cada categoría por separado y considerar modelos que capturen la estacionalidad, ya que un ARIMA simple podría no ser suficiente si hay patrones complejos o irregulares.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 5.3 — Tabla AIC para seleccionar (p, d, q)
# MAGIC
# MAGIC Probamos varias combinaciones de `p` y `q` entre 0 y 2 (capamos en 2 porque tenemos pocos meses de
# MAGIC datos y órdenes más altos tienden a sobreajustar). Nos quedamos con la combinación de **menor AIC**.

# COMMAND ----------

resultados_aic = []

for p in range(0, 3):
    for q in range(0, 3):
        try:
            modelo = ARIMA(serie_top.dropna(), order=(p, d_sugerido, q))
            ajuste = modelo.fit()
            resultados_aic.append({"p": p, "d": d_sugerido, "q": q, "AIC": ajuste.aic})
        except Exception:
            continue

tabla_aic = pd.DataFrame(resultados_aic).sort_values("AIC").reset_index(drop=True)
mejor = tabla_aic.iloc[0]
print(f"Mejor combinación: ARIMA({int(mejor.p)}, {int(mejor.d)}, {int(mejor.q)})  con AIC = {mejor.AIC:.2f}")
tabla_aic

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** la fila con el AIC más bajo es la combinación (p, d, q) que mejor equilibra
# MAGIC ajuste a los datos y simplicidad del modelo. Esa es la que usamos en la siguiente actividad para el
# MAGIC pronóstico.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Actividad 6 — Pronóstico ARIMA a 6 meses
# MAGIC
# MAGIC Ajustamos el modelo con la mejor combinación (p, d, q) encontrada arriba, generamos el pronóstico a
# MAGIC 6 meses con intervalo de confianza al 95%, y evaluamos el modelo con MAPE y RMSE sobre los últimos
# MAGIC 6 meses reales (usados como conjunto de prueba).

# COMMAND ----------

orden_final = (int(mejor.p), int(mejor.d), int(mejor.q))

# Separamos los últimos 6 meses como prueba, para poder calcular MAPE y RMSE
train = serie_top.dropna().iloc[:-6]
test = serie_top.dropna().iloc[-6:]

modelo_final = ARIMA(train, order=orden_final)
ajuste_final = modelo_final.fit()

pronostico_prueba = ajuste_final.get_forecast(steps=6)
pred_media = pronostico_prueba.predicted_mean
mape = mean_absolute_percentage_error(test, pred_media) * 100
rmse = np.sqrt(mean_squared_error(test, pred_media))

print(f"Modelo usado: ARIMA{orden_final}")
print(f"MAPE sobre los últimos 6 meses reales: {mape:.2f}%")
print(f"RMSE sobre los últimos 6 meses reales: {rmse:,.2f}")

print(f"\n{'='*60}")
print(f"INTERPRETACIÓN DEL MAPE:")
print(f"{'='*60}")
if mape < 10:
    print(f"✓ MAPE = {mape:.2f}% - Excelente precisión, modelo muy confiable")
elif mape < 15:
    print(f"✓ MAPE = {mape:.2f}% - Buena precisión, modelo aceptable para decisiones")
elif mape < 25:
    print(f"⚠ MAPE = {mape:.2f}% - Precisión moderada, usar con cautela")
else:
    print(f"✗ MAPE = {mape:.2f}% - Baja precisión, NO recomendado sin ajustes")

# COMMAND ----------

# MAGIC %md ¿Qué MAPE les parecería aceptable para una empresa que compra cada trimestre? 
# MAGIC
# MAGIC Un MAPE aceptable para una empresa de retail que planifica compras trimestrales suele estar por debajo del 10-15%. 
# MAGIC
# MAGIC En nuestro análisis, el modelo obtuvo un **MAPE de 25.21%** sobre los últimos 6 meses reales, lo cual **excede el umbral aceptable**. Si el MAPE está dentro de ese rango, el modelo es suficientemente preciso para tomar decisiones de compra; si es mayor, conviene revisar el modelo o complementar el pronóstico con criterio de negocio.
# MAGIC  
# MAGIC

# COMMAND ----------

# Ahora sí, el pronóstico real hacia adelante (6 meses después del último dato disponible)
modelo_completo = ARIMA(serie_top.dropna(), order=orden_final)
ajuste_completo = modelo_completo.fit()

pronostico = ajuste_completo.get_forecast(steps=6)
media_pronostico = pronostico.predicted_mean
intervalo = pronostico.conf_int(alpha=0.05)  # 95% de confianza

plt.plot(serie_top, label="Histórico")
plt.plot(media_pronostico.index, media_pronostico, label="Pronóstico", color="darkorange")
plt.fill_between(
    intervalo.index, intervalo.iloc[:, 0], intervalo.iloc[:, 1],
    color="darkorange", alpha=0.2, label="Intervalo 95%"
)
plt.title(f"Pronóstico a 6 meses — {categoria_top} — ARIMA{orden_final}")
plt.ylabel("Ventas ($)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nPronóstico de ventas para los próximos 6 meses ({categoria_top}):")
tabla_pronostico = pd.DataFrame({
    "Pronóstico": media_pronostico,
    "Límite inferior (95%)": intervalo.iloc[:, 0],
    "Límite superior (95%)": intervalo.iloc[:, 1],
})
tabla_pronostico

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** el intervalo de confianza al 95% significa que, si repitiéramos este análisis
# MAGIC muchas veces, esperaríamos que la venta real caiga dentro de ese rango en el 95% de los casos. Un MAPE
# MAGIC más bajo indica un modelo más confiable; un MAPE alto sugiere tomar el pronóstico con más cautela y
# MAGIC complementarlo con criterio de negocio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Repitiendo el análisis para las otras categorías
# MAGIC
# MAGIC El caso pide comparar el comportamiento entre las 3 categorías. Repetimos rápidamente la prueba ADF
# MAGIC y el AIC para las otras dos categorías, sin repetir todos los gráficos, para poder comparar.

# COMMAND ----------

resumen_categorias = []

for cat in serie_mensual.columns:
    s = serie_mensual[cat].dropna()
    p_adf = adfuller(s)[1]
    d_cat = 0 if p_adf < 0.05 else 1

    mejores = []
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                aj = ARIMA(s, order=(p, d_cat, q)).fit()
                mejores.append((p, d_cat, q, aj.aic))
            except Exception:
                continue
    mejor_cat = min(mejores, key=lambda x: x[3])

    resumen_categorias.append({
        "Categoría": cat,
        "p-value ADF": round(p_adf, 4),
        "Estacionaria (p<0.05)": p_adf < 0.05,
        "Mejor (p,d,q)": mejor_cat[:3],
        "AIC": round(mejor_cat[3], 2),
        "Ventas promedio mensual": round(s.mean(), 2),
    })

pd.DataFrame(resumen_categorias)

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** comparen el p-value de ADF y el orden ARIMA elegido entre categorías. Si son
# MAGIC distintos, eso confirma que **no se debe usar el mismo modelo para las 3 categorías sin verificar cada
# MAGIC una por separado**, tal como advertimos al inicio de la clase.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preguntas de análisis de negocio
# MAGIC
# MAGIC Respondan las siguientes preguntas usando los **números reales** que obtuvieron arriba en este
# MAGIC notebook. No se aceptan respuestas genéricas como "el modelo predice bien".
# MAGIC
# MAGIC **1. ¿Por qué las 3 categorías (Furniture, Office Supplies, Technology) tienen patrones de demanda
# MAGIC tan diferentes?**
# MAGIC _(Apóyense en la tabla de comparación de categorías: p-value de ADF, orden ARIMA elegido y factor
# MAGIC estacional de cada una.)_
# MAGIC
# MAGIC `Escriban aquí su respuesta con los números de su notebook.`
# MAGIC
# MAGIC Las 3 categorías muestran patrones de demanda diferentes porque ninguna es estacionaria según la prueba ADF: Furniture (p-value = 0.99), Office Supplies (p-value = 0.99) y Technology (p-value = 0.97), todas con p-value > 0.05. Esto indica que sus ventas varían significativamente a lo largo del tiempo y requieren diferenciación (d = 1) en el modelo ARIMA. 
# MAGIC
# MAGIC El mejor orden ARIMA para cada una fue: Furniture (2, 1, 2), Office Supplies (2, 1, 1) y Technology (2, 1, 2), lo que evidencia que cada categoría tiene ciclos y tendencias distintas y no se puede usar el mismo modelo para todas. Además, los factores estacionales y el volumen de ventas muestran diferencias en los meses de mayor demanda entre categorías.
# MAGIC
# MAGIC **2. ¿Cómo usarían el pronóstico a 6 meses para diseñar un sistema de alertas de inventario?**
# MAGIC _(Piensen en el límite inferior del intervalo de confianza: ¿qué pasaría si las ventas reales caen
# MAGIC por debajo de ese límite varios meses seguidos?)_
# MAGIC
# MAGIC `Escriban aquí su respuesta con los números de su notebook.
# MAGIC
# MAGIC Si las ventas reales caen por debajo del límite inferior del intervalo de confianza (por ejemplo, menos de $116,646.74 en enero, $106,422.70 en febrero, y así sucesivamente), durante varios meses seguidos (como en los valores de 2016-01-31 a 2016-06-30), esto puede indicar una señal de alerta temprana para el sistema de inventario. 
# MAGIC
# MAGIC  Indica que la demanda está siendo mucho menor de lo esperado, lo que puede llevar a exceso de inventario, costos de almacenamiento y posibles pérdidas por productos obsoletos. El sistema de alertas debería notificar a los responsables para revisar las causas (cambios de mercado, competencia, factores externos) y ajustar las compras o promociones. Si la tendencia persiste, conviene reducir pedidos futuros y analizar estrategias para estimular la demanda o liquidar inventario.
# MAGIC
# MAGIC
# MAGIC
# MAGIC **3. ¿Qué MAPE considerarían aceptable para una empresa de retail que planifica compras
# MAGIC trimestrales?**
# MAGIC _(Comparen con el MAPE que obtuvieron en la Actividad 6 y justifiquen si les parece un modelo
# MAGIC confiable para tomar decisiones de compra.)_
# MAGIC
# MAGIC **Análisis del MAPE obtenido:**
# MAGIC
# MAGIC Según la literatura y las prácticas de la industria, un **MAPE aceptable para retail que planifica compras trimestrales** debe estar **por debajo del 10-15%**. Este rango permite:
# MAGIC * Optimizar niveles de inventario sin excesos ni faltantes
# MAGIC * Reducir costos de almacenamiento y obsolescencia
# MAGIC * Mantener niveles de servicio adecuados (>95%)
# MAGIC
# MAGIC **Nuestros resultados (Actividad 6):**
# MAGIC * Modelo: ARIMA(2, 1, 2) para Technology
# MAGIC * **MAPE obtenido: 25.21%**
# MAGIC * RMSE: $16,428.37
# MAGIC
# MAGIC **Evaluación del modelo:**
# MAGIC
# MAGIC ❌** **MAPE*** = 25.21% > 15% → NO es aceptable para tomar decisiones de compra sin ajustes.
# MAGIC
# MAGIC Este error significa que, en promedio, nuestro pronóstico se desvía un 25% de las ventas reales. Para una empresa que compra cada trimestre, esto implica:
# MAGIC
# MAGIC 1. **Riesgo de sobrestock:** Si sobrestimamos en 25%, compramos $25,000 de más por cada $100,000 proyectados
# MAGIC 2. **Riesgo de faltantes:** Si subestimamos en 25%, perdemos ventas y clientes
# MAGIC 3. **Impacto financiero:** Con ventas promedio mensuales de $92,416 (Technology), un error del 25% representa ±$23,104/mes o ±$69,312/trimestre
# MAGIC
# MAGIC **Recomendaciones:**
# MAGIC * ⚠ **NO usar este modelo como única base para decisiones de compra**
# MAGIC * Complementar con criterio de expertos de negocio y análisis de tendencias del mercado
# MAGIC * Revisar el modelo: probar ARIMA para capturar mejor la estacionalidad, incluir variables exógenas (días festivos, campañas de marketing), aumentar el historial de datos si es posible.
# MAGIC
# MAGIC Recuerden responder las 3 preguntas restantes del documento **la guía de la actividad** publicado en
# MAGIC Blackboard, también con datos reales de su análisis.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre y checklist antes de exportar
# MAGIC
# MAGIC - [ ] Las 6 actividades técnicas están completas y ejecutadas.
# MAGIC - [ ] La tabla AIC muestra al menos 4 combinaciones de (p, q) comparadas.
# MAGIC - [ ] El gráfico de pronóstico incluye el intervalo de confianza al 95%.
# MAGIC - [ ] Las 6 preguntas de negocio (3 en este notebook + 3 en el documento del caso) están respondidas
# MAGIC       con valores numéricos reales, no genéricos.
# MAGIC - [ ] Ejecutaron **Run All** antes de exportar.
# MAGIC - [ ] Exportaron como HTML: **File → Export → HTML**.
# MAGIC - [ ] El archivo se llama `Apellido1_Apellido2_Superstore.html` y está subido en Blackboard.
# MAGIC
# MAGIC **La próxima semana (Módulo 13)** cambiamos el enfoque: en lugar de predecir qué va a pasar, vamos a
# MAGIC detectar qué está fuera de lo normal en los datos del negocio.
# MAGIC