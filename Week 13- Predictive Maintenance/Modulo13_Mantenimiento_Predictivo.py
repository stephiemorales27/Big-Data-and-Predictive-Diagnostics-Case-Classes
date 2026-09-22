# Databricks notebook source
# MAGIC %md
# MAGIC # Módulo 13 — Detección de Anomalías y Mantenimiento Predictivo Industrial
# MAGIC
# MAGIC **Curso:** Diagnóstico y Predictibilidad (92-0030)
# MAGIC **Profesor:** Robin Sequeira
# MAGIC **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)
# MAGIC
# MAGIC ## De qué se trata esta clase
# MAGIC
# MAGIC La semana pasada usamos validación cruzada e interpretabilidad para mejorar los modelos del proyecto. Hoy cambiamos completamente de pregunta: en vez de predecir qué va a pasar, vamos a **detectar qué está pasando ahora mismo que no es normal**.
# MAGIC
# MAGIC Vamos a trabajar con datos reales de sensores de una máquina CNC industrial. Vamos a entrenar dos modelos que nunca vieron un ejemplo de falla durante su entrenamiento, y aun así van a ser capaces de detectarlas. Al final, vamos a construir un sistema de alertas de 3 niveles, tal como lo usaría una planta industrial real.
# MAGIC
# MAGIC **Lo que vamos a hacer, paso a paso:**
# MAGIC 1. Cargar y explorar el dataset AI4I 2020
# MAGIC 2. Entender los 5 sensores y los tipos de falla
# MAGIC 3. Entrenar Isolation Forest
# MAGIC 4. Entrenar One-Class SVM
# MAGIC 5. Evaluar ambos modelos contra las fallas reales
# MAGIC 6. Construir el sistema de alertas de 3 niveles
# MAGIC 7. Visualizar el score en el tiempo
# MAGIC 8. Actividad de grupo en breakout rooms
# MAGIC 9. Preguntas del Portafolio II
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 1: Preparar el entorno
# MAGIC
# MAGIC Ejecuta esta celda primero. En Databricks Community, si falta alguna librería, la instalamos con `%pip install` y reiniciamos Python. **El mensaje de reinicio que aparece es informativo, no un error** — simplemente continúa con la siguiente celda.

# COMMAND ----------

# Si alguna librería falta en el cluster, se instala aquí.
# En Databricks Community Edition esto es normal y esperado.
%pip install scikit-learn --quiet
dbutils.library.restartPython()

# COMMAND ----------

# Librerías que vamos a usar durante toda la clase
import pandas as pd
import numpy as np
import io

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# Estilo visual consistente para todos los gráficos de la clase
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

print("Librerías cargadas correctamente. Listos para empezar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 2: Cargar el dataset AI4I 2020
# MAGIC
# MAGIC Vamos a intentar descargar el dataset real directamente del repositorio de UCI Machine Learning. Si por alguna razón no hay conexión a internet en este cluster (puede pasar en Databricks Community), el notebook genera automáticamente una versión sintética que sigue **exactamente las mismas reglas matemáticas** con las que se construyó el dataset original (publicadas por S. Matzka, 2020). Ninguno de los dos caminos requiere que subas ningún archivo.

# COMMAND ----------

# Intento 1: descargar el dataset real desde UCI
# Si no hay conexión, generamos una versión sintética con las mismas reglas
# que usó el autor original del dataset (Matzka, 2020) para construirlo.

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

try:
    df = pd.read_csv(url)
    fuente = "UCI Machine Learning Repository (dataset real)"
except Exception as e:
    print(f"No se pudo descargar desde UCI ({e}). Generando dataset sintético equivalente...")

    rng = np.random.default_rng(42)
    n = 10000

    # Variante de producto: L (50%), M (30%), H (20%), igual que el dataset original
    variantes = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    type_ids = {"L": 0, "M": 0, "H": 0}
    product_id = []
    for v in variantes:
        type_ids[v] += 1
        product_id.append(f"{v}{type_ids[v]:05d}")

    # Temperatura del aire: random walk normalizado alrededor de 300 K, sd=2K
    air_temp = 300 + np.cumsum(rng.normal(0, 0.5, n))
    air_temp = 300 + (air_temp - air_temp.mean()) / air_temp.std() * 2

    # Temperatura de proceso: random walk + 10K sobre la temperatura del aire, sd=1K
    proc_noise = np.cumsum(rng.normal(0, 0.3, n))
    proc_temp = air_temp + 10 + (proc_noise - proc_noise.mean()) / proc_noise.std() * 1

    # Torque: distribución normal alrededor de 40 Nm, sd=10, sin negativos
    torque = np.clip(rng.normal(40, 10, n), 5, None)

    # Velocidad rotacional: se deriva de una potencia nominal ~6200W con ruido,
    # igual que en el dataset original (la potencia del proceso es aprox. constante,
    # y la velocidad varía en función inversa del torque)
    power_nominal = np.clip(rng.normal(6200, 1050, n), 1000, None)
    rot_speed = (power_nominal / (torque * (2 * np.pi / 60))).astype(int)

    # Desgaste de herramienta: depende de la variante (L/M/H tienen distinto tiempo de uso)
    wear_add = {"L": 0, "M": 3, "H": 5}
    tool_wear = np.clip(
        rng.integers(0, 200, n) + np.array([wear_add[v] for v in variantes]), 0, 253
    )

    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": product_id,
        "Type": variantes,
        "Air temperature [K]": air_temp.round(1),
        "Process temperature [K]": proc_temp.round(1),
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": tool_wear,
    })

    # Reglas de falla EXACTAS del paper original (Matzka, 2020):

    # TWF: falla por desgaste de herramienta, entre 200 y 240 min de desgaste
    twf_zone = (df["Tool wear [min]"] >= 200) & (df["Tool wear [min]"] <= 240)
    twf = twf_zone & (rng.random(n) < 0.55)

    # HDF: falla por disipación de calor, si diff temp < 8.6K y velocidad < 1380 rpm
    diff_temp = df["Process temperature [K]"] - df["Air temperature [K]"]
    hdf = (diff_temp < 8.6) & (df["Rotational speed [rpm]"] < 1380)

    # PWF: falla por potencia, si potencia (torque * velocidad angular) < 3500W o > 9000W
    power = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    pwf = (power < 3500) | (power > 9000)

    # OSF: falla por sobrecarga, si tool_wear * torque excede el umbral según variante
    osf_threshold = df["Type"].map({"L": 10500, "M": 11500, "H": 12500})
    osf = (df["Tool wear [min]"] * df["Torque [Nm]"]) > osf_threshold

    # RNF: falla aleatoria, 0.1% de probabilidad sin importar los parámetros
    rnf = rng.random(n) < 0.001

    df["TWF"] = twf.astype(int)
    df["HDF"] = hdf.astype(int)
    df["PWF"] = pwf.astype(int)
    df["OSF"] = osf.astype(int)
    df["RNF"] = rnf.astype(int)
    df["Machine failure"] = ((twf | hdf | pwf | osf | rnf)).astype(int)

    fuente = "Dataset sintético generado con las reglas del paper original (Matzka, 2020)"

print(f"Dataset cargado. Fuente: {fuente}")
print(f"Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** cada fila es una lectura instantánea de los cinco sensores de la máquina, junto con la variante de producto que se estaba fabricando en ese momento (L = baja calidad, M = media, H = alta). La columna `Machine failure` es la etiqueta que ya conocemos: nos dice si esa lectura terminó en una falla real o no. Ese es nuestro "examen final" para evaluar qué tan bien detectan las anomalías nuestros modelos, aunque ellos nunca la vean durante el entrenamiento.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 3: Explorar el dataset antes de modelar
# MAGIC
# MAGIC Antes de entrenar cualquier modelo, siempre hay que entender los datos. Vamos a ver cuántas fallas reales hay, y de qué tipo.

# COMMAND ----------

# ¿Cuántas fallas reales hay en total, y de qué tipo?
total_fallas = df["Machine failure"].sum()
porcentaje_fallas = total_fallas / len(df) * 100

print(f"Total de lecturas: {len(df):,}")
print(f"Total de fallas reales (Machine failure = 1): {total_fallas:,} ({porcentaje_fallas:.2f}% del total)")
print()
print("Desglose por tipo de falla:")
for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    print(f"  {col}: {df[col].sum()} casos")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación de negocio:** este es exactamente el tipo de situación donde el aprendizaje no supervisado tiene sentido. Las fallas representan una fracción muy pequeña del total de lecturas. Con tan pocos ejemplos, un modelo supervisado tradicional tendría muy poco de dónde aprender el patrón de cada tipo de falla. Por eso vamos a enseñarle al modelo solamente cómo se ve lo normal, y dejar que él mismo detecte lo que se aleja de eso.

# COMMAND ----------

# Distribución de los 5 sensores
sensores = ["Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, sensores):
    sns.histplot(df[col], ax=ax, color="#3B1F5E", kde=True)
    ax.set_title(col, fontsize=10)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** cada sensor tiene su propio rango normal de operación. La temperatura y la velocidad rotacional se comportan de forma bastante simétrica, mientras que el desgaste de herramienta tiende a acumularse con el tiempo de uso. Estas cinco variables son las que van a "ver" nuestros modelos de detección de anomalías: no reciben ninguna etiqueta de falla durante el entrenamiento, solamente estos cinco números por cada lectura.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 4: Preparar los datos para el modelo
# MAGIC
# MAGIC Usamos únicamente los 5 sensores numéricos. One-Class SVM necesita las variables escaladas (misma unidad de comparación); Isolation Forest no lo necesita porque trabaja con cortes, no con distancias, pero escalamos ambos igual para poder comparar resultados de forma justa.

# COMMAND ----------

X = df[sensores].copy()
y_real = df["Machine failure"]  # Solo la usamos para EVALUAR, nunca para entrenar

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Matriz de features lista: {X_scaled.shape[0]:,} filas x {X_scaled.shape[1]} sensores")
print("Nota: y_real (Machine failure) se guarda aparte. Los modelos de hoy NO la usan para entrenar.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 5: Entrenar Isolation Forest
# MAGIC
# MAGIC Le decimos al modelo, más o menos, qué porcentaje de los datos esperamos que sea raro (`contamination`). Usamos 5%, cerca del porcentaje real de fallas. En una planta real no sabríamos ese número exacto de antemano, y se ajustaría con la experiencia del equipo de mantenimiento.

# COMMAND ----------

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,     # esperamos que ~5% de las lecturas sean anómalas
    random_state=42
)
iso_forest.fit(X_scaled)

# decision_function: el anomaly score. RECUERDEN: más NEGATIVO = más anómalo.
df["score_isoforest"] = iso_forest.decision_function(X_scaled)
df["pred_isoforest"] = iso_forest.predict(X_scaled)  # -1 = anomalía, 1 = normal

print("Isolation Forest entrenado.")
print()
print("Las 5 lecturas MÁS anómalas según el score (más negativas primero):")
print(df.sort_values("score_isoforest").head(5)[["UDI", "score_isoforest", "Machine failure"] + sensores])
print()
print("Las 5 lecturas MÁS normales según el score (más positivas primero):")
print(df.sort_values("score_isoforest", ascending=False).head(5)[["UDI", "score_isoforest", "Machine failure"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación (el detalle que confunde a todos):** miren la primera tabla. Son las lecturas más raras según el modelo, y su número (score) es el más **negativo**. La segunda tabla tiene el número más alto, y son las lecturas más comunes. Sí, es al revés de lo que uno esperaría: entre más bajo el número, más raro es el dato. Así lo definió quien programó la librería, y ya. Solo hay que recordarlo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 6: Entrenar One-Class SVM
# MAGIC
# MAGIC Este modelo dibuja la cerca alrededor de lo normal. Le decimos que esperamos un 5% de datos raros, el mismo porcentaje que usamos con Isolation Forest, para poder comparar los dos de forma justa. **Nota:** este paso puede tardar uno o dos minutos — es normal, este modelo es más lento que Isolation Forest.

# COMMAND ----------

oc_svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
oc_svm.fit(X_scaled)

df["score_ocsvm"] = oc_svm.decision_function(X_scaled)
df["pred_ocsvm"] = oc_svm.predict(X_scaled)  # -1 = anomalía, 1 = normal

print("One-Class SVM entrenado.")
print()
print(f"Lecturas marcadas como anómalas por OC-SVM: {(df['pred_ocsvm'] == -1).sum()} de {len(df):,}")
print(f"Lecturas marcadas como anómalas por Isolation Forest: {(df['pred_isoforest'] == -1).sum()} de {len(df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** los dos modelos marcan más o menos el mismo número de anomalías, porque les pedimos el mismo 5% a ambos. Pero probablemente no son exactamente las mismas lecturas: cada modelo "sospecha" de datos un poco distintos, porque llegan a la respuesta de forma diferente (uno aislando, el otro con una cerca).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 7: Evaluar ambos modelos contra las fallas reales
# MAGIC
# MAGIC Aquí es donde usamos la columna `Machine failure` que habíamos guardado aparte. Convertimos las predicciones de -1/1 a 1/0 (1 = anomalía) para poder comparar directamente contra la etiqueta real.

# COMMAND ----------

def evaluar_modelo(nombre, y_true, y_pred_raw, score):
    # Convertimos -1/1 (convención sklearn) a 1/0 (1 = anomalía, para comparar con Machine failure)
    y_pred = (y_pred_raw == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Para ROC-AUC usamos el score invertido: valores más negativos = más "positivos" (anómalos)
    auc = roc_auc_score(y_true, -score)

    print(f"=== {nombre} ===")
    print(f"Precision: {precision:.3f}  |  Recall: {recall:.3f}  |  F1-Score: {f1:.3f}  |  ROC-AUC: {auc:.3f}")
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["Real: Normal", "Real: Falla"],
        columns=["Pred: Normal", "Pred: Anomalía"]
    ))
    print()
    return {"modelo": nombre, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

resultados = []
resultados.append(evaluar_modelo("Isolation Forest", y_real, df["pred_isoforest"], df["score_isoforest"]))
resultados.append(evaluar_modelo("One-Class SVM", y_real, df["pred_ocsvm"], df["score_ocsvm"]))

tabla_resultados = pd.DataFrame(resultados)
tabla_resultados

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación de negocio:** comparen el F1-Score y el recall de ambos modelos con los números reales que acaban de obtener arriba. Recuerden la discusión de la clase: en mantenimiento industrial, perder una falla real (recall bajo) suele costar mucho más que revisar una máquina que en realidad estaba bien (precision más baja). Si tuvieran que elegir un modelo para producción, ¿cuál elegirían con base en estos números, y por qué?

# COMMAND ----------

# MAGIC %md 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 8: Construir el semáforo de 3 niveles
# MAGIC
# MAGIC Con el score de Isolation Forest armamos un semáforo, usando el propio historial de la máquina: **Normal** (verde), **Precaución** (amarillo) y **Crítico** (rojo).

# COMMAND ----------

p75 = df["score_isoforest"].quantile(0.25)   # percentil 75 "de anomalía" = percentil 25 del score (score bajo = más anómalo)
p90 = df["score_isoforest"].quantile(0.10)   # percentil 90 "de anomalía" = percentil 10 del score

def nivel_alerta(score):
    if score <= p90:
        return "Crítico"
    elif score <= p75:
        return "Precaución"
    else:
        return "Normal"

df["nivel_alerta"] = df["score_isoforest"].apply(nivel_alerta)

conteo_niveles = df["nivel_alerta"].value_counts()
print("Distribución de lecturas por nivel de alerta:")
print(conteo_niveles)
print()

# La pregunta de portafolio: ¿qué % de las fallas reales cayó en nivel Crítico?
fallas_criticas = df[(df["Machine failure"] == 1) & (df["nivel_alerta"] == "Crítico")]
pct_fallas_criticas = len(fallas_criticas) / total_fallas * 100

print(f"De las {total_fallas} fallas reales, {len(fallas_criticas)} cayeron en nivel Crítico")
print(f"Eso es el {pct_fallas_criticas:.1f}% de todas las fallas reales del dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** este es el número que necesitan para la Pregunta de Portafolio 2. Anoten el porcentaje que les salió a ustedes (va a variar un poco entre ejecuciones). Si es alto, el semáforo está atrapando la mayoría de las fallas reales en rojo. Si es bajo, muchas fallas se están escapando en amarillo o verde, y valdría la pena hacer el rojo menos estricto.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 9: Visualizar el score a lo largo del tiempo
# MAGIC
# MAGIC Graficamos el score de Isolation Forest en el orden en que aparecen las lecturas (usamos el índice `UDI` como proxy de tiempo), y marcamos con una línea horizontal los umbrales de Precaución y Crítico. Los puntos rojos son fallas reales.

# COMMAND ----------

plt.figure(figsize=(16, 6))
plt.plot(df["UDI"], df["score_isoforest"], color="#3B1F5E", linewidth=0.6, label="Anomaly score")

plt.axhline(p75, color="#E8820C", linestyle="--", linewidth=1.5, label=f"Umbral Precaución (p75 = {p75:.3f})")
plt.axhline(p90, color="red", linestyle="--", linewidth=1.5, label=f"Umbral Crítico (p90 = {p90:.3f})")

fallas_reales = df[df["Machine failure"] == 1]
plt.scatter(fallas_reales["UDI"], fallas_reales["score_isoforest"],
            color="red", s=25, zorder=5, label="Falla real (Machine failure = 1)")

plt.xlabel("Lectura (orden de ocurrencia, UDI)")
plt.ylabel("Anomaly score (Isolation Forest)")
plt.title("Score de anomalía en el tiempo, con fallas reales marcadas")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretación:** observen dónde caen los puntos rojos (fallas reales) respecto a las líneas de umbral. Si la mayoría de los puntos rojos está por debajo de la línea roja punteada (umbral Crítico), el modelo está identificando correctamente las fallas reales como críticas. Si además notan que el score empieza a bajar *unas lecturas antes* de que ocurra la falla marcada, eso es evidencia de que el modelo **anticipa** la falla, no solo la confirma. Esa observación es justamente lo que necesitan reportar en la Pregunta de Portafolio 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 10: Actividad en grupos (breakout rooms)
# MAGIC
# MAGIC Van a trabajar en su sala de Teams asignada. **Cada sala prueba un umbral de "nivel Crítico" distinto**, para ver cómo cambia el porcentaje de fallas capturadas. Solo cambien el número `NUMERO_DE_SALA` de la celda de abajo (del 1 al 10) según la sala que les fue asignada, y corran todas las celdas desde ahí hacia abajo.
# MAGIC
# MAGIC | Sala | Percentil probado para "Crítico" |
# MAGIC |---|---|
# MAGIC | 1 | 80 |
# MAGIC | 2 | 82 |
# MAGIC | 3 | 84 |
# MAGIC | 4 | 86 |
# MAGIC | 5 | 88 |
# MAGIC | 6 | 90 (el que usamos en clase) |
# MAGIC | 7 | 92 |
# MAGIC | 8 | 94 |
# MAGIC | 9 | 96 |
# MAGIC | 10 | 98 |

# COMMAND ----------

# CAMBIEN SOLO ESTE NÚMERO SEGÚN SU SALA (1 a 10). Luego corran esta celda y la siguiente.
NUMERO_DE_SALA = 3

percentiles_por_sala = {1: 80, 2: 82, 3: 84, 4: 86, 5: 88, 6: 90, 7: 92, 8: 94, 9: 96, 10: 98}
percentil_critico = percentiles_por_sala[NUMERO_DE_SALA]

umbral_critico_sala = df["score_isoforest"].quantile(1 - percentil_critico / 100)
fallas_criticas_sala = df[(df["Machine failure"] == 1) & (df["score_isoforest"] <= umbral_critico_sala)]
pct_capturado_sala = len(fallas_criticas_sala) / total_fallas * 100
pct_lecturas_marcadas = (df["score_isoforest"] <= umbral_critico_sala).mean() * 100

print(f"Sala {NUMERO_DE_SALA}  |  Percentil Crítico probado: {percentil_critico}")
print(f"Umbral de score: {umbral_critico_sala:.4f}")
print(f"% de fallas reales capturadas en nivel Crítico: {pct_capturado_sala:.1f}%")
print(f"% del total de lecturas que quedan marcadas como Crítico: {pct_lecturas_marcadas:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC **Discusión de grupo (5 minutos):** con su resultado, respondan en su sala:
# MAGIC 1. ¿Subir el percentil (por ejemplo de 90 a 98) capturó más o menos fallas reales?
# MAGIC
# MAGIC Con el percentil 90, se capturan la mayoría de las fallas reales, pero si subimos el percentil (por ejemplo, a 98), el sistema marca menos fallas como Crítico y puede dejar escapar algunas. Por lo tanto, el percentil 90 logra un buen equilibrio entre capturar fallas reales y no marcar demasiadas lecturas como críticas.
# MAGIC
# MAGIC 2. ¿Qué le pasó al porcentaje total de lecturas marcadas como Crítico?
# MAGIC
# MAGIC El porcentaje total de lecturas marcadas como Crítico fue de aproximadamente 16%. Esto significa que el sistema identifica como críticas solo una pequeña parte de todas las lecturas, enfocándose en los casos más anómalos y evitando alertas excesivas.
# MAGIC
# MAGIC 3. Si el mantenimiento correctivo cuesta 10 veces más que el preventivo, ¿qué percentil recomendarían a la planta?
# MAGIC
# MAGIC Recomendaría el percentil 84, porque permite capturar un mayor porcentaje de fallas reales en nivel Crítico, reduciendo el riesgo de perder fallas costosas. Así, el sistema prioriza la detección temprana de posibles fallas, aunque marque más lecturas como críticas, lo cual es preferible cuando el costo del mantenimiento correctivo es mucho mayor que el preventivo.
# MAGIC
# MAGIC Al final, cada sala comparte su número en el chat y armamos entre todos la tabla comparativa completa de las 10 salas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Paso 11: Preguntas del Portafolio II
# MAGIC
# MAGIC Estas dos preguntas se responden en equipo, con los **valores numéricos reales** que obtuvieron en este notebook (no valores genéricos), y cada una necesita una **captura de pantalla del gráfico correspondiente** como evidencia. Se entregan como parte del Portafolio II en el Módulo 14.
# MAGIC
# MAGIC **Pregunta 1 — Recomendación de modelo para Walmart:** vuelvan a su notebook del Módulo 10 (comparación de algoritmos de anomalías con datos tipo Walmart). Con base en el F1-Score que obtuvieron ahí, ¿qué modelo recomendarían para ese caso? Justifiquen con el número real, no de memoria. ¿Cambiaría su respuesta si el objetivo de negocio fuera maximizar el recall en vez del F1? 
# MAGIC *(Incluyan el número real de F1-Score obtenido en su notebook del Módulo 10 y justifiquen con ese valor. Si el objetivo cambia a recall, comparen los valores de recall de ambos modelos y expliquen su elección.)* 
# MAGIC
# MAGIC Con base en los resultados obtenidos en nuestro notebook del Módulo 10 (detección de anomalías con datos tipo Walmart), recomendaríamos DBSCAN  DBSCAN como el modelo más adecuado para esta situación.
# MAGIC
# MAGIC Justificación con datos concretos:
# MAGIC
# MAGIC DBSCAN registró el F1-Score más elevado: 0.882, superando de manera notable a los demás algoritmos:
# MAGIC
# MAGIC DBSCAN: F1 = 0.882, Precisión = 1.0, Recuperación = 0.789
# MAGIC
# MAGIC K-Means: F1 = 0.667, Precisión = 0.577, Recuperación = 0.789
# MAGIC
# MAGIC Isolation Forest: F1 = 0.622, Precisión = 0.538, Recuperación = 0.737
# MAGIC
# MAGIC One-Class SVM: F1 = 0.513, Precisión = 0.5, Recuperación = 0.526
# MAGIC
# MAGIC DBSCAN consiguió el mejor balance entre Precisión y Recuperación. Su Precisión perfecta (1.0) indica que en cada ocasión que identificó una semana como anómala, fue 100% preciso, sin generar falsos positivos. A su vez, su Recuperación de 0.789 refleja que detectó aproximadamente el 79% de todas las anomalías reales en el conjunto de datos de ventas de Walmart.
# MAGIC
# MAGIC ¿Se modificaría la recomendación si el objetivo fuese maximizar la Recuperación?
# MAGIC
# MAGIC No se alteraría la recomendación. A pesar de que K-Means y DBSCAN empatan en Recuperación (ambos tienen 0.789), DBSCAN continúa siendo superior porque:
# MAGIC
# MAGIC Ambos identifican el mismo porcentaje de anomalías reales (78.9%)
# MAGIC
# MAGIC Sin embargo, DBSCAN cuenta con una Precisión de 1.0 en comparación con la Precisión de 0.577 de K-Means.
# MAGIC
# MAGIC Esto implica que DBSCAN identifica las mismas anomalías sin ocasionar falsos positivos, mientras que K-Means obliga al equipo de Walmart a revisar numerosas semanas que realmente eran normales.
# MAGIC
# MAGIC En un entorno empresarial como el de Walmart, donde el tiempo del equipo de análisis es valioso, DBSCAN resulta ser la opción más adecuada tanto para maximizar el F1-Score como para optimizar la Recuperación, ya que cumple con ambos objetivos sin consumir recursos en falsas alarmas.
# MAGIC
# MAGIC **Pregunta 2 — Análisis del sistema de alertas AI4I:** usando el resultado del Paso 8 de este notebook, ¿qué porcentaje de las fallas reales cayó en nivel Crítico? Agreguen la captura del gráfico del Paso 9 (score en el tiempo) señalando si el modelo anticipó la falla o solo la confirmó en el mismo momento.
# MAGIC
# MAGIC ![image_1786513311025.png](./image_1786513311025.png "image_1786513311025.png") 
# MAGIC
# MAGIC De acuerdo con los resultados obtenidos en el Paso 8 de nuestro cuaderno de trabajo, de las 339 fallas efectivamente reportadas en el conjunto de datos AI4I, 117 se clasificaron en el nivel Crítico, lo que representa un 34.5% del total de fallas identificadas.
# MAGIC
# MAGIC Distribución del sistema de alertas:
# MAGIC
# MAGIC Normal: 7,500 lecturas (75%)
# MAGIC
# MAGIC Precaución: 1,500 lecturas (15%)
# MAGIC
# MAGIC Crítico: 1,000 lecturas (10%)
# MAGIC
# MAGIC Análisis del gráfico (Paso 9) — Potencial predictivo del modelo:
# MAGIC
# MAGIC Al analizar el gráfico titulado "Puntuación de anomalía a lo largo del tiempo, con fallas reales destacadas", se pueden extraer las siguientes conclusiones:
# MAGIC
# MAGIC El modelo efectivamente predijo diversas fallas: En múltiples áreas del gráfico, los puntos rojos (que representan fallas reales) indican que la puntuación de anomalía comenzó a disminuir antes de que se registrara oficialmente la falla. Esto es particularmente evidente en los grupos de fallas alrededor de las lecturas de 2000 a 4000 y de 5000 a 7000, donde se aprecia una tendencia decreciente en la puntuación antes de que ocurran las fallas.
# MAGIC
# MAGIC Desminución paulatina de la puntuación: En lugar de experimentar un descenso brusco en el instante de la falla, la puntuación exhibe una caída progresiva en las lecturas previas. Esto sugiere que el modelo está reconociendo señales tempranas de deterioro en los sensores (temperatura, torque, velocidad rotacional, desgaste de herramientas) que anticipan la falla mecánica real.
# MAGIC
# MAGIC Fallas no registradas en el nivel Crítico: El 65.5% restante de fallas (222 de 339) se situó en los niveles Normal o Precaución, lo que indica que el umbral p90 es bastante cauteloso. Estas fallas poseían puntuaciones menos extremas, pero continuaron siendo eventos de falla reales.
# MAGIC
# MAGIC Interpretación práctica:
# MAGIC
# MAGIC Este valor del 34.5% sugiere que el sistema de alertas con un umbral p90 prioriza una alta precisión en lugar de una gran sensibilidad: cuando se clasifica algo como Crítico, existe una alta probabilidad de que se trate de una anomalía real, aunque a su vez deja escapar algunas fallas con puntuaciones menos drásticas.
# MAGIC
# MAGIC Para la implementación de mantenimiento predictivo en una instalación industrial, esta capacidad de anticipación que se observa en el gráfico es sumamente valiosa: permite programar el mantenimiento preventivo antes de que ocurra una falla costosa, en lugar de únicamente reaccionar ante una máquina que ya ha fallado. Si la empresa deseara identificar un mayor número de fallas en el nivel Crítico (incrementar el recall), podría modificar el umbral a un percentil menos restrictivo (por ejemplo, p84, como se discutió en la actividad grupal del Paso 10).
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

f1_isoforest = resultados[0]["f1"]
f1_ocsvm = resultados[1]["f1"]

print("=" * 60)
print("RESUMEN PARA LA PREGUNTA DE PORTAFOLIO 2")
print("=" * 60)
print(f"Total de fallas reales en el dataset: {total_fallas}")
print(f"Fallas capturadas en nivel Critico (umbral p90 usado en clase): {len(fallas_criticas)} ({pct_fallas_criticas:.1f}%)")
print(f"F1-Score de Isolation Forest: {f1_isoforest:.3f}")
print(f"F1-Score de One-Class SVM: {f1_ocsvm:.3f}")
print()
print("Copien estos valores (los suyos, de su propia ejecucion) en su respuesta de portafolio,")
print("junto con la captura de pantalla del grafico del Paso 9.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cierre: preguntas de interpretación y razonamiento
# MAGIC
# MAGIC Respondan estas preguntas usando los números reales que obtuvieron en su propia ejecución de este notebook. No hay una única respuesta correcta, lo importante es que su razonamiento esté fundamentado en los datos.
# MAGIC
# MAGIC 1. Según su tabla de resultados del Paso 7, ¿qué modelo tuvo mejor F1-Score en su ejecución: Isolation Forest o One-Class SVM? ¿Por qué creen que pasó eso con estos datos en particular?
# MAGIC
# MAGIC One-Class SVM es superior
# MAGIC  Recall significativamente mayor (lo más importante)
# MAGIC One-Class SVM: detecta 117 de 339 fallas (34.5% recall)
# MAGIC Isolation Forest: solo detecta 71 de 339 fallas (20.9% recall)
# MAGIC Impacto real: One-Class SVM evita 46 fallas adicionales que podrían causar paros no planificados, accidentes o daños costosos. 
# MAGIC
# MAGIC 2. En el Paso 8, ¿qué porcentaje de fallas reales cayó en nivel Crítico? Si ese porcentaje les pareció bajo, ¿qué cambiarían del sistema de alertas para mejorarlo? 
# MAGIC
# MAGIC Menos fallas perdidas (Falsos Negativos)
# MAGIC One-Class SVM: pierde 222 fallas
# MAGIC Isolation Forest: pierde 268 fallas
# MAGIC En mantenimiento, cada falla perdida puede significar daños catastróficos en equipos o accidentes. 
# MAGIC
# MAGIC
# MAGIC 3. Observando el gráfico del Paso 9, ¿el score de Isolation Forest bajó *antes* de que ocurrieran las fallas reales, o solo en el mismo momento? ¿Qué tan útil sería este modelo en una planta real según lo que observaron? 
# MAGIC
# MAGIC Mejor balance general
# MAGIC Precision casi el doble (0.233 vs 0.142)
# MAGIC F1-Score 64% superior (0.278 vs 0.169)
# MAGIC Menos falsos positivos (385 vs 429): menos revisiones innecesarias.
# MAGIC
# MAGIC
# MAGIC 4. En la actividad de grupo (Paso 10), ¿qué percentil recomendarían ustedes como umbral Crítico para una planta donde el mantenimiento correctivo cuesta 10 veces más que el preventivo? Justifiquen con el trade-off que observaron entre fallas capturadas y lecturas marcadas. 
# MAGIC
# MAGIC El ROC-AUC no es determinante aquí
# MAGIC Aunque Isolation Forest tiene mejor ROC-AUC (0.817), esto mide el desempeño teórico en todos los thresholds posibles
# MAGIC En producción usas un threshold específico, y ahí One-Class SVM claramente domina
# MAGIC Impacto en costos
# MAGIC Asumiendo que una falla perdida cuesta 10x más que una revisión innecesaria:
# MAGIC One-Class SVM: 222 fallas perdidas + 385 revisiones innecesarias ≈ 2,605 unidades de costo
# MAGIC Isolation Forest: 268 fallas perdidas + 429 revisiones innecesarias ≈ 3,109 unidades de costo
# MAGIC One-Class SVM representa ~16% menos costo operativo.
# MAGIC
# MAGIC 5. Isolation Forest no necesita las variables escaladas para funcionar bien, pero One-Class SVM sí. ¿Por qué creen que existe esa diferencia entre los dos algoritmos? 
# MAGIC
# MAGIC Isolation Forest es un algoritmo basado en la idea de aislar puntos de datos para detectar anomalías, y su funcionamiento no depende de la escala de las variables porque utiliza divisiones aleatorias en cada dimensión para separar los datos. Por eso, puede identificar outliers aunque las variables tengan rangos muy diferentes. En cambio, One-Class SVM calcula distancias en el espacio multidimensional y construye una frontera alrededor de los datos normales; si las variables no están escaladas, las de mayor rango dominan el cálculo y distorsionan la frontera, haciendo que el modelo pierda precisión. Por eso, One-Class SVM requiere variables escaladas para funcionar correctamente.
# MAGIC
# MAGIC 6. Si tuvieran que explicarle este sistema de alertas a un gerente de planta que no sabe nada de machine learning, ¿cómo se lo explicarían en dos o tres frases? 
# MAGIC
# MAGIC Este sistema de alertas monitorea continuamente los datos de la maquinaria y asigna un nivel de riesgo a cada lectura: Normal, Precaución o Crítico. Cuando detecta señales inusuales que podrían indicar una falla, marca la lectura como Crítico para que el equipo de mantenimiento actúe antes de que ocurra un daño mayor. Así, ayuda a prevenir paros inesperados y reduce los costos, priorizando la seguridad y el funcionamiento eficiente de la planta.
# MAGIC