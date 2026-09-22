# Databricks notebook source
# MAGIC %md
# MAGIC ## Cargar base de datos

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 3: Cargar base de datos
df = spark.read.csv('/Volumes/workspace/big_data/ventas/sales_data_sample.csv', header=True)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Limpieza de datos

# COMMAND ----------

# Eliminar filas duplicadas
df_clean = df.dropDuplicates()

# Eliminar filas con valores nulos 
df_clean = df_clean.dropna()

# Convertir columnas numéricas a tipo adecuado
df_clean = df_clean.withColumn('QUANTITYORDERED', df_clean['QUANTITYORDERED'].cast('int'))
df_clean = df_clean.withColumn('PRICEEACH', df_clean['PRICEEACH'].cast('float'))

# Eliminar espacios en blanco en columnas de texto
from pyspark.sql.functions import trim
df_clean = df_clean.withColumn('PRODUCTCODE', trim(df_clean['PRODUCTCODE']))
df_clean = df_clean.withColumn('CUSTOMERNAME', trim(df_clean['CUSTOMERNAME']))
display(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicción con Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Preparación de datos

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.ml.feature import VectorAssembler

# Agrupar ventas promedio por PRODUCTCODE y PRODUCTLINE
df_aggregated = df_clean.groupBy("PRODUCTCODE", "PRODUCTLINE") \
                        .agg(avg("SALES").alias("AVG_SALES"))

# Convertir en formato adecuado para ML
assembler = VectorAssembler(inputCols=["AVG_SALES"], outputCol="features")
df_ml = assembler.transform(df_aggregated)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Método del codo

# COMMAND ----------

from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

k_values = list(range(2, 11))  # probaremos K de 2 a 10
wss_values = []

for k in k_values:
    kmeans = KMeans(k=k, seed=42, featuresCol="features")
    model = kmeans.fit(df_ml)
    wss = model.summary.trainingCost
    wss_values.append(wss)

# 📊 Graficar el método del codo
plt.figure()
plt.plot(k_values, wss_values, marker='o')
plt.title("Método del Codo - WSS")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("WSS")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenar K-Means 

# COMMAND ----------

k_optimo = 5

kmeans_final = KMeans(k=k_optimo, seed=42, featuresCol="features")
modelo_final = kmeans_final.fit(df_ml)

df_clusters = modelo_final.transform(df_ml)
df_clusters.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizamos los clusters

# COMMAND ----------

import pandas as pd

# Convertir a Pandas para graficar
result_pd = df_clusters.select("AVG_SALES", "prediction").toPandas()

plt.figure()
for c in result_pd.prediction.unique():
    cluster_data = result_pd[result_pd.prediction == c]
    plt.scatter(cluster_data["AVG_SALES"], [c]*len(cluster_data), label=f"Cluster {c}")

plt.title("Clusters por ventas promedio")
plt.xlabel("AVG_SALES")
plt.ylabel("Cluster")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC | Cluster        | Nivel de ventas         | Acción estratégica sugerida             |
# MAGIC | -------------- | ----------------------- | --------------------------------------- |
# MAGIC | 2 (bajo)       | Muy bajas               | Revisar si deben mantenerse en catálogo |
# MAGIC | 4 (medio-bajo) | Estables pero discretos | Potenciar ventas con promociones        |
# MAGIC | 0 (medio)      | Estable + margen        | Optimizar precios/costos                |
# MAGIC | 1 (medio-alto) | Alto rendimiento        | Mantener e impulsar                     |
# MAGIC | 3 (muy alto)   | Clave para el negocio   | Priorizar disponibilidad y marketing    |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Indice de silueta

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
silhouette_score = evaluator.evaluate(df_clusters)

print(f"Índice de Silueta para K={k_optimo}: {silhouette_score}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Analisis de silueta

# COMMAND ----------

# MAGIC %md
# MAGIC | Valor     | Interpretación         |
# MAGIC | --------- | ---------------------- |
# MAGIC | 0.7 – 1.0 | Excelente agrupamiento |
# MAGIC | 0.5 – 0.7 | Bueno                  |
# MAGIC | 0.2 – 0.5 | Podría mejorar         |
# MAGIC | < 0.2     | Mala separación        |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster 2 - AVG Sales y Quantity Order

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# 1. Agregaciones por producto
df_agg2 = df_clean.groupBy("PRODUCTCODE", "PRODUCTLINE") \
                 .agg(avg("SALES").alias("AVG_SALES"),
                      avg("QUANTITYORDERED").alias("AVG_QTY"))

# 2. Vector para clustering
assembler = VectorAssembler(
    inputCols=["AVG_SALES", "AVG_QTY"],
    outputCol="features"
)
df_ml2 = assembler.transform(df_agg2)

# 3. Escalamiento recomendable para clustering
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_ml2)
df_scaled = scaler_model.transform(df_ml2)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Metodo del codo

# COMMAND ----------

from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

k_values = range(2, 11)
wss_list = []

for k in k_values:
    model = KMeans(k=k, seed=10, featuresCol="scaled_features").fit(df_scaled)
    wss_list.append(model.summary.trainingCost)

plt.plot(k_values, wss_list, marker='o')
plt.xlabel("Número de clusters (K)")
plt.ylabel("WSS")
plt.title("Método del Codo - Ventas y Cantidad")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenamos el modelo

# COMMAND ----------

k_optimo = 4

kmeans_final = KMeans(k=k_optimo, seed=10, featuresCol="scaled_features")
model_final = kmeans_final.fit(df_scaled)

df_clusters2 = model_final.transform(df_scaled)
df_clusters2.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizar clusters

# COMMAND ----------

import pandas as pd

pdf = df_clusters2.select("AVG_SALES", "AVG_QTY", "prediction").toPandas()

plt.figure(figsize=(8,6))
for c in pdf.prediction.unique():
    cluster = pdf[pdf.prediction == c]
    plt.scatter(cluster["AVG_SALES"], cluster["AVG_QTY"], label=f"Cluster {c}")

plt.xlabel("Promedio de Ventas (AVG_SALES)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.title("Clusters por Ventas y Cantidad Vendida")
plt.legend()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC | Cluster   | Color      | Características de Ventas | Características de Cantidad Vendida | Interpretación                        |
# MAGIC | --------- | ---------- | ------------------------- | ----------------------------------- | ------------------------------------- |
# MAGIC | Cluster 2 | Rojo 🔴    | Altas                     | Altas                               | Productos estrella / alto rendimiento |
# MAGIC | Cluster 0 | Azul 🔵    | Medias                    | Medias                              | Productos estables en ventas          |
# MAGIC | Cluster 1 | Verde 🟢   | Bajas–Medias              | Medias–Altas                        | Alta rotación con precios bajos       |
# MAGIC | Cluster 3 | Naranja 🟠 | Bajas                     | Bajas                               | Bajo rendimiento – requiere atención  |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Con centroides

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas
pdf = df_clusters2.select(
    "AVG_SALES",
    "AVG_QTY",
    "prediction"
).toPandas()

# Get centroids from trained model
centroids_scaled = model_final.clusterCenters()

# Get original min and max from scaler_model
orig_min = scaler_model.originalMin
orig_max = scaler_model.originalMax
scaler_min = scaler_model.getMin()
scaler_max = scaler_model.getMax()

# Manually invert MinMax scaling for each centroid
centroids_unscaled = []
for c in centroids_scaled:
    unscaled = [
        (c[i] - scaler_min) / (scaler_max - scaler_min) * (orig_max[i] - orig_min[i]) + orig_min[i]
        for i in range(len(c))
    ]
    centroids_unscaled.append(unscaled)

centroids = pd.DataFrame(
    centroids_unscaled,
    columns=["feat1", "feat2"]
)

# Plot
plt.figure(figsize=(9,6))

for c in pdf.prediction.unique():
    cluster_data = pdf[pdf.prediction == c]
    plt.scatter(
        cluster_data["AVG_SALES"],
        cluster_data["AVG_QTY"],
        label=f"Cluster {c}"
    )

plt.scatter(
    centroids["feat1"],
    centroids["feat2"],
    s=250, marker='X', color='black', edgecolors='white', linewidths=2,
    label="Centroides"
)

plt.title("Clusters por Ventas y Cantidad Vendida con Centroides")
plt.xlabel("Promedio de Ventas (AVG_SALES)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

# Convert Spark DataFrame to Pandas
pdf = df_clusters2.select("AVG_SALES", "AVG_QTY", "prediction").toPandas()

# Plot
plt.figure(figsize=(10,7))

colors = ["blue", "orange", "green", "red","purple","brown"]  # Ajusta según tus clusters

for c in sorted(pdf.prediction.unique()):
    cluster_data = pdf[pdf.prediction == c]
    
    # Gráfico de los puntos
    plt.scatter(
        cluster_data["AVG_SALES"],
        cluster_data["AVG_QTY"],
        label=f"Cluster {c}",
        s=60,
        color=colors[c]
    )
    
    # Trazar convex hull si hay suficientes puntos
    if len(cluster_data) >= 3:
        points = cluster_data[["AVG_SALES", "AVG_QTY"]].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        plt.fill(
            hull_points[:,0],
            hull_points[:,1],
            alpha=0.15,
            color=colors[c]
        )

# Centroides
plt.scatter(
    centroids["feat1"],
    centroids["feat2"],
    s=300, marker='X', color='black', edgecolors='white', linewidths=2,
    label="Centroides"
)

plt.title("Clusters por Ventas y Cantidad Vendida con Áreas Encerradas")
plt.xlabel("Promedio de Ventas (AVG_SALES)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Validacion mediante silueta

# COMMAND ----------

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol="scaled_features", metricName="silhouette")
sil_score = evaluator.evaluate(df_clusters2)

print(f"Índice de silueta: {sil_score}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejercicios de clase

# COMMAND ----------

# MAGIC %md
# MAGIC En Databricks, utilizando la base de datos de ventas, realizar los siguientes análisis de clustering en PySpark con K-Means:
# MAGIC
# MAGIC 1. Clustering con PRICEEACH y AVG_QTY
# MAGIC Análisis del impacto del precio en la cantidad vendida.
# MAGIC
# MAGIC 2. Clustering con PRICEEACH y AVG_SALES
# MAGIC Análisis del aporte económico de los productos según su precio.

# COMMAND ----------

# MAGIC %md
# MAGIC # Clustering con PRICEEACH y AVG_QTY Análisis del impacto del precio en la cantidad vendida.

# COMMAND ----------

from pyspark.sql.functions import avg
from pyspark.ml.feature import VectorAssembler, MinMaxScaler

# 1. Agregaciones por producto
df_agg_price_qty = df_clean.groupBy("PRODUCTCODE", "PRODUCTLINE") \
    .agg(avg("PRICEEACH").alias("AVG_PRICE"),
         avg("QUANTITYORDERED").alias("AVG_QTY"))

# 2. Vector para clustering
assembler_price_qty = VectorAssembler(
    inputCols=["AVG_PRICE", "AVG_QTY"],
    outputCol="features"
)
df_ml_price_qty = assembler_price_qty.transform(df_agg_price_qty)

# 3. Escalamiento recomendable para clustering
scaler_price_qty = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model_price_qty = scaler_price_qty.fit(df_ml_price_qty)
df_scaled_price_qty = scaler_model_price_qty.transform(df_ml_price_qty)

display(df_scaled_price_qty.select("PRODUCTCODE", "PRODUCTLINE", "AVG_PRICE", "AVG_QTY", "scaled_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Metodo del Codo
# MAGIC

# COMMAND ----------

from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

k_values = range(2, 11)
wss_list = []

for k in k_values:
    model = KMeans(k=k, seed=10, featuresCol="scaled_features").fit(df_scaled)
    wss_list.append(model.summary.trainingCost)

plt.plot(k_values, wss_list, marker='o')
plt.xlabel("Número de clusters (K)")
plt.ylabel("WSS")
plt.title("Método del Codo - Precio y Cantidad")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenar el Modelo
# MAGIC

# COMMAND ----------

k_optimo = 4
kmeans_final = KMeans(k=k_optimo, seed=10, featuresCol="scaled_features")
model_final = kmeans_final.fit(df_scaled)

df_clusters2 = model_final.transform(df_scaled)
df_clusters2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualizar clusters

# COMMAND ----------

# DBTITLE 1,Cell 43
import pandas as pd

pdf = df_scaled_price_qty.select("AVG_PRICE", "AVG_QTY").toPandas()

plt.figure(figsize=(8,6))
plt.scatter(pdf["AVG_PRICE"], pdf["AVG_QTY"], label="Datos")
plt.xlabel("Promedio de Ventas (AVG_PRICE)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.title("Clusters por Precio y Cantidad Vendida")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Con Centroides

# COMMAND ----------

# DBTITLE 1,Cell 45
import pandas as pd
import matplotlib.pyplot as plt

# Convert Spark DataFrame to Pandas
pdf = df_scaled_price_qty.select(
    "AVG_PRICE",
    "AVG_QTY"
).toPandas()

# Get centroids from trained model
centroids_scaled = model_final.clusterCenters()

# Get original min and max from scaler_model
orig_min = scaler_model.originalMin
orig_max = scaler_model.originalMax
scaler_min = scaler_model.getMin()
scaler_max = scaler_model.getMax()

# Manually invert MinMax scaling for each centroid
centroids_unscaled = []
for c in centroids_scaled:
    unscaled = [
        (c[i] - scaler_min) / (scaler_max - scaler_min) * (orig_max[i] - orig_min[i]) + orig_min[i]
        for i in range(len(c))
    ]
    centroids_unscaled.append(unscaled)

centroids = pd.DataFrame(
    centroids_unscaled,
    columns=["feat1", "feat2"]
)

# Plot
plt.figure(figsize=(9,6))

plt.scatter(
    pdf["AVG_PRICE"],
    pdf["AVG_QTY"],
    label="Datos"
)

plt.scatter(
    centroids["feat1"],
    centroids["feat2"],
    s=250, marker='X', color='black', edgecolors='white', linewidths=2,
    label="Centroides"
)

plt.title("Clusters por Precio y Cantidad Vendida con Centroides")
plt.xlabel("Promedio de Precio (AVG_PRICE)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# COMMAND ----------

# DBTITLE 1,Cell 46
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

# Convert Spark DataFrame to Pandas
pdf = model_final.transform(df_scaled_price_qty).select("AVG_PRICE", "AVG_QTY", "prediction").toPandas()

# Plot
plt.figure(figsize=(10,7))

colors = ["blue", "orange", "green", "red","purple","brown"]  # Ajusta según tus clusters

for c in sorted(pdf.prediction.unique()):
    cluster_data = pdf[pdf.prediction == c]
    
    # Gráfico de los puntos
    plt.scatter(
        cluster_data["AVG_PRICE"],
        cluster_data["AVG_QTY"],
        label=f"Cluster {c}",
        s=60,
        color=colors[c]
    )
    
    # Trazar convex hull si hay suficientes puntos
    if len(cluster_data) >= 3:
        points = cluster_data[["AVG_PRICE", "AVG_QTY"]].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        
        plt.fill(
            hull_points[:,0],
            hull_points[:,1],
            alpha=0.15,
            color=colors[c]
        )

# Centroides
plt.scatter(
    centroids["feat1"],
    centroids["feat2"],
    s=300, marker='X', color='black', edgecolors='white', linewidths=2,
    label="Centroides"
)

plt.title("Clusters por Precio y Cantidad Vendida con Áreas Encerradas")
plt.xlabel("Promedio de Precio (AVG_PRICE)")
plt.ylabel("Promedio de Cantidad Vendida (AVG_QTY)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()