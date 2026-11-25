import os
import glob
import warnings

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, mutual_info_classif

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    jaccard_score,
)

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10

RANDOM_STATE = 42
K_RANGE = list(range(2, 11))

path = kagglehub.dataset_download("himanshunakrani/iris-dataset")

csv_files = glob.glob(
    os.path.join(path, "**", "*.csv"),
    recursive=True,
)

if not csv_files:
    raise FileNotFoundError("CSV-файл не найден!")

df = pd.read_csv(csv_files[0])

print("Первая строка датасета:")
print(df.head(1).to_string())

print("\nРазмерность датасета:", df.shape)

print("\nИнформация о датасете:")
df.info()

print("\nКоличество пропусков по столбцам:")
print(df.isnull().sum())

print("\nОбщее количество пропусков:", int(df.isnull().sum().sum()))
print("Количество дубликатов:", int(df.duplicated().sum()))

target_candidates = ["species", "class", "variety", "target", "type"]
target_col = next(
    (col for col in df.columns if col.strip().lower() in target_candidates),
    None,
)

y = df[target_col].copy() if target_col is not None else None

technical_cols = [
    col for col in df.columns if col.strip().lower() in ["id", "unnamed: 0"]
]

X = df.select_dtypes(include=np.number).copy()
X.drop(columns=technical_cols, inplace=True, errors="ignore")
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median())

print("\nИспользуемые числовые признаки:")
print(X.columns.tolist())
print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])

if y is not None:
    print(f"\nНайдена целевая переменная '{target_col}' (используется только для внешней оценки):")
    print(y.value_counts())
else:
    print("\nЦелевая переменная не найдена — задача полностью без учителя.")

X.hist(bins=15, figsize=(14, 10), edgecolor="black")
plt.suptitle("Распределение числовых признаков", fontsize=16)
plt.tight_layout()
plt.show()

dist_stats = pd.DataFrame({
    "Среднее": X.mean(),
    "Стд. откл.": X.std(),
    "Асимметрия (Skew)": X.skew(),
    "Эксцесс (Kurtosis)": X.kurtosis(),
})
print("\nХарактеристики распределения признаков:")
print(dist_stats.round(4))

corr_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    square=True,
)
plt.title("Корреляционная матрица признаков")
plt.show()

if y is not None:
    pairplot_data = X.copy()
    pairplot_data[target_col] = y.astype(str).values
    sns.pairplot(pairplot_data, hue=target_col, diag_kind="hist")
else:
    sns.pairplot(X, diag_kind="hist")

plt.show()

outliers_rows = []
for col in X.columns:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (X[col] < lower) | (X[col] > upper)
    outliers_rows.append({
        "Признак": col,
        "Нижняя граница": lower,
        "Верхняя граница": upper,
        "Выбросов": int(mask.sum()),
        "Доля, %": round(mask.mean() * 100, 2),
    })

outliers_df = pd.DataFrame(outliers_rows)
print("\nАнализ выбросов по правилу 1.5 * IQR:")
print(outliers_df.round(4).to_string(index=False))

plt.figure(figsize=(12, 6))
sns.boxplot(data=X)
plt.title("Диаграммы размаха числовых признаков (до масштабирования)")
plt.xticks(rotation=30)
plt.show()

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

print("\nПервые 5 строк стандартизированных данных:")
print(X_scaled.head().round(3))

print("\nСредние значения после StandardScaler:")
print(X_scaled.mean().round(4))

print("\nСтандартные отклонения после StandardScaler:")
print(X_scaled.std().round(4))

inertias = []
kmeans_silhouettes = []

for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    kmeans_silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_RANGE, inertias, marker="o")
axes[0].set_title("Метод локтя (K-means)")
axes[0].set_xlabel("Количество кластеров k")
axes[0].set_ylabel("Inertia (WCSS)")

axes[1].plot(K_RANGE, kmeans_silhouettes, marker="o", color="green")
axes[1].set_title("Силуэтный анализ (K-means)")
axes[1].set_xlabel("Количество кластеров k")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

best_k_silhouette = K_RANGE[int(np.argmax(kmeans_silhouettes))]

x_values = np.array(K_RANGE, dtype=float)
y_values = np.array(inertias, dtype=float)

x_norm = (x_values - x_values.min()) / (x_values.max() - x_values.min())
y_norm = (y_values - y_values.min()) / (y_values.max() - y_values.min())

line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
distances_to_chord = np.abs(y_norm - line_y)
best_k_elbow = K_RANGE[int(np.argmax(distances_to_chord))]

print("\nОптимальное k для K-means по методу локтя:", best_k_elbow)
print("Оптимальное k для K-means по силуэтному анализу:", best_k_silhouette)

optimal_k = best_k_silhouette
print("Выбранное количество кластеров для K-means:", optimal_k)

kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

df_results = df.copy()
df_results["KMeans_Cluster"] = kmeans_labels

print("\nРазмеры кластеров K-means:")
print(pd.Series(kmeans_labels).value_counts().sort_index())

centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=X.columns)
centers_df.index.name = "Cluster"

print("\nЦентры кластеров K-means (в исходных единицах измерения):")
print(centers_df.round(3))

Z = linkage(X_scaled, method="ward", metric="euclidean")

plt.figure(figsize=(15, 7))
dendrogram(Z, leaf_rotation=90, leaf_font_size=7)
plt.title("Дендрограмма иерархической кластеризации (метод Уорда)")
plt.xlabel("Объекты")
plt.ylabel("Евклидово расстояние слияния")
plt.show()

hierarchical_scores = []
for k in K_RANGE:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled)
    hierarchical_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(9, 5))
plt.plot(K_RANGE, hierarchical_scores, marker="o", color="teal")
plt.title("Силуэтный анализ для иерархической кластеризации")
plt.xlabel("Количество кластеров k")
plt.ylabel("Silhouette Score")
plt.show()

best_k_hierarchical = K_RANGE[int(np.argmax(hierarchical_scores))]
print("\nОптимальное k для иерархической кластеризации по силуэту:", best_k_hierarchical)

hierarchical = AgglomerativeClustering(n_clusters=best_k_hierarchical, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X_scaled)

df_results["Hierarchical_Cluster"] = hierarchical_labels

print("\nРазмеры кластеров иерархической кластеризации:")
print(pd.Series(hierarchical_labels).value_counts().sort_index())

f_vals, p_vals = f_classif(X_scaled, kmeans_labels)
importance_data = {
    "Признак": X.columns,
    "Стд. откл. (исх.)": X.std().values,
    "ANOVA F (по кластерам K-means)": f_vals,
    "p-value (ANOVA)": p_vals,
}

if y is not None:
    y_encoded_for_mi = pd.factorize(y)[0]
    mi_vals = mutual_info_classif(
        X_scaled,
        y_encoded_for_mi,
        random_state=RANDOM_STATE,
    )
    importance_data["Mutual Info (с классами y)"] = mi_vals

feature_importance_df = pd.DataFrame(importance_data).sort_values(
    "ANOVA F (по кластерам K-means)",
    ascending=False,
)

print("\nИнформационная значимость признаков:")
print(feature_importance_df.round(4).to_string(index=False))

def calculate_internal_metrics(X_data, labels, method_name):
    return {
        "Метод": method_name,
        "Число кластеров": len(np.unique(labels)),
        "Silhouette Score": silhouette_score(X_data, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X_data, labels),
        "Davies-Bouldin": davies_bouldin_score(X_data, labels),
    }


metrics_kmeans = calculate_internal_metrics(X_scaled, kmeans_labels, "K-means")
metrics_hierarchical = calculate_internal_metrics(
    X_scaled,
    hierarchical_labels,
    "Hierarchical",
)

metrics_df = pd.DataFrame([metrics_kmeans, metrics_hierarchical])
print("\nСравнение внутренних метрик качества:")
print(metrics_df.round(4).to_string(index=False))


def calculate_external_metrics(y_true, labels, method_name):
    y_encoded = pd.factorize(y_true)[0]

    contingency = pd.crosstab(
        pd.Series(y_encoded, name="True"),
        pd.Series(labels, name="Cluster"),
    )
    row_ind, col_ind = linear_sum_assignment(-contingency.values)
    mapping = {
        contingency.columns[c]: contingency.index[r]
        for r, c in zip(row_ind, col_ind)
    }
    mapped_labels = np.array([mapping.get(label, -1) for label in labels])

    return {
        "Метод": method_name,
        "ARI": adjusted_rand_score(y_encoded, labels),
        "Rand Index": rand_score(y_encoded, labels),
        "AMI": adjusted_mutual_info_score(y_encoded, labels),
        "NMI": normalized_mutual_info_score(y_encoded, labels),
        "Homogeneity": homogeneity_score(y_encoded, labels),
        "Completeness": completeness_score(y_encoded, labels),
        "V-measure": v_measure_score(y_encoded, labels),
        "Jaccard (macro)": jaccard_score(
            y_encoded,
            mapped_labels,
            average="macro",
            zero_division=0,
        ),
    }


if y is not None:
    external_kmeans = calculate_external_metrics(y, kmeans_labels, "K-means")
    external_hierarchical = calculate_external_metrics(
        y,
        hierarchical_labels,
        "Hierarchical",
    )
    external_df = pd.DataFrame([external_kmeans, external_hierarchical])
    print("\nВнешние метрики качества:")
    print(external_df.round(4).to_string(index=False))
else:
    external_df = None
    print("\nВнешние метрики не вычислялись: метки классов отсутствуют.")

def analyze_distances(X_data, labels, method_name):
    X_array = np.asarray(X_data)
    unique_labels = np.unique(labels)

    centers = []
    intra_distances = []
    cluster_sizes = {}

    for cluster_id in unique_labels:
        cluster_data = X_array[labels == cluster_id]
        cluster_sizes[cluster_id] = len(cluster_data)
        centers.append(cluster_data.mean(axis=0))

        if len(cluster_data) > 1:
            dists = pdist(cluster_data, metric="euclidean")
            intra_distances.append(float(dists.mean()))
        else:
            intra_distances.append(0.0)

    centers = np.array(centers)
    inter_distances = squareform(pdist(centers, metric="euclidean"))

    print("\n" + "=" * 50)
    print(f"Анализ расстояний: {method_name}")
    print("=" * 50)
    print("Размеры кластеров:", cluster_sizes)
    print("Средние внутрикластерные расстояния:")
    for cluster_id, distance in zip(unique_labels, intra_distances):
        print(f"  Кластер {cluster_id}: {distance:.4f}")

    print("\nМатрица евклидовых расстояний между центрами кластеров:")
    print(
        pd.DataFrame(
            inter_distances,
            index=unique_labels,
            columns=unique_labels,
        ).round(3)
    )

    print(
        "\nСреднее внутрикластерное расстояние:",
        round(float(np.mean(intra_distances)), 4),
    )

    if len(inter_distances) > 1:
        upper_vals = inter_distances[np.triu_indices(len(inter_distances), k=1)]
        print("Минимальное межкластерное расстояние:", round(float(upper_vals.min()), 4))
        print("Максимальное межкластерное расстояние:", round(float(upper_vals.max()), 4))

    return centers, intra_distances, inter_distances


centers_kmeans, intra_kmeans, inter_kmeans = analyze_distances(
    X_scaled,
    kmeans_labels,
    "K-means",
)

centers_hierarchical, intra_hierarchical, inter_hierarchical = analyze_distances(
    X_scaled,
    hierarchical_labels,
    "Иерархическая кластеризация",
)

results_k = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    results_k.append({
        "k": k,
        "Inertia": model.inertia_,
        "Silhouette": silhouette_score(X_scaled, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X_scaled, labels),
        "Davies-Bouldin": davies_bouldin_score(X_scaled, labels),
    })

parameter_df = pd.DataFrame(results_k)
print("\nЗависимость метрик K-means от параметра k:")
print(parameter_df.round(4).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(parameter_df["k"], parameter_df["Inertia"], marker="o")
axes[0, 0].set_title("Зависимость Inertia от k")
axes[0, 0].set_xlabel("k")
axes[0, 0].set_ylabel("Inertia")

axes[0, 1].plot(parameter_df["k"], parameter_df["Silhouette"], marker="o")
axes[0, 1].set_title("Зависимость Silhouette Score от k")
axes[0, 1].set_xlabel("k")
axes[0, 1].set_ylabel("Silhouette")

axes[1, 0].plot(parameter_df["k"], parameter_df["Calinski-Harabasz"], marker="o")
axes[1, 0].set_title("Зависимость Calinski-Harabasz от k")
axes[1, 0].set_xlabel("k")
axes[1, 0].set_ylabel("CH Index")

axes[1, 1].plot(parameter_df["k"], parameter_df["Davies-Bouldin"], marker="o")
axes[1, 1].set_title("Зависимость Davies-Bouldin от k")
axes[1, 1].set_xlabel("k")
axes[1, 1].set_ylabel("DB Index")

plt.tight_layout()
plt.show()

seeds = [0, 1, 7, 21, 42, 99, 123]
stability_scores = []
for seed in seeds:
    alt_model = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
    alt_labels = alt_model.fit_predict(X_scaled)
    stability_scores.append(adjusted_rand_score(kmeans_labels, alt_labels))

print(
    f"\nУстойчивость K-means (ARI по seeds {seeds}): "
    f"среднее={np.mean(stability_scores):.4f}, "
    f"мин={np.min(stability_scores):.4f}, макс={np.max(stability_scores):.4f}"
)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print("\nОбъясненная дисперсия компонентами PCA:", np.round(pca.explained_variance_ratio_, 4))
print("Суммарная объясненная дисперсия (2 компоненты):", round(float(pca.explained_variance_ratio_.sum()), 4))

loadings_df = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=["PC1", "PC2"],
)
print("\nНагрузки признаков на главные компоненты:")
print(loadings_df.round(3))

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "KMeans": kmeans_labels,
    "Hierarchical": hierarchical_labels,
})
if y is not None:
    pca_df["TrueClass"] = y.astype(str).values

n_plots = 3 if y is not None else 2
fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="KMeans",
    palette="Set1",
    s=70,
    ax=axes[0],
)
axes[0].set_title(f"K-means (k={optimal_k}) в пространстве PCA")

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Hierarchical",
    palette="Set2",
    s=70,
    ax=axes[1],
)
axes[1].set_title(f"Hierarchical (k={best_k_hierarchical}) в пространстве PCA")

if y is not None:
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="TrueClass",
        palette="Set1",
        s=70,
        ax=axes[2],
    )
    axes[2].set_title("Исходные метки классов в пространстве PCA")

plt.tight_layout()
plt.show()

centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=kmeans_labels,
    palette="Set1",
    s=60,
)
plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    marker="X",
    s=250,
    color="black",
    label="Центры кластеров K-means",
)
plt.title("Кластеры K-means и их центры в проекции PCA")
plt.xlabel("Первая главная компонента (PC1)")
plt.ylabel("Вторая главная компонента (PC2)")
plt.legend()
plt.show()

interpretation_df = X.copy()
interpretation_df["Cluster"] = kmeans_labels
cluster_means = interpretation_df.groupby("Cluster").mean()

print("\nСредние значения признаков по кластерам K-means (исходный масштаб):")
print(cluster_means.round(3))

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Тепловая карта средних значений признаков по кластерам K-means")
plt.xlabel("Признаки")
plt.ylabel("Кластеры")
plt.show()

if y is not None:
    comparison = pd.crosstab(
        pd.Series(y.values, name="Настоящий класс"),
        pd.Series(kmeans_labels, name="Кластер K-means"),
    )
    print("\nТаблица сопряженности (настоящие классы vs кластеры K-means):")
    print(comparison)

    plt.figure(figsize=(9, 6))
    sns.heatmap(comparison, annot=True, cmap="Blues", fmt="d")
    plt.title("Сопоставление настоящих классов и кластеров K-means")
    plt.show()

method_similarity_ari = adjusted_rand_score(kmeans_labels, hierarchical_labels)
print("\nСогласованность разбиений K-means и Hierarchical (ARI):", round(method_similarity_ari, 4))

comparison_df = pd.DataFrame({
    "Метрика": ["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"],
    "K-means": [
        metrics_kmeans["Silhouette Score"],
        metrics_kmeans["Calinski-Harabasz"],
        metrics_kmeans["Davies-Bouldin"],
    ],
    "Hierarchical": [
        metrics_hierarchical["Silhouette Score"],
        metrics_hierarchical["Calinski-Harabasz"],
        metrics_hierarchical["Davies-Bouldin"],
    ],
})

print("\nСводная таблица сравнения методов:")
print(comparison_df.round(4).to_string(index=False))

comparison_df.set_index("Метрика").plot(kind="bar", figsize=(12, 6))
plt.title("Сравнение качества K-means и иерархической кластеризации")
plt.ylabel("Значение метрики")
plt.xticks(rotation=0)
plt.legend(title="Метод")
plt.tight_layout()
plt.show()

df_results.to_csv("iris_clustering_results.csv", index=False)
metrics_df.to_csv("clustering_metrics.csv", index=False)
parameter_df.to_csv("kmeans_parameter_analysis.csv", index=False)
feature_importance_df.to_csv("feature_importance.csv", index=False)

if external_df is not None:
    external_df.to_csv("external_metrics.csv", index=False)

print("\nФайлы успешно сохранены:")
print("1. iris_clustering_results.csv")
print("2. clustering_metrics.csv")
print("3. kmeans_parameter_analysis.csv")
print("4. feature_importance.csv")
if external_df is not None:
    print("5. external_metrics.csv")

print("ИТОГОВОЕ ЗАКЛЮЧЕНИЕ")
print(f"Объектов: {X.shape[0]}, числовых признаков: {X.shape[1]}")
print(f"Оптимальное k: K-means = {optimal_k}, Hierarchical = {best_k_hierarchical}")
print(
    f"K-means -> Silhouette: {metrics_kmeans['Silhouette Score']:.4f}, "
    f"CH: {metrics_kmeans['Calinski-Harabasz']:.4f}, "
    f"DB: {metrics_kmeans['Davies-Bouldin']:.4f}"
)
print(
    f"Hierarchical -> Silhouette: {metrics_hierarchical['Silhouette Score']:.4f}, "
    f"CH: {metrics_hierarchical['Calinski-Harabasz']:.4f}, "
    f"DB: {metrics_hierarchical['Davies-Bouldin']:.4f}"
)
print(f"Сходство разбиений двух методов (ARI): {method_similarity_ari:.4f}")
if y is not None:
    print(
        f"Внешний ARI -> K-means: {external_kmeans['ARI']:.4f}, "
        f"Hierarchical: {external_hierarchical['ARI']:.4f}"
    )