import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import tensorflow as tf

from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, classification_report
)

from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

SEED = 67
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

path = kagglehub.dataset_download("himanshunakrani/iris-dataset")
csv_files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("В скачанном датасете не найден CSV-файл.")

df = pd.read_csv(os.path.join(path, csv_files[0]))
df.columns = df.columns.str.strip()

print("Размерность датасета:", df.shape)
print("Первая строка датасета:")
print(df.head(1).to_string())
print("\nТипы столбцов:")
print(df.dtypes)

target_candidates = [
    c for c in df.columns
    if c.lower() in ["variety", "species", "class", "target"]
]

if not target_candidates:
    raise ValueError(
        "Не найден столбец класса. Проверьте названия столбцов: "
        + str(df.columns.tolist())
    )

target_col = target_candidates[0]

df = df.dropna(subset=[target_col]).copy()

id_cols = [
    c for c in df.columns
    if c.lower() in ["id", "unnamed: 0"]
]
df = df.drop(columns=id_cols, errors="ignore")

X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(X.median(numeric_only=True))

if X.isnull().any().any():
    raise ValueError("После обработки остались пропуски в признаках.")

le = LabelEncoder()
y = le.fit_transform(y)

n_classes = len(le.classes_)
if n_classes < 2:
    raise ValueError("Для классификации необходимо минимум два класса.")

print("\nКлассы:", le.classes_)
print("Количество объектов:", len(X))
print("Количество признаков:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_mm = scaler_minmax.fit_transform(X_train)
X_test_mm = scaler_minmax.transform(X_test)

models = {
    "GaussianNB": (GaussianNB(), "std"),
    "MultinomialNB": (MultinomialNB(), "mm"),
    "ComplementNB": (ComplementNB(), "mm"),
    "BernoulliNB": (BernoulliNB(), "std"),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED), "std"
    ),
    "LDA": (LinearDiscriminantAnalysis(), "std"),
    "SVM": (SVC(probability=True, random_state=SEED), "std"),
    "KNN": (KNeighborsClassifier(), "std")
}

X_data = {
    "std": (X_train_std, X_test_std),
    "mm": (X_train_mm, X_test_mm)
}

y_test_bin = label_binarize(
    y_test, classes=np.arange(n_classes)
)

results = []
trained_models = {}
roc_scores = {}

for name, (model, data_type) in models.items():
    Xtr, Xte = X_data[data_type]

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte)
    else:
        y_score = model.decision_function(Xte)

    if n_classes == 2:
        score = (
            y_score[:, 1] if y_score.ndim == 2
            else y_score
        )
        roc_auc = roc_auc_score(y_test, score)
    else:
        roc_auc = roc_auc_score(
            y_test, y_score,
            multi_class="ovr",
            average="macro"
        )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model
    roc_scores[name] = y_score

results_df = pd.DataFrame(results).sort_values(
    "F1-Score", ascending=False
)

print("\nСРАВНЕНИЕ КЛАССИФИКАТОРОВ")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "classification_results.csv", index=False
)

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results_df,
        x="Model", y=metric
    )
    plt.title(f"Сравнение моделей по метрике {metric}")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()

for model_name, y_score in roc_scores.items():
    plt.figure(figsize=(7, 5))

    if n_classes == 2:
        score = (
            y_score[:, 1] if y_score.ndim == 2
            else y_score
        )
        fpr, tpr, _ = roc_curve(y_test, score)
        plt.plot(
            fpr, tpr,
            label=f"AUC = {auc(fpr, tpr):.3f}"
        )
    else:
        for i, class_name in enumerate(le.classes_):
            fpr, tpr, _ = roc_curve(
                y_test_bin[:, i], y_score[:, i]
            )
            plt.plot(
                fpr, tpr,
                label=f"{class_name} (AUC={auc(fpr, tpr):.3f})"
            )

    plt.plot([0, 1], [0, 1], "k--", label="Случайный классификатор")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC-кривая: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=SEED
)

search_configs = {
    "GaussianNB": (
        GaussianNB(),
        StandardScaler(),
        {"model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5]}
    ),
    "MultinomialNB": (
        MultinomialNB(),
        MinMaxScaler(),
        {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
    ),
    "ComplementNB": (
        ComplementNB(),
        MinMaxScaler(),
        {"model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
    ),
    "BernoulliNB": (
        BernoulliNB(),
        StandardScaler(),
        {
            "model__alpha": [0.1, 0.5, 1.0, 2.0],
            "model__binarize": [0.0, 0.5, 1.0]
        }
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        StandardScaler(),
        {
            "model__max_depth": [2, 3, 4, 5, None],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"]
        }
    ),
    "LDA": (
        LinearDiscriminantAnalysis(),
        StandardScaler(),
        {
            "model__solver": ["svd", "lsqr"],
            "model__shrinkage": [None, "auto"]
        }
    ),
    "SVM": (
    SVC(random_state=SEED),
    StandardScaler(),
    {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    }
    ),
    "KNN": (
        KNeighborsClassifier(),
        StandardScaler(),
        {
            "model__n_neighbors": [1, 3, 5, 7, 9, 11],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }
    )
}

tuning_results = []
best_models = {}

for name, (estimator, scaler, param_grid) in search_configs.items():
    pipe = Pipeline([
        ("scaler", scaler),
        ("model", estimator)
    ])

    if name == "LDA":
        param_grid = [
            {
                "model__solver": ["svd"]
            },
            {
                "model__solver": ["lsqr"],
                "model__shrinkage": [None, "auto"]
            }
        ]

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    y_pred = search.predict(X_test)

    tuning_results.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test F1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "Best Parameters": str(search.best_params_)
    })

    print(f"\n{name}")
    print("Лучшие параметры:", search.best_params_)
    print("Лучший CV F1:", round(search.best_score_, 4))

tuning_df = pd.DataFrame(tuning_results)
print("\nРЕЗУЛЬТАТЫ ПОДБОРА ГИПЕРПАРАМЕТРОВ")
print(tuning_df.round(4).to_string(index=False))
tuning_df.to_csv("hyperparameter_results.csv", index=False)

X_nn_train, X_val, y_nn_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=SEED,
    stratify=y_train
)

nn_scaler = StandardScaler()
X_nn_train = nn_scaler.fit_transform(X_nn_train)
X_val = nn_scaler.transform(X_val)
X_nn_test = nn_scaler.transform(X_test)

y_nn_train_cat = tf.keras.utils.to_categorical(
    y_nn_train, num_classes=n_classes
)
y_val_cat = tf.keras.utils.to_categorical(
    y_val, num_classes=n_classes
)
y_test_cat = tf.keras.utils.to_categorical(
    y_test, num_classes=n_classes
)

def create_nn(units=16, dropout_rate=0.2, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_nn_train.shape[1],)),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(max(units // 2, 4), activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

log_root = os.path.join(os.getcwd(), "logs", "fit")
os.makedirs(log_root, exist_ok=True)

nn_configs = [
    {"units": 16, "dropout_rate": 0.2, "learning_rate": 0.001},
    {"units": 32, "dropout_rate": 0.2, "learning_rate": 0.001},
    {"units": 32, "dropout_rate": 0.4, "learning_rate": 0.001},
    {"units": 32, "dropout_rate": 0.2, "learning_rate": 0.01}
]

nn_results = []
nn_histories = []
nn_models = []

for i, config in enumerate(nn_configs, start=1):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    nn = create_nn(**config)
    run_name = (
        f"run_{i}_units{config['units']}"
        f"_dropout{config['dropout_rate']}"
        f"_lr{config['learning_rate']}_"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    log_dir = os.path.join(log_root, run_name)

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
    ]

    history = nn.fit(
        X_nn_train, y_nn_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=150,
        batch_size=8,
        callbacks=callbacks,
        verbose=0
    )

    test_loss, test_accuracy = nn.evaluate(
        X_nn_test, y_test_cat, verbose=0
    )
    test_pred = np.argmax(nn.predict(X_nn_test, verbose=0), axis=1)

    nn_results.append({
        **config,
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Test F1": f1_score(
            y_test, test_pred, average="macro", zero_division=0
        ),
        "Log Directory": log_dir
    })

    nn_histories.append((config, history))
    nn_models.append(nn)

nn_results_df = pd.DataFrame(nn_results)
print("\n=== РЕЗУЛЬТАТЫ НЕЙРОННОЙ СЕТИ ===")
print(nn_results_df.round(4).to_string(index=False))
nn_results_df.to_csv("neural_network_results.csv", index=False)

for config, history in nn_histories:
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist["loss"], label="Train loss")
    axes[0].plot(hist["val_loss"], label="Validation loss")
    axes[0].set_title("Функция потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(hist["accuracy"], label="Train accuracy")
    axes[1].plot(hist["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Точность обучения")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(f"Нейронная сеть: {config}")
    plt.tight_layout()
    plt.show()

print("\nДля запуска TensorBoard выполните в терминале:")
print(f"tensorboard --logdir=\"{log_root}\"")