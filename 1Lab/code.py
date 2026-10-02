import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from ucimlrepo import fetch_ucirepo
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 110


def calculate_excel_descriptive_stats(df_num):
    metrics = []
    
    for col in df_num.columns:
        series = df_num[col].dropna()
        n = len(series)
        
        mean_val = series.mean()
        std_val = series.std(ddof=1)
        sem_val = stats.sem(series)
        median_val = series.median()
        
        mode_series = series.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan
        
        var_val = series.var(ddof=1)
        kurt_val = series.kurtosis()
        skew_val = series.skew()
        
        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        sum_val = series.sum()
        
        if n > 1:
            confidence_val = stats.t.ppf((1 + 0.95) / 2., n - 1) * (std_val / np.sqrt(n))
        else:
            confidence_val = np.nan
            
        metrics.append({
            'Признак': col,
            'Среднее': mean_val,
            'Стандартная ошибка': sem_val,
            'Медиана': median_val,
            'Мода': mode_val,
            'Стандартное отклонение': std_val,
            'Дисперсия выборки': var_val,
            'Эксцесс': kurt_val,
            'Асимметричность': skew_val,
            'Интервал (Размах)': range_val,
            'Минимум': min_val,
            'Максимум': max_val,
            'Сумма': sum_val,
            'Счет': n,
            'Уровень надежности(95,0%)': confidence_val
        })
        
    return pd.DataFrame(metrics).set_index('Признак').T


def main():
    show_plots = '--no-show' not in sys.argv
    
    print("1. Подготовка данных")
    concrete_compressive_strength = fetch_ucirepo(id=165)

    X_raw = concrete_compressive_strength.data.features
    y_raw = concrete_compressive_strength.data.targets

    df_raw = pd.concat([X_raw, y_raw], axis=1)
    print(f"Размерность датасета: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов\n")

    col_map = {
        df_raw.columns[0]: 'Цемент (кг/м³)',
        df_raw.columns[1]: 'Доменный шлак (кг/м³)',
        df_raw.columns[2]: 'Летучая зола (кг/м³)',
        df_raw.columns[3]: 'Вода (кг/м³)',
        df_raw.columns[4]: 'Пластификатор (кг/м³)',
        df_raw.columns[5]: 'Крупный заполнитель (кг/м³)',
        df_raw.columns[6]: 'Мелкий заполнитель (кг/м³)',
        df_raw.columns[7]: 'Возраст (дни)',
        df_raw.columns[8]: 'Прочность на сжатие (МПа)'
    }
    df = df_raw.rename(columns=col_map)

    print("Первая строка датасета:")
    print(df.head(1).to_string())

    duplicates_count = df.duplicated().sum()
    if duplicates_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"\nУдалено дубликатов: {duplicates_count}. Новый размер: {df.shape}")

    df['Водоцементное отношение (W/C)'] = df['Вода (кг/м³)'] / (df['Цемент (кг/м³)'] + 1e-5)
    df['Лог(Возраст)'] = np.log1p(df['Возраст (дни)'])

    print("\n2. Описательная статистика")
    stats_table = calculate_excel_descriptive_stats(df)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats_table.round(4).to_string())

    c_main = '#3B5998'      
    c_secondary = '#4B8BBE' 
    c_teal = '#2A9D8F'      
    c_accent = '#E76F51'    
    c_purple = '#6A4C93'    
    c_dark = '#264653'     
    c_olive = '#606C38'     
    c_amber = '#D4A373'     
    c_coral = '#F4A261'     
    c_slate = '#4A5568'     

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))

    sns.histplot(df['Прочность на сжатие (МПа)'], kde=True, ax=axes[0, 0], color=c_main, bins=35, edgecolor='black')
    axes[0, 0].set_title('Прочность на сжатие (МПа)')
    axes[0, 0].set_xlabel('Прочность (МПа)')
    axes[0, 0].set_ylabel('Частота')

    sns.histplot(df['Цемент (кг/м³)'], kde=True, ax=axes[0, 1], color=c_secondary, bins=25, edgecolor='black')
    axes[0, 1].set_title('Расход цемента')
    axes[0, 1].set_xlabel('Цемент (кг/м³)')
    axes[0, 1].set_ylabel('Частота')

    sns.histplot(df['Водоцементное отношение (W/C)'], kde=True, ax=axes[0, 2], color=c_teal, bins=30, edgecolor='black')
    axes[0, 2].set_title('Водоцементное отношение (W/C)')
    axes[0, 2].set_xlabel('W/C')
    axes[0, 2].set_ylabel('Частота')

    sns.histplot(df['Вода (кг/м³)'], kde=True, ax=axes[0, 3], color=c_dark, bins=25, edgecolor='black')
    axes[0, 3].set_title('Расход воды')
    axes[0, 3].set_xlabel('Вода (кг/м³)')
    axes[0, 3].set_ylabel('Частота')

    sns.histplot(df['Доменный шлак (кг/м³)'], kde=True, ax=axes[1, 0], color=c_olive, bins=25, edgecolor='black')
    axes[1, 0].set_title('Распределение доменного шлака')
    axes[1, 0].set_xlabel('Доменный шлак (кг/м³)')
    axes[1, 0].set_ylabel('Частота')

    sns.histplot(df['Летучая зола (кг/м³)'], kde=True, ax=axes[1, 1], color=c_amber, bins=25, edgecolor='black')
    axes[1, 1].set_title('Распределение летучей золы')
    axes[1, 1].set_xlabel('Летучая зола (кг/м³)')
    axes[1, 1].set_ylabel('Частота')

    sns.histplot(df['Пластификатор (кг/м³)'], kde=True, ax=axes[1, 2], color=c_purple, bins=25, edgecolor='black')
    axes[1, 2].set_title('Распределение пластификатора')
    axes[1, 2].set_xlabel('Пластификатор (кг/м³)')
    axes[1, 2].set_ylabel('Частота')

    sns.histplot(df['Возраст (дни)'], kde=True, ax=axes[1, 3], color=c_accent, bins=30, edgecolor='black')
    axes[1, 3].set_title('Возраст выдержки')
    axes[1, 3].set_xlabel('Возраст (дни)')
    axes[1, 3].set_ylabel('Частота')

    sns.histplot(df['Крупный заполнитель (кг/м³)'], kde=True, ax=axes[2, 0], color=c_coral, bins=25, edgecolor='black')
    axes[2, 0].set_title('Крупный заполнитель (щебень/гравий)')
    axes[2, 0].set_xlabel('Крупный заполнитель (кг/м³)')
    axes[2, 0].set_ylabel('Частота')

    sns.histplot(df['Мелкий заполнитель (кг/м³)'], kde=True, ax=axes[2, 1], color=c_slate, bins=25, edgecolor='black')
    axes[2, 1].set_title('Мелкий заполнитель (песок)')
    axes[2, 1].set_xlabel('Мелкий заполнитель (кг/м³)')
    axes[2, 1].set_ylabel('Частота')

    sns.boxplot(data=df, x='Возраст (дни)', y='Прочность на сжатие (МПа)', hue='Возраст (дни)', ax=axes[2, 2], palette='mako', legend=False)
    axes[2, 2].set_title('Прочность по возрасту')
    axes[2, 2].set_xlabel('Возраст (дни)')
    axes[2, 2].set_ylabel('Прочность (МПа)')
    axes[2, 2].tick_params(axis='x', rotation=45)

    sns.scatterplot(data=df, x='Мелкий заполнитель (кг/м³)', y='Крупный заполнитель (кг/м³)', hue='Прочность на сжатие (МПа)',palette='viridis', alpha=0.8, ax=axes[2, 3])
    axes[2, 3].set_title('Баланс мелкого и крупного заполнителей')
    axes[2, 3].set_xlabel('Мелкий заполнитель (кг/м³)')
    axes[2, 3].set_ylabel('Крупный заполнитель (кг/м³)')

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df['Наличие пластификатора'] = np.where(df['Пластификатор (кг/м³)'] > 0, 'С пластификатором', 'Без пластификатора')
    palette_plast = {'Без пластификатора': c_teal, 'С пластификатором': c_accent}

    sns.scatterplot(data=df, x='Цемент (кг/м³)', y='Прочность на сжатие (МПа)', hue='Наличие пластификатора',
                    palette=palette_plast, alpha=0.7, s=50, ax=axes[0])
    sns.regplot(data=df[df['Наличие пластификатора'] == 'Без пластификатора'], x='Цемент (кг/м³)', y='Прочность на сжатие (МПа)', scatter=False, ax=axes[0], color=c_teal)
    sns.regplot(data=df[df['Наличие пластификатора'] == 'С пластификатором'], x='Цемент (кг/м³)', y='Прочность на сжатие (МПа)', scatter=False, ax=axes[0], color=c_accent)
    axes[0].set_title('Зависимость прочности от цемента и добавок')
    axes[0].set_xlabel('Цемент (кг/м³)')
    axes[0].set_ylabel('Прочность на сжатие (МПа)')

    sns.scatterplot(data=df, x='Водоцементное отношение (W/C)', y='Прочность на сжатие (МПа)', hue='Наличие пластификатора',
                    palette=palette_plast, alpha=0.7, s=50, ax=axes[1])
    sns.regplot(data=df[df['Наличие пластификатора'] == 'Без пластификатора'], x='Водоцементное отношение (W/C)', y='Прочность на сжатие (МПа)', scatter=False, ax=axes[1], color=c_teal)
    sns.regplot(data=df[df['Наличие пластификатора'] == 'С пластификатором'], x='Водоцементное отношение (W/C)', y='Прочность на сжатие (МПа)', scatter=False, ax=axes[1], color=c_accent)
    axes[1].set_title('Закон Абрамса: Влияние W/C на прочность')
    axes[1].set_xlabel('Водоцементное отношение (W/C)')
    axes[1].set_ylabel('Прочность на сжатие (МПа)')

    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    df_analysis = df.drop(columns=['Наличие пластификатора'])

    corr_matrix = df_analysis.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='vlag', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Тепловая карта корреляций Пирсона')
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    target_col = 'Прочность на сжатие (МПа)'
    concrete_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    print("\n3. Корреляция признаков с целевой переменной (Прочность на сжатие):")
    print(concrete_corr.round(3).to_string())

    X_features = df_analysis.drop(columns=[target_col])

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_features), columns=X_features.columns)

    vif_scaled = pd.DataFrame()
    vif_scaled["Признак"] = X_scaled.columns
    vif_scaled["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
    vif_scaled = vif_scaled.sort_values(by="VIF", ascending=False).reset_index(drop=True)

    print("\n4. Таблица коэффициентов вздутия дисперсии (VIF):")
    print(vif_scaled.round(3).to_string(index=False))

    X = df_analysis.drop(columns=[target_col, 'Возраст (дни)'])
    y = df_analysis[target_col]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models_orig = {
        'Линейная регрессия (OLS)': LinearRegression(),
        'Гребневая регрессия (Ridge, alpha=10)': Ridge(alpha=10.0, random_state=42),
        'Лассо регрессия (Lasso, alpha=0.1)': Lasso(alpha=0.1, random_state=42)
    }

    results_orig = {}
    preds_orig = {}
    residuals_orig = {}

    for name, model in models_orig.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='r2')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
            
        preds_orig[name] = y_pred
        residuals_orig[name] = y_test - y_pred

        results_orig[name] = {
            'R2_CV_mean': float(cv_scores.mean()),
            'R2_CV_std': float(cv_scores.std()),
            'R2_Test': float(r2_score(y_test, y_pred)),
            'RMSE_Test': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'MAPE_Test': float(mean_absolute_percentage_error(y_test, y_pred)),
            'intercept': float(model.intercept_),
            'coefficients': {col: float(coef) for col, coef in zip(feature_names, model.coef_)}
        }

    summary_rows = []
    for name, res in results_orig.items():
        summary_rows.append({
            'Модель': name,
            'R² (CV)': f"{res['R2_CV_mean']:.4f} ± {res['R2_CV_std']:.4f}",
            'R² (Тест)': round(res['R2_Test'], 4),
            'RMSE (МПа)': round(res['RMSE_Test'], 2),
            'MAPE (%)': round(res['MAPE_Test'] * 100, 2)
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n5. Результаты обучения моделей на исходных стандартизированных признаках:")
    print(summary_df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (name, res) in zip(axes, residuals_orig.items()):
        ax.scatter(preds_orig[name], res, alpha=0.6, color='steelblue', edgecolor='k')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Нулевой уровень остатков')
            
        clean_name = name.split('(')[0].strip() 
        ax.set_title(f'Остатки: {clean_name}', fontsize=12, pad=10)
            
        ax.set_xlabel('Прогнозируемая прочность (МПа)', fontsize=10)
        ax.set_ylabel('Остатки', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Анализ остатков и гетероскедастичности (Исходные признаки)", fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show_plots:
        plt.show()
    plt.close()

    pca_full = PCA().fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)

    pca_df = pd.DataFrame({
            'Главная компонента': [f'ГК {i+1}' for i in range(len(eigenvalues))],
            'Собственное значение (Lambda)': eigenvalues,
            'Доля дисперсии': exp_var_ratio,
            'Накопленная дисперсия': cum_var
    })
    print("\n6. Результаты факторного анализа (PCA)")
    print(pca_df.round(4).to_string(index=False))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x_coords = range(1, len(eigenvalues) + 1)

    line1 = ax1.plot(x_coords, eigenvalues, 'bo-', linewidth=2, markersize=7, label=r'Собственное значение ($\lambda$)')
    line_kaiser = ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.8, label=r'Критерий Кайзера ($\lambda = 1.0$)')
    ax1.set_xlabel("Номер главной компоненты", fontsize=11)
    ax1.set_ylabel(r"Собственное значение ($\lambda$)", color='blue', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x_coords)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    bars = ax2.bar(x_coords, exp_var_ratio, alpha=0.35, color='teal', width=0.4, label='Индивидуальная доля дисперсии')

    line2 = ax2.plot(x_coords, cum_var, 'o-', linewidth=2, markersize=7, color='firebrick', label='Кумулятивная дисперсия')
    line_thresh = ax2.axhline(0.85, color='darkorange', linestyle='--', linewidth=1.8, label='Порог 85% дисперсии')

    ax2.set_ylabel("Доля объясненной дисперсии", color='firebrick', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.set_ylim(0, 1.05)

    for i, ratio in enumerate(exp_var_ratio):
        ax2.text(i + 1, ratio + 0.02, f"{ratio*100:.1f}%", ha='center', va='bottom', fontsize=9, color='teal', fontweight='bold')

    handles = [line1[0], line_kaiser, bars, line2[0], line_thresh]
    labels = [r'Собственное значение ($\lambda$)', r'Критерий Кайзера ($\lambda = 1.0$)', 
            'Индивидуальная доля', 'Кумулятивная дисперсия', 'Порог 85% дисперсии']

    ax1.legend(handles, labels, loc='center right', fontsize=9.5)

    plt.title("Факторный анализ (PCA): График каменистой осыпи и объясненная дисперсия", fontsize=13, pad=15)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()
    
    k_opt = 5

    pca_opt = PCA(n_components=k_opt).fit(X_train_scaled)
    pca_feature_names = [f'ГК {i+1}' for i in range(k_opt)]
    loadings = pd.DataFrame(
        pca_opt.components_.T,
        columns=pca_feature_names,
        index=feature_names
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
    plt.title(f"Факторные нагрузки первых {k_opt} главных компонент")
    plt.ylabel("Исходные признаки")
    plt.xlabel("Главные компоненты")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    X_train_pca = pd.DataFrame(pca_opt.transform(X_train_scaled), columns=pca_feature_names)
    X_test_pca = pd.DataFrame(pca_opt.transform(X_test_scaled), columns=pca_feature_names)

    models_pca = {
        'Линейная регрессия (на ГК)': LinearRegression(),
        'Гребневая регрессия (Ridge на ГК, alpha=10)': Ridge(alpha=10.0, random_state=42),
        'Лассо регрессия (Lasso на ГК, alpha=0.1)': Lasso(alpha=0.1, random_state=42)
    }

    results_pca = {}
    preds_pca = {}
    residuals_pca = {}

    for name, model in models_pca.items():
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring='r2')
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
            
        preds_pca[name] = y_pred
        residuals_pca[name] = y_test - y_pred

        results_pca[name] = {
            'R2_CV_mean': float(cv_scores.mean()),
            'R2_CV_std': float(cv_scores.std()),
            'R2_Test': float(r2_score(y_test, y_pred)),
            'RMSE_Test': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'MAPE_Test': float(mean_absolute_percentage_error(y_test, y_pred)),
            'intercept': float(model.intercept_),
            'coefficients': {col: float(coef) for col, coef in zip(pca_feature_names, model.coef_)}
        }

    summary_pca_rows = []
    for name, res in results_pca.items():
        summary_pca_rows.append({
            'Модель (PCA)': name,
            'R² (CV)': f"{res['R2_CV_mean']:.4f} ± {res['R2_CV_std']:.4f}",
            'R² (Тест)': round(res['R2_Test'], 4),
            'RMSE (МПа)': round(res['RMSE_Test'], 2),
            'MAPE (%)': round(res['MAPE_Test'] * 100, 2)
        })

    summary_pca_df = pd.DataFrame(summary_pca_rows)

    print("\n7. Обучение моделей регрессии на главных компонентах")
    print(summary_pca_df.to_string(index=False))

    # Графики остатков (PCA)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, residuals_pca.items()):
        ax.scatter(preds_pca[name], res, alpha=0.6, color='seagreen', edgecolor='k')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Нулевой уровень остатков')
        ax.set_title(f'Остатки: {name}')
        ax.set_xlabel('Прогнозируемая прочность (МПа)')
        ax.set_ylabel('Остатки (y_true - y_pred)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Анализ остатков и гетероскедастичности (Главные компоненты)", fontsize=14, y=1.02)
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    summary_rows = []

    for name, res in results_orig.items():
        short_name = name.split('(')[0].strip()
        summary_rows.append({
            'Модель': short_name,
            'Пространство признаков': f'Исходные ({len(feature_names)})',
            'R² (CV)': res['R2_CV_mean'],
            'R² (Тест)': res['R2_Test'],
            'RMSE (Тест, МПа)': res['RMSE_Test'],
            'MAPE (Тест, %)': res['MAPE_Test'] * 100
        })

    for name, res in results_pca.items():
        short_name = name.split('(')[0].strip()
        summary_rows.append({
            'Модель': short_name,
            'Пространство признаков': f'PCA ({k_opt} ГК)',
            'R² (CV)': res['R2_CV_mean'],
            'R² (Тест)': res['R2_Test'],
            'RMSE (Тест, МПа)': res['RMSE_Test'],
            'MAPE (Тест, %)': res['MAPE_Test'] * 100
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df_sorted = summary_df.sort_values(by='R² (Тест)', ascending=False).reset_index(drop=True)

    print("\n8. Итоговая сравнительная таблица всех моделей")
    print(summary_df_sorted.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ('R² (Тест)', 'Коэффициент детерминации $R^2$', '.3f'),
        ('RMSE (Тест, МПа)', 'Ошибка RMSE (МПа)', '.2f'),
        ('MAPE (Тест, %)', 'Ошибка MAPE (%)', '.2f')
    ]

    for ax, (metric, title, fmt) in zip(axes, metrics):
        sns.barplot(
            data=summary_df, 
            x='Модель', 
            y=metric, 
            hue='Пространство признаков', 
            ax=ax, 
            palette='Set2', 
            edgecolor='black'
        )
        
        ax.set_title(f"Сравнение: {title}", fontsize=12, pad=12)
        ax.set_xlabel("Модель", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        loc_legend = 'lower right' if 'R²' in metric else 'upper right'
        ax.legend(title='Признаки', loc=loc_legend)

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height != 0:
                ax.annotate(
                    f'{height:{fmt}}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=9, xytext=(0, 3),
                    textcoords='offset points'
                )

    plt.suptitle("Сравнительный анализ моделей (Исходные признаки vs PCA)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show_plots:
        plt.show()
    plt.close()

    weights_df = pd.DataFrame({
    'Признак': feature_names,
    'Linear': models_orig['Линейная регрессия (OLS)'].coef_,
    'Ridge': models_orig['Гребневая регрессия (Ridge, alpha=10)'].coef_,
    'Lasso': models_orig['Лассо регрессия (Lasso, alpha=0.1)'].coef_
    })

    print("\n9. Веса моделей на стандартизированных признаках")
    print(weights_df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()