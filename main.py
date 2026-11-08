"""Compare four classifiers on the approved UCI glioma dataset."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo


def load_data(refresh=False):
    """Use the saved UCI data, or download it if needed. Do not clean it here."""
    data_folder = Path(__file__).resolve().parent / 'data'
    data_folder.mkdir(exist_ok=True)
    csv_path = data_folder / 'glioma.csv'
    if csv_path.exists() and not refresh:
        data = pd.read_csv(csv_path)
        X = data.drop(columns='Grade')
        y = data['Grade']
    else:
        dataset = fetch_ucirepo(id=759)
        X = dataset.data.features.copy()
        y = dataset.data.targets['Grade'].copy()
        pd.concat([X, y], axis=1).to_csv(csv_path, index=False)
        dataset.variables.to_csv(data_folder / 'variable_dictionary.csv', index=False)
    # Selecting one column makes y a Series rather than a two-dimensional table.
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Grade should contain only 0 (LGG) and 1 (GBM).')
    if not X.index.equals(y.index):
        raise ValueError('The input and target rows must stay in the same order.')

    return X, y


def inspect_data(X, y):
    """Show the basic checks before deciding whether anything needs cleaning."""
    data = pd.concat([X, y], axis=1)
    print('FIRST FIVE ROWS')
    print(data.head().to_string(index=False))
    print('\nDIMENSIONS')
    print('Input features:', X.shape)
    print('Target:', y.shape)
    print('\nCOLUMN NAMES AND DATA TYPES')
    print(data.dtypes.to_string())
    print('\nDESCRIPTIVE STATISTICS')
    print(data.describe(include='all').to_string())
    print('\nMISSING VALUES')
    print(data.isna().sum().to_string())
    # Check whole records, not just X: two people can share input values.
    print('\nDUPLICATE CHECKS')
    print('Repeated full records (including Grade):', data.duplicated().sum())
    print('Repeated input rows (not including Grade):', X.duplicated().sum())
    print('These counts alone do not prove that the same person appears twice.')
    print('\nCLASS COUNTS: 0 = LGG, 1 = GBM')
    print(y.value_counts().sort_index().to_string())
    print('\nCLASS PERCENTAGES')
    print((y.value_counts(normalize=True).sort_index() * 100).round(2).to_string())
    print('\nRACE VALUES')
    print(X['Race'].value_counts(dropna=False).to_string())
    print('\nGENDER VALUES')
    print(X['Gender'].value_counts(dropna=False).to_string())
    print('\nGENE RESPONSE CODES')
    # The other three columns describe age, race, and gender.
    for column in X.columns:
        if column not in ['Age_at_diagnosis', 'Race', 'Gender']:
            print(column, sorted(X[column].unique()))
    print('\nNo rows were removed or changed during this inspection.')


def plot_eda(X, y):
    """Create the six exploration plots and save them in results/figures."""
    folder = Path(__file__).resolve().parent / 'results' / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    data = X.copy()
    data['Class'] = y.map({0: 'LGG', 1: 'GBM'})
    genes = X.columns.drop(['Gender', 'Age_at_diagnosis', 'Race'])
    sns.set_theme(style='whitegrid')

    plt.figure(figsize=(6, 4))
    sns.countplot(data=data, x='Class', order=['LGG', 'GBM'], color='steelblue')
    plt.title('Number of records in each glioma class')
    plt.xlabel('Glioma class')
    plt.ylabel('Number of records')
    plt.tight_layout()
    plt.savefig(folder / 'class_distribution.png', dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=data, x='Age_at_diagnosis', bins=20, color='steelblue')
    plt.title('Age at diagnosis across all records')
    plt.xlabel('Age at diagnosis (years)')
    plt.ylabel('Number of records')
    plt.tight_layout()
    plt.savefig(folder / 'age_distribution.png', dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data, x='Class', y='Age_at_diagnosis',
                order=['LGG', 'GBM'], color='lightblue')
    plt.title('Age at diagnosis by glioma class')
    plt.xlabel('Glioma class')
    plt.ylabel('Age at diagnosis (years)')
    plt.tight_layout()
    plt.savefig(folder / 'age_by_class.png', dpi=150)
    plt.show()
    plt.close()

    # Show the five most common mutations in this dataset. This is for the
    # chart only: no genes are removed from the later model inputs.
    mutation_counts = X[genes].sum().sort_values(ascending=False).head(5)
    plt.figure(figsize=(7, 4))
    mutation_counts.plot.bar(color='steelblue', rot=0)
    plt.title('Five most frequently mutated genes in this dataset')
    plt.xlabel('Gene')
    plt.ylabel('Number of records with mutation = 1')
    plt.tight_layout()
    plt.savefig(folder / 'mutation_frequency.png', dpi=150)
    plt.show()
    plt.close()

    # Exclude Race: text categories do not have a meaningful numerical order.
    # Include Grade to explore relationships, but never use it as a predictor.
    numerical = pd.concat([X.drop(columns='Race'), y], axis=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical.corr(), cmap='coolwarm', vmin=-1, vmax=1,
                center=0, square=True, cbar_kws={'label': 'Pearson correlation'})
    plt.title('Correlations between numerical features and grade')
    plt.xlabel('Feature or grade')
    plt.ylabel('Feature or grade')
    plt.tight_layout()
    plt.savefig(folder / 'correlation_heatmap.png', dpi=150)
    plt.show()
    plt.close()

    # Count mutations only for this scatter plot, not as an extra model feature.
    data['Mutation_count'] = X[genes].sum(axis=1)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x='Age_at_diagnosis', y='Mutation_count',
                    hue='Class', style='Class', hue_order=['LGG', 'GBM'], alpha=0.6)
    plt.title('Age and number of recorded gene mutations by class')
    plt.xlabel('Age at diagnosis (years)')
    plt.ylabel('Number of mutated genes among the 20 recorded genes')
    plt.legend(title='Glioma class')
    plt.tight_layout()
    plt.savefig(folder / 'age_mutation_scatter.png', dpi=150)
    plt.show()
    plt.close()


def prepare_data(X, y):
    """Remove exact repeated records, check age, and encode Race as yes/no columns."""
    data = pd.concat([X, y], axis=1)
    repeated = data.duplicated()
    print('Exact repeated records removed:', repeated.sum())
    # Keep the first copy. Without patient IDs we cannot prove these are the
    # same person, but this avoids identical records entering both split sets.
    data = data.loc[~repeated].copy()
    if data.isna().any().any():
        raise ValueError('Missing values found. Review them before preprocessing.')

    # The IQR rule flags unusually low or high ages. It is a descriptive check,
    # not a reason to remove a person automatically. No age values are changed.
    age = data['Age_at_diagnosis']
    q1 = age.quantile(0.25)
    q3 = age.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    unusual_age = (age < lower) | (age > upper)
    print(f'Age IQR limits: {lower:.2f} to {upper:.2f} years')
    print('Ages flagged by the IQR rule:', unusual_age.sum())
    print('No ages were removed or changed.')

    X_ready = data.drop(columns='Grade')
    y_ready = data['Grade'].copy()
    # These are the four categories listed by UCI, not categories learned from
    # test outcomes. Each becomes a yes/no column; race has no artificial order.
    race_categories = ['white', 'black or african american', 'asian',
                       'american indian or alaska native']
    if not X_ready['Race'].isin(race_categories).all():
        raise ValueError('An unexpected race label needs review.')
    X_ready['Race'] = pd.Categorical(X_ready['Race'], categories=race_categories)
    X_ready = pd.get_dummies(X_ready, columns=['Race'], dtype=int)
    print('Prepared input dimensions:', X_ready.shape)
    print('Prepared class counts:')
    print(y_ready.value_counts().sort_index())
    return X_ready, y_ready


def split_data(X, y, random_state=42, normalize=False):
    """Make a stratified 80/20 split, with optional training-only standardization."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state)
    if normalize:
        # Learn the mean and SD from training rows only, then apply the same
        # changes to test rows. Never fit the scaler on the test set.
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test),
                              columns=X.columns, index=X_test.index)
    # For later CV tuning, use normalize=False here and put StandardScaler in
    # the model pipeline so it is fitted separately inside every training fold.
    return X_train, X_test, y_train, y_test


def build_models(seed):
    """Return the four pipelines and small parameter grids for a fair comparison."""
    # Only Logistic Regression needs scaling. The scaler stays inside the
    # pipeline so each CV fold learns its own training means and standard deviations.
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=seed))]),
        'Random Forest': Pipeline([
            ('model', RandomForestClassifier(random_state=seed, n_jobs=1))]),
        'AdaBoost': Pipeline([
            ('model', AdaBoostClassifier(random_state=seed))]),
        'XGBoost': Pipeline([
            ('model', XGBClassifier(random_state=seed, n_jobs=1,
                                    eval_metric='logloss', tree_method='hist'))])
    }
    # model__ means the parameter belongs to the pipeline's model step.
    grids = {
        'Logistic Regression': {'model__C': [0.1, 1, 10]},
        'Random Forest': {'model__n_estimators': [50, 100],
                          'model__max_depth': [5, None]},
        'AdaBoost': {'model__n_estimators': [30, 60],
                     'model__learning_rate': [0.1, 0.5]},
        'XGBoost': {'model__n_estimators': [50],
                    'model__max_depth': [2, 3],
                    'model__learning_rate': [0.05, 0.1]}
    }
    return models, grids


def evaluate_model(model, X, y):
    """Calculate binary metrics with GBM (1) as the positive class."""
    predicted = model.predict(X)
    return {
        'Accuracy': accuracy_score(y, predicted),
        'Precision': precision_score(y, predicted, pos_label=1, zero_division=0),
        'Recall': recall_score(y, predicted, pos_label=1, zero_division=0),
        'F1': f1_score(y, predicted, pos_label=1, zero_division=0)
    }


def run_experiments(X, y, gene_columns, n_repeats=30):
    """Tune and evaluate all models on the same split in every replicate."""
    folder = Path(__file__).resolve().parent / 'results'
    folder.mkdir(exist_ok=True)
    rows = []
    representative = {}
    for replicate in range(1, n_repeats + 1):
        seed = 99 + replicate  # Reproducible seeds 100 through 129.
        X_train, X_test, y_train, y_test = split_data(X, y, seed, normalize=False)
        # The optional gene-only comparison uses the SAME people and splits.
        feature_sets = {'All features': list(X.columns), 'Mutations only': list(gene_columns)}
        for feature_set, columns in feature_sets.items():
            train = X_train[columns]
            test = X_test[columns]
            models, grids = build_models(seed)
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for name, pipeline in models.items():
                # Use the same folds and F1 scoring for every model. F1 balances
                # precision and recall for GBM; no test scores choose settings.
                search = GridSearchCV(pipeline, grids[name], scoring='f1', cv=folds,
                                      refit=True, n_jobs=1, error_score='raise')
                search.fit(train, y_train)
                model = search.best_estimator_  # Already refitted on all training rows.
                test_scores = evaluate_model(model, test, y_test)
                train_scores = evaluate_model(model, train, y_train)
                row = {'Feature_set': feature_set, 'Model': name,
                       'Replicate': replicate, 'Seed': seed,
                       'Best_CV_F1': search.best_score_,
                       'Best_parameters': str(search.best_params_)}
                for metric in test_scores:
                    row[metric] = test_scores[metric]
                    row['Training_' + metric] = train_scores[metric]
                rows.append(row)
                # Pick the first replicate in advance for confusion matrices.
                # Do not choose the replicate with the best test score.
                if replicate == 1 and feature_set == 'All features':
                    representative[name] = pd.DataFrame({
                        'Original_row': y_test.index, 'Actual': y_test.to_numpy(),
                        'Predicted': model.predict(test)})
                print(f'Replicate {replicate:02d}/{n_repeats}, {feature_set}, '
                      f'{name}: test F1 = {test_scores["F1"]:.4f}', flush=True)
        # Save after each complete replicate so progress is not lost.
        pd.DataFrame(rows).to_csv(folder / 'model_results.csv', index=False)
    for name, predictions in representative.items():
        filename = name.lower().replace(' ', '_') + '_seed100_predictions.csv'
        predictions.to_csv(folder / filename, index=False)
    return pd.DataFrame(rows), representative


def summarize_results(results):
    """Summarize held-out test scores across the 30 repeats, not across CV folds."""
    rows = []
    for feature_set in results['Feature_set'].unique():
        for model in results['Model'].unique():
            selected = results[(results['Feature_set'] == feature_set) &
                               (results['Model'] == model)]
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                values = selected[metric]
                rows.append({'Feature_set': feature_set, 'Model': model, 'Metric': metric,
                             'Mean': values.mean(), 'SD': values.std(ddof=1),
                             'Minimum': values.min(), 'Maximum': values.max()})
    summary = pd.DataFrame(rows)
    folder = Path(__file__).resolve().parent / 'results'
    summary.to_csv(folder / 'summary_results.csv', index=False)
    return summary


def plot_model_results(results, summary, representative):
    """Save comparison charts and clearly labeled first-replicate confusion matrices."""
    folder = Path(__file__).resolve().parent / 'results' / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    main_results = results[results['Feature_set'] == 'All features']
    main_summary = summary[summary['Feature_set'] == 'All features']
    models = main_results['Model'].unique()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for metric, ax in zip(metrics, axes.flat):
        values = main_summary[main_summary['Metric'] == metric].set_index('Model').loc[models]
        ax.bar(models, values['Mean'], yerr=values['SD'], capsize=4, color='steelblue')
        ax.set_title(f'Test {metric}: mean +/- sample SD')
        ax.set_ylabel(metric + ' score')
        ax.set_xlabel('Model')
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', labelrotation=20)
    fig.suptitle('All features: 30-replicate comparison')
    fig.tight_layout()
    fig.savefig(folder / 'model_mean_scores.png', dpi=150)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for metric, ax in zip(metrics, axes.flat):
        sns.boxplot(data=main_results, x='Model', y=metric, order=models,
                    color='lightblue', ax=ax)
        ax.set_title(f'Test {metric} across 30 splits')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric + ' score')
        ax.tick_params(axis='x', labelrotation=20)
    fig.tight_layout()
    fig.savefig(folder / 'model_score_boxplots.png', dpi=150)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for name, ax in zip(models, axes.flat):
        predictions = representative[name]
        ConfusionMatrixDisplay.from_predictions(
            predictions['Actual'], predictions['Predicted'], labels=[0, 1],
            display_labels=['LGG (0)', 'GBM (1)'], cmap='Blues',
            values_format='d', colorbar=False, ax=ax)
        ax.grid(False)
        ax.set_title(name + ': seed 100 only')
    fig.suptitle('Illustrative confusion matrices: replicate 1, 168 test records')
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrices_seed100.png', dpi=150)
    plt.show()
    plt.close(fig)

    means = main_summary.pivot(index='Model', columns='Metric', values='Mean')
    plt.figure(figsize=(7, 4))
    sns.heatmap(means.loc[models, metrics], annot=True, fmt='.3f',
                cmap='Blues', vmin=0, vmax=1, cbar_kws={'label': 'Mean test score'})
    plt.title('All features: mean test performance over 30 replicates')
    plt.xlabel('Metric')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(folder / 'model_summary_heatmap.png', dpi=150)
    plt.show()
    plt.close()

    # Secondary comparison: all clinical + mutation features versus 20 genes only.
    f1 = summary[summary['Metric'] == 'F1']
    f1.pivot(index='Model', columns='Feature_set', values='Mean').loc[models].plot.bar(
        figsize=(9, 5), rot=20)
    plt.title('Secondary comparison: mean test F1 across the same 30 splits')
    plt.xlabel('Model')
    plt.ylabel('Mean GBM F1 score')
    plt.ylim(0, 1.05)
    plt.legend(title='Input features')
    plt.tight_layout()
    plt.savefig(folder / 'clinical_mutation_comparison.png', dpi=150)
    plt.show()
    plt.close()


def main():
    """Run the project from data inspection through the full model comparison."""
    X, y = load_data()
    inspect_data(X, y)
    plot_eda(X, y)
    X_ready, y_ready = prepare_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_ready, y_ready)
    print('Training and test input shapes:', X_train.shape, X_test.shape)
    print(pd.DataFrame({'Training': y_train.value_counts(),
                        'Testing': y_test.value_counts()}).sort_index())
    gene_columns = X.columns.drop(['Gender', 'Age_at_diagnosis', 'Race'])
    results, representative = run_experiments(X_ready, y_ready, gene_columns)
    summary = summarize_results(results)
    print(summary.to_string(index=False))
    plot_model_results(results, summary, representative)


if __name__ == '__main__':
    main()
