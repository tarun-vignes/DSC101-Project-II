"""Step 2: load and inspect the approved glioma dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    """Download the official UCI data and save a local copy without cleaning it."""
    dataset = fetch_ucirepo(id=759)
    X = dataset.data.features.copy()
    # Selecting one column makes y a Series rather than a two-dimensional table.
    y = dataset.data.targets['Grade'].copy()
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Grade should contain only 0 (LGG) and 1 (GBM).')
    if not X.index.equals(y.index):
        raise ValueError('The input and target rows must stay in the same order.')

    # Save relative to this script so it also works when run from another folder.
    data_folder = Path(__file__).resolve().parent / 'data'
    data_folder.mkdir(exist_ok=True)
    data = pd.concat([X, y], axis=1)
    data.to_csv(data_folder / 'glioma.csv', index=False)
    dataset.variables.to_csv(data_folder / 'variable_dictionary.csv', index=False)
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


if __name__ == '__main__':
    X, y = load_data()
    inspect_data(X, y)
