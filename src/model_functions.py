import os
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from itertools import combinations


def get_stations():
    stations = ['GŁOGÓW', 'ŚCINAWA', 'MALCZYCE', 'BRZEG DOLNY', 'OŁAWA',  'BRZEG', 'RACIBÓRZ-MIEDONIA', 'KRZYŻANOWICE',
                'OLZA', 'CHAŁUPKI']
    return stations


def read_data(start_year: int):
    """Zwraca zbiór danych od zadanego roku włącznie."""

    old_url = "../data/B00020S.pkl"
    new_url = f"../data/B00020S_since_{start_year}.pkl"

    stations_offset_values_dict = {
        'GŁOGÓW': 0,
        'ŚCINAWA': 1,
        'MALCZYCE': 1,
        'BRZEG DOLNY': 1,
        'OŁAWA': 2,
        'BRZEG': 2,
        'RACIBÓRZ-MIEDONIA': 3,
        'KRZYŻANOWICE': 3,
        'OLZA': 3,
        'CHAŁUPKI': 3
    }

    stations = get_stations()

    if os.path.exists(new_url):
        data = pd.read_pickle(new_url)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    else:
        data = pd.read_pickle(old_url)
        data['Date'] = pd.to_datetime(data['Date'])

        offset_values = [stations_offset_values_dict[station] for station in stations]

        condition = ((data['Date'].dt.year >= start_year) & (data['Station'].isin(stations)))
        data_filtered = data[condition]

        station_data = (pd.pivot_table(data_filtered, index=['Date'], columns=['Station'], values='B00020S',
                                       aggfunc='mean').reset_index())

        new_column_order = ['Date'] + stations
        station_data = station_data[new_column_order]

        for i, col in enumerate(station_data.columns[1:]):
            station_data[col] = station_data[col].shift(periods=-offset_values[i])

        station_data = station_data.dropna()

        data = station_data

        data.to_pickle(new_url)

        return data


def adjust_data(data: pd.DataFrame, stations_list: [str]):
    """Zwraca zbiór danych o zmienionej liście kolumn."""
    if 'GŁOGÓW' not in stations_list:
        raise Exception("Nie ma Głogowa w liście stacji!")
    columns = ['Date'] + stations_list
    new_data = data[columns]
    return new_data


def get_train_and_test_data(data: pd.DataFrame, train_proportion: float = 0.8):
    """Zwraca zbiór dany treningowy i testowy na podstawie danego zbioru danych i proporcji."""
    split_point = int(len(data) * train_proportion)

    train_data, test_data = data.iloc[:split_point], data.iloc[split_point:]
    return train_data, test_data


def train_model(train_data: pd.DataFrame):
    """Zwraca wytrenowany model liniowy na podstawie danego treningowego zbioru danych."""
    try:
        x = train_data.iloc[:, 2:]
    except IndexError:
        raise (IndexError("Nie wystarczająco kolumn!"))
    x = sm.add_constant(x)
    y = train_data.iloc[:, 1]
    model = sm.OLS(y, x).fit()
    return model


def get_predictions(model, train_data: pd.DataFrame = None, test_data: pd.DataFrame = None):
    """Zwraca predykcje treningowe lub testowe danego modelu."""
    train_predictions = None
    test_predictions = None

    if train_data is not None:
        if len(train_data.columns) < 3:
            raise IndexError("Nie wystarczająca liczba kolumn w zbiorze treningowym!")
        x_train = train_data.iloc[:, 2:]
        x_train = sm.add_constant(x_train)
        train_predictions = model.predict(x_train)

    if test_data is not None:
        if len(test_data.columns) < 3:
            raise IndexError("Nie wystarczająca liczba kolumn w zbiorze testowym!")
        x_test = test_data.iloc[:, 2:]
        x_test = sm.add_constant(x_test)
        test_predictions = model.predict(x_test)

    return train_predictions, test_predictions


def plot_residuals(model, data: pd.DataFrame, train_proportion: float = 0.8):
    """Rysuje resztki modelu"""
    residuals = pd.DataFrame(model.resid)
    train_data, _ = get_train_and_test_data(data, train_proportion)

    fig, axes = plt.subplots(figsize=(36, 32), ncols=1, nrows=2, gridspec_kw={'hspace': 0.3})
    months_locator = MonthLocator()
    months_fmt = DateFormatter("%b")
    years_locator = YearLocator()
    years_fmt = DateFormatter("%Y")

    axes[0].plot(data['Date'][:len(train_data)], residuals.iloc[:, 0], linewidth=3)
    axes[0].set_xlim(data['Date'][:len(train_data)].min(), data['Date'][:len(train_data)].max())
    axes[0].set_xlabel('Data', fontsize=20)
    axes[0].set_title('Reszty', fontsize=30)
    axes[0].grid()
    axes[0].xaxis.set_minor_locator(months_locator)
    axes[0].xaxis.set_minor_formatter(months_fmt)
    axes[0].xaxis.set_major_locator(years_locator)
    axes[0].xaxis.set_major_formatter(years_fmt)
    axes[0].tick_params(axis='both', which='minor', labelsize=15)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    plt.setp(axes[0].xaxis.get_minorticklabels(), rotation=45)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

    residuals.plot(kind='kde', ax=axes[1], linewidth=4)
    axes[1].set_xlim(-100, 100)
    axes[1].set_ylim(0, 0.03)
    axes[1].set_title('Wykres gęstości jądra / rozkładu prawdopodobieństwa reszt', fontsize=30)
    axes[1].set_ylabel('Gęstość', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].grid()

    plt.show()
    print(residuals.describe())


def get_errors(y_true_train, train_predictions, y_true_test, test_predictions):
    """Zwraca słownik ze wszystkimi miarami błędu"""
    mse = mean_squared_error(y_true_test, test_predictions, squared=True)
    rmse = mean_squared_error(y_true_test, test_predictions, squared=False)
    mape = mean_absolute_percentage_error(y_true_test, test_predictions)
    max_absolute_error = max_error(y_true_test, test_predictions)
    max_relative_error = max(abs((y_true_test - test_predictions) / y_true_test))
    train_absolute_error = abs(y_true_train - train_predictions)
    train_relative_error = abs((y_true_train - train_predictions) / y_true_train)
    test_absolute_error = abs(y_true_test - test_predictions)
    test_relative_error = abs((y_true_test - test_predictions) / y_true_test)

    return {
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "train_absolute_error": train_absolute_error,
        "train_relative_error": train_relative_error,
        "test_absolute_error": test_absolute_error,
        "test_relative_error": test_relative_error
    }


def set_common_plot_properties(ax, title, y_label, train_dates, test_dates):
    """Jak sama nazwa mówi."""
    months_locator = MonthLocator()
    months_fmt = DateFormatter("%b")
    years_locator = YearLocator()
    years_fmt = DateFormatter("%Y")

    ax.set_xlabel('Data', fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_xlim(train_dates.min(), test_dates.max())
    ax.set_title(title, fontsize=30)
    ax.legend(loc='upper right', fontsize=20)
    ax.grid()
    ax.xaxis.set_minor_locator(months_locator)
    ax.xaxis.set_minor_formatter(months_fmt)
    ax.xaxis.set_major_locator(years_locator)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.setp(ax.xaxis.get_minorticklabels(), rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def plot_predictions(ax, data, train_predictions, test_predictions, train_data, test_data):
    """Rysuje predykcje modelu."""
    train_dates = data['Date'][:len(train_data)]
    test_dates = data['Date'][len(train_data):len(train_data) + len(test_data)]

    ax.plot(train_dates, train_data['GŁOGÓW'], color='b', label='Faktyczne dane', linewidth=2)
    ax.plot(train_dates, train_predictions, color='r', label='Model', linewidth=2)
    ax.plot(test_dates, test_data['GŁOGÓW'], color='g', label='Faktyczne dane (test)', linewidth=2)
    ax.plot(test_dates, test_predictions, color='m', label='Model (test)', linewidth=2)
    set_common_plot_properties(ax, 'Predykcja poziomu wody w Głogowie', 'Poziom wody (cm)', train_dates, test_dates)


def plot_errors(ax, data, train_error, test_error, train_data, test_data, labels: [str], title: [str], y_label):
    """Rysuje wykresu błędów."""
    train_dates = data['Date'][:len(train_data)]
    test_dates = data['Date'][len(train_data):len(train_data) + len(test_data)]

    ax.plot(train_dates, train_error, color='b', label=labels[0], linewidth=2)
    ax.plot(test_dates, test_error, color='r', label=labels[1], linewidth=2)
    set_common_plot_properties(ax, title, y_label, train_dates, test_dates)


def find_best_model_features(
        data: pd.DataFrame,
        stat: str = 'aic',
        train_proportion: float = 0.8):
    """Zwraca listę najlepszego zbioru stacji pod względem AiC, RMSE lub MAPE (domyślnie AiC)."""

    stations = get_stations()

    if 'GŁOGÓW' in stations:
        stations.remove('GŁOGÓW')

    df_columns = ['model', 'aic', 'rmse', 'mape']
    models_comparison = pd.DataFrame(columns=df_columns)
    for i in range(2, len(stations) + 1):
        for stations_combo in combinations(stations, i):
            columns = ['GŁOGÓW'] + list(stations_combo)
            data_2 = data[columns]

            train_data, test_data = get_train_and_test_data(data_2, train_proportion=train_proportion)
            model = train_model(train_data)

            train_predictions, test_predictions = get_predictions(model, train_data, test_data)

            errors_dict = get_errors(train_data['GŁOGÓW'], train_predictions, test_data['GŁOGÓW'], test_predictions)

            row = {
                'model': list(columns),
                'aic': round(model.aic, 2),
                'rmse': round(errors_dict['rmse'], 3),
                'mape': round(errors_dict['mape'], 4)
            }

            models_comparison.loc[len(models_comparison.index)] = row

    best_features = models_comparison[models_comparison[stat] == models_comparison[stat].min()]['model'].values[0]
    return models_comparison, best_features
