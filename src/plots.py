import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from presentation_functions import faded_color_vectors
from scipy.signal import find_peaks, savgol_filter
from matplotlib.dates import DateFormatter, MonthLocator


def make_figure(rows):
    fig, axes = plt.subplots(
        figsize=(36,16*(rows-1)),
        ncols=1,
        nrows=rows,
        gridspec_kw={'hspace': 0.3}
    )
    return fig, axes


def set_plot_properties(ax, title, xlim=None):
    ax.legend(loc='upper right', fontsize=20)
    ax.set_title(title, fontsize=30)
    ax.set_xlabel('Miesiąc', fontsize=15)
    ax.set_ylabel('Poziom wody (cm)', fontsize=15)
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1000)
    months_locator = MonthLocator()
    months_fmt = DateFormatter("%b")
    ax.xaxis.set_major_locator(months_locator)
    ax.xaxis.set_major_formatter(months_fmt)


def multiple_stations_plot(data, stations, years):
    stations_palette = faded_color_vectors((1,0,0), len(stations))

    fig, axes = make_figure(len(years))

    data_grouped = data \
        .groupby(['Date', 'Station'])['B00020S'] \
        .mean() \
        .reset_index()

    for i, year in enumerate(years):
        data_grouped_year = data_grouped[data_grouped['Date'].dt.year == year]
        for j, station in enumerate(stations):
            data_temp = data_grouped[data_grouped['Station'] == station]

            color = stations_palette[j]
            axes[i].plot(
                data_temp['Date'],
                data_temp['B00020S'],
                label=station,
                linewidth=3,
                color=color)

        set_plot_properties(
            ax=axes[i],
            title='Poziom wody w wybranych stacjach w {year}',
            xlim=(data_grouped_year['Date'].min(), data_grouped_year['Date'].max())
        )

    plt.show()


def one_station_plot(data, station, years_per_chart):
    data_station = data[data['Station'] == station]
    data_grouped = data_station \
        .groupby(['Date'])['B00020S'] \
        .mean() \
        .reset_index()
    data_grouped['Year'] = pd.to_datetime(data_grouped['Date'], format='%Y-%m-%d').dt.year
    data_grouped['Date'] = pd.to_datetime(data_grouped['Date'], format='%Y-%m-%d').dt.strftime('%m-%d')

    n_charts = ceil(16/years_per_chart)

    fig, axes = make_figure(n_charts)

    for i in range(n_charts):
        years = range(2008 + i * years_per_chart, 2008 + years_per_chart + i * years_per_chart)
        for j, year in enumerate(years):
            data_year = data_grouped[data_grouped['Year'] == year]

            axes[i].plot(
                data_year['Date'],
                data_year['B00020S'],
                label=year, linewidth=3
            )
            set_plot_properties(
                ax=axes[i],
                title=f'Stany wody w Głogowie w latach {2008 + i * years_per_chart} - {2008 + years_per_chart + i * years_per_chart - 1}'
            )

    plt.show()


def peaks_plot(data, stations, years, prominence):
    stations_palette = faded_color_vectors((1,0,0), len(stations))
    peaks_palette = faded_color_vectors((0,1,1), len(stations))

    fig, axes = make_figure(len(years))

    data_grouped = data \
        .groupby(['Date', 'Station'])['B00020S'] \
        .mean() \
        .reset_index()

    for i, year in enumerate(years):
        data_grouped_year = data_grouped[data_grouped['Date'].dt.year == year]
        for j, station in enumerate(stations):
            data_station = data_grouped_year[data_grouped_year['Station'] == station]
            station_color = stations_palette[j]
            peak_color = peaks_palette[j]

            peaks, _ = find_peaks(data_station['B00020S'], prominence=prominence)

            axes[i].plot(
                data_station['Date'],
                data_station['B00020S'],
                label=station,
                linewidth=3,
                color=station_color
            )

            axes[i].plot(
                data_station['Date'].iloc[peaks],
                data_station['B00020S'].iloc[peaks],
                'o',
                markersize=10,
                color=peak_color
            )

        set_plot_properties(
            ax=axes[i],
            title=f'Peaki poziomu wody w wybranych stacjach w {year}',
            xlim=(data_grouped_year['Date'].min(), data_grouped_year['Date'].max())
        )

    plt.show()


def savgol_plot(data, stations, years, polyorder, window_length):
    mode = 'nearest'
    n_charts = len(years) * len(stations)

    fig, axes = make_figure(n_charts)

    data_grouped = data \
        .groupby(['Date', 'Station'])['B00020S'] \
        .mean() \
        .reset_index()

    for i, year in enumerate(years):
        for j, station in enumerate(stations):
            data_station = data_grouped[(data_grouped['Station'] == station) & (data_grouped['Date'].dt.year == year)]
            data_filtered = savgol_filter(
                x=data_station['B00020S'],
                window_length=window_length,
                polyorder=polyorder,
                mode=mode
            )

            chart_id = len(stations) * i + j

            axes[chart_id].plot(
                data_station['Date'],
                data_station['B00020S'],
                label='Oryginał',
                linewidth=6,
                color='red'
            )
            axes[chart_id].plot(
                data_station['Date'],
                data_filtered,
                label='Przybliżenie',
                linewidth=3,
                color='yellow'
            )

            set_plot_properties(
                ax=axes[chart_id],
                title=f'Wygładzenie wykresu poziomu wody w {station.capitalize()} w {year}',
                xlim=(data_station['Date'].min(), data_station['Date'].max())
            )

    plt.show()


def peaks_savgol_plot(data, stations, years, prominence, polyorder, window_length):
    mode = 'nearest'

    stations_palette = faded_color_vectors((0,0,1), len(stations))
    peaks_palette = faded_color_vectors((0,1,1), len(stations))

    fig, axes = make_figure(len(years))

    data_grouped = data \
        .groupby(['Date', 'Station'])['B00020S'] \
        .mean() \
        .reset_index()

    for i, year in enumerate(years):
        data_grouped_year = data_grouped[data_grouped['Date'].dt.year == year]
        for j, station in enumerate(stations):
            data_station = data_grouped_year[data_grouped_year['Station'] == station]
            data_filtered = savgol_filter(
                x=data_station['B00020S'],
                window_length=window_length,
                polyorder=polyorder,
                mode=mode
            )

            station_color = stations_palette[j]
            peak_color = peaks_palette[j]

            peaks, _ = find_peaks(
                x=data_filtered,
                prominence=prominence
            )

            axes[i].plot(
                data_station['Date'],
                data_station['B00020S'],
                label=station,
                linewidth=3,
                color=station_color
            )

            axes[i].plot(
                data_station['Date'].iloc[peaks],
                data_station['B00020S'].iloc[peaks],
                'o',
                markersize=10,
                color=peak_color
            )

        set_plot_properties(
            ax=axes[i],
            title=f'Peaki poziomu wody w wybranych stacjach w {year}',
            xlim=(data_grouped_year['Date'].min(), data_grouped_year['Date'].max())
        )

    plt.show()