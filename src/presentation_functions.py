from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.pyplot as plt


def faded_color_vectors(initial_color, n):
    step_size = 1.0 / n
    faded_red_vectors = [
        (
            (round(max(0.0, initial_color[0] - i * step_size), 2),
             round(max(0.0, initial_color[1] - i * step_size), 2),
             round(max(0.0, initial_color[2] - i * step_size), 2))
        )
        for i in range(n)
    ]
    return faded_red_vectors


def make_figure(rows):
    fig, axes = plt.subplots(
        figsize=(36, 16*(rows-1)),
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
