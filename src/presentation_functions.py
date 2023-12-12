def faded_color_vectors(initial_color, n):
    step_size = 1.0 / n
    faded_red_vectors = [
        (
            f'rgb({round(max(0, initial_color[0] - i * step_size), 2)},'
            f'{round(max(0, initial_color[1] - i * step_size), 2)},'
            f'{round(max(0, initial_color[2] - i * step_size), 2)})'
        )
        for i in range(n)
    ]
    return faded_red_vectors

