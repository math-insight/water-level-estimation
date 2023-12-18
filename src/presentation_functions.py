def faded_color_vectors(initial_color: tuple[float, float, float], n: int) -> list[tuple[float, float, float]]:
    step_size = 1.0 / n
    faded_red_vectors: list[tuple[float, float, float]] = [
        (
            (round(max(0.0, initial_color[0] - i * step_size), 2),
            round(max(0.0, initial_color[1] - i * step_size), 2),
            round(max(0.0, initial_color[2] - i * step_size), 2))
        )
        for i in range(n)
    ]
    return faded_red_vectors

