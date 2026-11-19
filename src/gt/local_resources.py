"""Консервативные оценки рабочих буферов; это не измерение OOM."""


def estimated_scaling_peak_mb(method, n, d, heads, layers, degree):
    base = 512 + n * (d * 4 + degree * 16) / 2**20
    if method == 'full':
        # Attention, softmax, backward и сохранённые активации всех слоёв.
        working = n * n * heads * 4 * (4 * layers + 4)
    elif method.startswith('sparse'):
        interaction_degree = degree + (15 if method == 'sparse-expander' else 1)
        working = n * interaction_degree * d * 4 * (6 * layers + 4)
    elif method == 'linear':
        working = n * d * 4 * (24 * layers + 8)
    elif method == 'hop-token':
        working = n * d * 4 * 12 + 2048 * d * 4 * 5 * (24 * layers + 8)
    else:
        raise ValueError(method)
    return base + working / 2**20


def separate_scaling_estimates_mb(method, n, d, heads, layers, degree):
    """Раздельные консервативные оценки CPU и CUDA; не измеренный предел OOM."""
    host = 2048 + n * (d * 4 + degree * 32) * 4 / 2**20
    if method == 'hop-token':
        host += n * d * 4 * 5 * 3 / 2**20  # tokens и временные CPU тензоры
    if method == 'full':
        gpu = 512 + (n*n*heads*4*(2*layers+3) + n*d*4*32) / 2**20
    else:
        gpu = estimated_scaling_peak_mb(method, n, d, heads, layers, degree)
    return dict(host=host, gpu=gpu)
