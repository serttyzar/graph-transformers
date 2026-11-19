"""Показать результаты курсовой из сохранённых CSV. Нужен только Python 3."""
import csv
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent / "experiments"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def show_quality(directory, title, percent=False):
    rows = read_rows(EXPERIMENTS / directory / "summary.csv")
    count = sum(int(row["n_runs"]) for row in rows)
    print(f"\n{title} · число обучений — {count}")
    for row in sorted(rows, key=lambda item: float(item["test_mean"]), reverse=percent):
        mean = float(row["test_mean"])
        score = f"{mean * 100:.2f} %" if percent else f"{mean:.4f}"
        print(f"  {row['variant']:<19} {score.replace('.', ',')}")


def show_scaling():
    rows = read_rows(EXPERIMENTS / "03_scaling" / "runs.csv")
    measured = sum(row["status"] == "ok" for row in rows)
    print(f"\nМасштабирование · {measured} измерений из {len(rows)} записей")
    medians = read_rows(EXPERIMENTS / "03_scaling" / "medians.csv")
    print("  Вершины    Оператор            Время прохода")
    for size in (5000, 20000, 100000):
        for row in medians:
            if int(row["n_nodes"]) == size and row["method"] in ("full", "sparse-expander"):
                time = row["fwd_bwd_time_s"]
                value = f"{float(time) * 1000:.3f} мс".replace(".", ",") if time else "пропущен по оценке памяти"
                print(f"  {size:>7}    {row['method']:<18}  {value}")


if __name__ == "__main__":
    show_quality("01_zinc", "ZINC · средняя абсолютная ошибка")
    show_quality("02_arxiv", "ogbn-arxiv · точность", percent=True)
    show_scaling()
