# NetStream ML

Датасеты, признаки, модели и оценка для детектора сетевых атак NetStream. Данные берутся из ClickHouse стека `deploy` (размеченные представления `labeled_flows` и `labeled_dns`), обученные модели используются онлайн-детектором.

## Состав

| Путь | Назначение |
|---|---|
| `src/netstream_ml/` | пакет и CLI `netstream-ml`; `cli.py` — только парсинг аргументов и то, что отличает один `baseline`-алгоритм от другого (сам fit/tuning), `baseline_cli.py` — общий для всех четырёх жизненный цикл команды (загрузка датасета, метрики по сплитам/сценариям, отчёт, tracking, save/register/export) |
| `tests/` | тесты |
| `data/` | выгрузки датасетов (не в git) |
| `artifacts/` | модели и отчёты (не в git) |

## Разработка

Нужны Python 3.12+ и [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
just all
```

- `just format` форматирует и правит `ruff`.
- `just lint` запускает `ruff`, проверку форматирования и `mypy --strict`.
- `just test` запускает `pytest`.

## Документация

| Документ | Что внутри |
|---|---|
| [`docs/datasets.md`](docs/datasets.md) | Выгрузка датасета из ClickHouse, разбиение на train/val/test, построение оконных признаков |
| [`docs/baselines.md`](docs/baselines.md) | Все четыре baseline-модели (пороги, Isolation Forest, Random Forest, Gradient Boosting) и фреймворк их сравнения (PR-кривые, разброс по seed, тест Макнемара) |
| [`docs/experiments.md`](docs/experiments.md) | Три эксперимента на устойчивость (чувствительность к интенсивности, перенос на внешний датасет, смещение эмуляция/реальный инструмент) и отслеживание прогонов в MLflow |
| [`docs/online-detector.md`](docs/online-detector.md) | Онлайн-детектор: надёжность Kafka-конвейера, сохранение/регистрация/подпись модели, экспорт в ONNX, `detect run`, формат `DetectionResult` |
