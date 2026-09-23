# NetStream ML

Датасеты, признаки, модели и оценка для детектора сетевых атак NetStream. Данные берутся из ClickHouse стека `deploy` (размеченные представления `labeled_flows` и `labeled_dns`), обученные модели используются онлайн-детектором.

## Состав

| Путь | Назначение |
|---|---|
| `src/netstream_ml/` | пакет и CLI `netstream-ml` |
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
