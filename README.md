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

## Выгрузка датасета

Нужен запущенный стек `deploy` (`just up` или лаборатория `just lab-up`/`just lab-campaign`) с доступным HTTP-интерфейсом ClickHouse.

```bash
uv run netstream-ml data export data/v1 \
    --url http://127.0.0.1:8123 \
    --campaign-manifest ../deploy/lab/campaigns/manifests/v1-<timestamp>.json
```

Подключение можно задать переменными окружения `CLICKHOUSE_URL`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`, `CLICKHOUSE_DATABASE` вместо флагов.

`labeled_flows` и `labeled_dns` — представления, которые для каждой строки перебирают все окна из `labels`, поэтому на датасете с полсотней запусков и сотней тысяч потоков запрос может упереться в лимит памяти ClickHouse по умолчанию. `data export` сам поднимает `max_memory_usage` до 2 ГиБ (`--max-memory-usage` или `CLICKHOUSE_MAX_MEMORY_USAGE`); если и этого не хватает, поднимать дальше или выносить разметку в материализованную таблицу — отдельная задача для `deploy`.

Команда выгружает `labeled_flows`, `labeled_dns` и `labels` в `flows.parquet`, `dns.parquet` и `labels.parquet` (ClickHouse отдаёт Parquet сам через `FORMAT Parquet`, без промежуточной конвертации) и пишет `manifest.json`: версия `netstream-ml`, версия сервера ClickHouse, список сценариев, число строк и sha256 каждого файла. С `--campaign-manifest` в манифест выгрузки попадает содержимое манифеста кампании (`deploy/lab/campaign.py`) целиком, с его собственной контрольной суммой — так выгрузка прослеживается до конкретного прогона лаборатории, включая git-ревизию агента.
