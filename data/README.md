# Data directory

No data file in this tree is version-controlled (see `.gitignore`); only this
document and the `.gitkeep` markers are. Datasets are rebuilt from source by the
ingestion pipeline.

| Directory | Contents | Written by |
|---|---|---|
| `raw/` | Provider responses stored verbatim with a SHA-256, never edited. | `mru_forecast.data.ingestion` |
| `interim/` | Tidy but uncleaned, plus the ingestion manifest and the cross-check sample. | `mru_forecast.data.ingestion` |
| `processed/` | The modelling dataset (Parquet), the bad-tick flag log and the cleaning report. | `mru_forecast.data.cleaning` |
| `external/` | Reserved for macro series ingested from a separate provider. | `mru_forecast.data.ingestion` |

## Files

| File | Written by |
|---|---|
| `interim/rates_interim.parquet` | tidy long frame: one row per (series, date) |
| `interim/ingestion_manifest.json` | provenance: provider, symbol, rate type, row counts, payload checksums |
| `interim/cross_check.csv` | independent second-provider sample used to verify the level |
| `processed/rates_processed.parquet` | wide `date x series` modelling dataset, business-day calendar |
| `processed/cleaning_flags.csv` | every observation removed, with the reason |
| `processed/cleaning_report.json` | per-series cleaning actions and the validation report |

**Provenance rule.** Every file in `processed/` must be reproducible from files in
`raw/` by re-running the pipeline. If it is not, it does not belong here.

Rate types, sources and the reasoning behind every cleaning decision are documented in
[`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

## Rebuild

```bash
python scripts/download_data.py
python scripts/build_dataset.py
```
