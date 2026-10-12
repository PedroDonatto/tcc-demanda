# TCC – Previsão de demanda de internações (F30–F39) em SP

Pipeline ponta a ponta: **SIH-SUS + CNES + IBGE → painel mensal → atributos
sem vazamento → baselines × Random Forest × XGBoost × LightGBM (walk-forward)**.

## Instalação (Windows, uma vez só)

Abra o **cmd** dentro da pasta do projeto e rode:

```
py -V:3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.checar
```

Se o `checar` mostrar tudo `[ok]`, o ambiente está pronto.
Toda vez que abrir um cmd novo, ative o ambiente: `.venv\Scripts\activate`.

## Rodar

Tudo de uma vez: dê dois cliques em **`rodar_tudo.bat`**.
O primeiro download do SIH leva de 30 min a 1 h, e se cair dá pra rodar de novo
que ele continua de onde parou.

Ou etapa por etapa:

| Etapa | Comando | Gera |
|---|---|---|
| 1. Internações | `python -m src.ingest_sih` | `sih_mensal_municipio.parquet` |
| 2. Leitos | `python -m src.ingest_cnes` | `cnes_leitos_municipio.parquet` |
| 3. IBGE | `python -m src.ingest_ibge` | `ibge_municipios`, `ibge_populacao` |
| 4. Painel | `python -m src.build_dataset regiao` | `painel_regiao.parquet` |
| 5. Modelos | `python -m src.train regiao 12` | `outputs/metricas_regiao.csv` + gráfico |
| Banco SQL | `python -m src.db "SELECT * FROM painel_regiao LIMIT 5"` | `data/tcc.duckdb` |

Testes (sem internet): `python -m tests.test_pipeline`

## Decisões metodológicas (para o texto do TCC)

- **Alvo:** internações com diagnóstico principal F30–F39, por **mês de
  internação** (`DT_INTER`), não pela competência de processamento. Baixamos
  6 competências além de dez/2025 para capturar AIHs apresentadas com atraso.
- **Só AIHs novas:** `IDENT = 1`. As AIHs de continuação de longa permanência
  (`IDENT = 5`) são excluídas, porque não representam internação nova. Duplicatas
  de `N_AIH` também saem.
- **Local:** município de **residência** (`MUNIC_RES`) do paciente em SP.
- **Unidade de análise:** região geográfica imediata do IBGE. No nível de
  município a série tem zeros demais.
- **Leitos (CNES):** códigos 47 (psiquiatria) e 87 (saúde mental), que podem
  ser ajustados em `src/config.py`. Entram com **2 meses de defasagem**, pela
  demora na publicação. Meses sem registro ficam como ausentes, não como zero.
- **População:** estimativas IBGE (SIDRA 6579) + Censo 2022 (4709), com
  interpolação linear nos anos faltantes. Entra defasada em 12 meses.
- **Atributos:** lags 1, 2, 3, 6 e 12; médias e desvios móveis de 3, 6 e 12
  meses e média expansiva, sempre com `shift(1)` dentro de cada região;
  diferenças d1 e d12; mês em seno/cosseno; indicador de pandemia (mar/2020 a
  dez/2021); internações por ansiedade (F40–F48) defasadas.
- **Validação:** walk-forward com janela expansiva. Para cada mês de teste, o
  modelo é treinado do zero só com meses anteriores (h = 1).
  O teste `tests/test_pipeline.py` prova que alterar o futuro não muda nenhum
  atributo do passado.

## Configurar

Recorte, período, CIDs e códigos de leito ficam em `src/config.py`.
