"""Etapa 3 - IBGE: municípios, regiões e população anual.

Uso:
    python -m src.ingest_ibge

Saídas (data/processed):
  - ibge_municipios.parquet : código (6 e 7 dígitos), nome, região imediata
    e região intermediária (usadas para agregar a série e reduzir zeros)
  - ibge_populacao.parquet  : município x ano (estimativas IBGE/SIDRA 6579;
    anos sem estimativa são interpolados; 2022 vem do Censo, tabela 4709)
"""
from __future__ import annotations

import pandas as pd
import requests

from . import config, io_utils

URL_MUNIC = ("https://servicodados.ibge.gov.br/api/v1/localidades/estados/"
             f"{config.UF_IBGE}/municipios")
URL_SIDRA = "https://apisidra.ibge.gov.br/values"


def _get_json(url: str):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()


def municipios() -> pd.DataFrame:
    linhas = []
    for m in _get_json(URL_MUNIC):
        imed = m.get("regiao-imediata") or {}
        inter = imed.get("regiao-intermediaria") or {}
        linhas.append({
            "cod_ibge7": str(m["id"]),
            "munic": str(m["id"])[:6],  # SIH/CNES usam 6 dígitos
            "nome": m["nome"],
            "reg_imediata_id": str(imed.get("id", "")),
            "reg_imediata": imed.get("nome", ""),
            "reg_intermediaria_id": str(inter.get("id", "")),
            "reg_intermediaria": inter.get("nome", ""),
        })
    return pd.DataFrame(linhas)


def _sidra(tabela: int, variavel: int, periodo: str) -> pd.DataFrame:
    url = (f"{URL_SIDRA}/t/{tabela}/n6/in%20n3%20{config.UF_IBGE}"
           f"/v/{variavel}/p/{periodo}")
    dados = _get_json(url)[1:]  # 1º elemento é o cabeçalho
    df = pd.DataFrame(dados)
    out = pd.DataFrame({
        "munic": df["D1C"].str[:6],
        "ano": pd.to_numeric(df["D3C"], errors="coerce"),
        "pop": pd.to_numeric(df["V"], errors="coerce"),
    })
    return out.dropna()


def populacao() -> pd.DataFrame:
    est = _sidra(6579, 9324, "all")          # estimativas anuais
    try:
        censo = _sidra(4709, 93, "2022")     # Censo 2022
        est = pd.concat([est, censo], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (Censo 2022 indisponível: {e}; será interpolado)")
    est = est.drop_duplicates(["munic", "ano"], keep="last")

    anos = range(config.ANO_INI - 1, config.ANO_FIM + 1)
    grade = pd.MultiIndex.from_product([est["munic"].unique(), anos],
                                       names=["munic", "ano"])
    pop = (est.set_index(["munic", "ano"])["pop"]
              .reindex(grade.union(est.set_index(["munic", "ano"]).index))
              .sort_index())
    pop = (pop.groupby(level="munic")
              .transform(lambda s: s.interpolate(method="linear",
                                                 limit_direction="both")))
    pop = pop.reset_index()
    pop = pop[pop["ano"].isin(anos)]
    pop["ano"] = pop["ano"].astype(int)
    return pop


def main() -> None:
    mun = municipios()
    io_utils.salvar(mun, config.PROCESSED / "ibge_municipios.parquet")
    print(f"Municípios: {len(mun)} | regiões imediatas: "
          f"{mun['reg_imediata_id'].nunique()}")
    pop = populacao()
    io_utils.salvar(pop, config.PROCESSED / "ibge_populacao.parquet")
    print(f"População: {len(pop):,} linhas município x ano")
    print(pop.groupby("ano")["pop"].sum().astype(int).to_string())


if __name__ == "__main__":
    main()
