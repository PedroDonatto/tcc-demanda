"""Etapa 2 - Leitos (CNES, arquivos LT) -> leitos por município e mês.

Uso:
    python -m src.ingest_cnes

É a mesma informação da base "Hospitais e Leitos" do portal de dados abertos,
mas baixada direto do FTP do DATASUS, mês a mês, no mesmo formato do SIH.

Saída (data/processed):
  - cnes_leitos_municipio.parquet : município x mês com
      leitos_sm_exist / leitos_sm_sus  (códigos em config.CODLEITOS_SAUDE_MENTAL)
      leitos_tot_exist / leitos_tot_sus (todos os leitos)
  - cnes_codleito_contagem.csv : quantos leitos por CODLEITO (para conferir
      no dicionário quais códigos são de saúde mental)
"""
from __future__ import annotations

import pandas as pd

from . import config, datasus, io_utils

COLUNAS_LT = ["CNES", "CODUFMUN", "TP_LEITO", "CODLEITO",
              "QT_EXIST", "QT_SUS", "COMPETEN"]


def agregar_lt(df: pd.DataFrame, ano: int, mes: int) -> pd.DataFrame:
    """Agrega um arquivo LT bruto em município x código de leito."""
    df = df.copy()
    df["munic"] = df["CODUFMUN"].str[:6]
    df["codleito"] = df["CODLEITO"].str.zfill(2)
    for c in ("QT_EXIST", "QT_SUS"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    out = (df.groupby(["munic", "codleito"], as_index=False)
             .agg(qt_exist=("QT_EXIST", "sum"), qt_sus=("QT_SUS", "sum")))
    out["mes"] = pd.Timestamp(year=ano, month=mes, day=1)
    return out


def processar_competencia(ano: int, mes: int) -> bool:
    nome = f"LT{config.UF}{ano % 100:02d}{mes:02d}.dbc"
    cache = config.INTERIM / "cnes_lt" / nome.replace(".dbc", ".parquet")
    if io_utils.existe(cache):
        return True
    destino = config.RAW / "cnes_lt" / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    caminho = datasus.baixar_ftp(config.FTP_DIR_CNES_LT, nome, destino)
    if caminho is None:
        return False
    df = datasus.ler_dbc(caminho, colunas=COLUNAS_LT)
    io_utils.salvar(agregar_lt(df, ano, mes), cache)
    if config.APAGAR_DBC_APOS_PROCESSAR:
        caminho.unlink(missing_ok=True)
    return True


def montar_painel(partes: pd.DataFrame) -> pd.DataFrame:
    sm = partes["codleito"].isin(config.CODLEITOS_SAUDE_MENTAL)
    tot = partes.groupby(["munic", "mes"], as_index=False).agg(
        leitos_tot_exist=("qt_exist", "sum"), leitos_tot_sus=("qt_sus", "sum"))
    sm_ag = partes[sm].groupby(["munic", "mes"], as_index=False).agg(
        leitos_sm_exist=("qt_exist", "sum"), leitos_sm_sus=("qt_sus", "sum"))
    painel = tot.merge(sm_ag, on=["munic", "mes"], how="left")
    painel[["leitos_sm_exist", "leitos_sm_sus"]] = (
        painel[["leitos_sm_exist", "leitos_sm_sus"]].fillna(0))
    return painel


def main() -> None:
    lista = list(datasus.competencias(config.ANO_INI, 1, config.ANO_FIM, 12))
    print(f"CNES-LT {config.UF}: {len(lista)} competências")
    faltantes = []
    for i, (a, m) in enumerate(lista, 1):
        print(f"[{i}/{len(lista)}] {a}-{m:02d}")
        if not processar_competencia(a, m):
            faltantes.append(f"{a}-{m:02d}")

    partes = pd.concat([io_utils.ler(p) for p in
                        sorted((config.INTERIM / "cnes_lt").glob("*.parquet"))],
                       ignore_index=True)

    contagem = (partes.groupby("codleito")["qt_exist"].sum()
                      .sort_values(ascending=False))
    contagem.to_csv(config.PROCESSED / "cnes_codleito_contagem.csv")
    print("\nCódigos de leito mais frequentes (confira no dicionário do CNES):")
    print(contagem.head(25).to_string())
    print(f"Considerados saúde mental: {config.CODLEITOS_SAUDE_MENTAL}")

    painel = montar_painel(partes)
    io_utils.salvar(painel, config.PROCESSED / "cnes_leitos_municipio.parquet")
    print(f"\nOK: {len(painel):,} linhas município x mês")
    if faltantes:
        print("AVISO: competências não encontradas:", ", ".join(faltantes))


if __name__ == "__main__":
    main()
