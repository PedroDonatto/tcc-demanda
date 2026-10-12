"""Etapa 1 - Internações (SIH-SUS, arquivos RD) -> série mensal por município.

Uso:
    python -m src.ingest_sih

O que faz:
  1. Baixa RD{UF}{AA}{MM}.dbc de cada competência (com cache: se cair a
     internet, é só rodar de novo que continua de onde parou).
  2. Mantém só AIHs com diagnóstico principal nos grupos de config.GRUPOS_CID.
  3. Remove AIHs de continuação (IDENT=5, longa permanência), que não são
     internações novas, e duplicatas de N_AIH.
  4. Agrega por município de RESIDÊNCIA e mês da INTERNAÇÃO (DT_INTER),
     não pelo mês de processamento.

Saídas (data/processed):
  - sih_aih_saude_mental.parquet : uma linha por AIH filtrada
  - sih_mensal_municipio.parquet : município x mês x grupo
"""
from __future__ import annotations

import pandas as pd

from . import config, datasus, io_utils

COLUNAS_RD = [
    "N_AIH", "IDENT", "ANO_CMPT", "MES_CMPT", "MUNIC_RES", "MUNIC_MOV",
    "DIAG_PRINC", "DT_INTER", "DT_SAIDA", "DIAS_PERM", "CAR_INT",
    "SEXO", "IDADE", "COD_IDADE", "VAL_TOT", "ESPEC",
]


def grupo_cid(diag: pd.Series) -> pd.Series:
    """Classifica o diagnóstico principal nos grupos configurados (ou NA)."""
    cid3 = diag.str[:3].str.upper()
    out = pd.Series(pd.NA, index=diag.index, dtype="object")
    for nome, (ini, fim) in config.GRUPOS_CID.items():
        out[(cid3 >= ini) & (cid3 <= fim)] = nome
    return out


def filtrar_rd(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra um arquivo RD bruto para as AIHs de interesse."""
    df = df.copy()
    df["grupo"] = grupo_cid(df["DIAG_PRINC"])
    return df[df["grupo"].notna()].reset_index(drop=True)


def processar_competencia(ano: int, mes: int) -> bool:
    """Baixa e filtra uma competência. Retorna False se não existe no FTP."""
    nome = f"RD{config.UF}{ano % 100:02d}{mes:02d}.dbc"
    cache = config.INTERIM / "sih" / nome.replace(".dbc", ".parquet")
    if io_utils.existe(cache):
        return True

    destino = config.RAW / "sih" / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    caminho = datasus.baixar_ftp(config.FTP_DIR_SIH, nome, destino)
    if caminho is None:
        return False

    df = filtrar_rd(datasus.ler_dbc(caminho, colunas=COLUNAS_RD))
    io_utils.salvar(df, cache)
    print(f"    {nome}: {len(df):,} AIHs de saúde mental")
    if config.APAGAR_DBC_APOS_PROCESSAR:
        caminho.unlink(missing_ok=True)
    return True


def consolidar(partes: list[pd.DataFrame]) -> pd.DataFrame:
    """Junta as competências e prepara a base no nível de AIH."""
    aih = pd.concat(partes, ignore_index=True)

    # IDENT: 1 = AIH principal; 5 = continuação de longa permanência
    aih = aih[aih["IDENT"] == "1"]
    aih = aih.drop_duplicates(subset="N_AIH", keep="first")

    aih["dt_inter"] = pd.to_datetime(aih["DT_INTER"], format="%Y%m%d",
                                     errors="coerce")
    aih = aih[aih["dt_inter"].notna()]
    aih = aih[(aih["dt_inter"].dt.year >= config.ANO_INI)
              & (aih["dt_inter"].dt.year <= config.ANO_FIM)]

    aih["mes"] = aih["dt_inter"].dt.to_period("M").dt.to_timestamp()
    aih["munic_res"] = aih["MUNIC_RES"].str[:6]
    aih["dias_perm"] = pd.to_numeric(aih["DIAS_PERM"], errors="coerce")
    aih["val_tot"] = pd.to_numeric(aih["VAL_TOT"], errors="coerce")
    aih["urgencia"] = (aih["CAR_INT"] == "02").astype(int)
    return aih.reset_index(drop=True)


def agregar_mensal(aih: pd.DataFrame) -> pd.DataFrame:
    return (aih.groupby(["munic_res", "mes", "grupo"], as_index=False)
               .agg(internacoes=("N_AIH", "size"),
                    dias_perm=("dias_perm", "sum"),
                    val_tot=("val_tot", "sum"),
                    urgencias=("urgencia", "sum")))


def main() -> None:
    ano_fim_dl, mes_fim_dl = config.ANO_FIM, 12
    for _ in range(config.MESES_EXTRAS_SIH):
        mes_fim_dl += 1
        if mes_fim_dl == 13:
            ano_fim_dl, mes_fim_dl = ano_fim_dl + 1, 1

    lista = list(datasus.competencias(config.ANO_INI, 1, ano_fim_dl, mes_fim_dl))
    print(f"SIH {config.UF}: {len(lista)} competências "
          f"({config.ANO_INI}-01 a {ano_fim_dl}-{mes_fim_dl:02d})")

    faltantes = []
    for i, (a, m) in enumerate(lista, 1):
        print(f"[{i}/{len(lista)}] {a}-{m:02d}")
        if not processar_competencia(a, m):
            faltantes.append(f"{a}-{m:02d}")
            print("    (não publicado no FTP ainda)")

    partes = [io_utils.ler(p)
              for p in sorted((config.INTERIM / "sih").glob("*.parquet"))]
    aih = consolidar(partes)
    mensal = agregar_mensal(aih)

    io_utils.salvar(aih, config.PROCESSED / "sih_aih_saude_mental.parquet")
    io_utils.salvar(mensal, config.PROCESSED / "sih_mensal_municipio.parquet")

    print("\nResumo por grupo e ano (internações):")
    print(mensal.assign(ano=mensal["mes"].dt.year)
                .pivot_table(index="ano", columns="grupo",
                             values="internacoes", aggfunc="sum"))
    if faltantes:
        print("\nAVISO: competências não encontradas:", ", ".join(faltantes))
        print("Os últimos meses da série podem estar incompletos (AIHs "
              "atrasadas). Considere encerrar a série antes.")


if __name__ == "__main__":
    main()
