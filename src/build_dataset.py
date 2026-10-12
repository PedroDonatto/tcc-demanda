"""Etapa 4 - Junta SIH + CNES + IBGE num painel mensal.

Uso:
    python -m src.build_dataset            # nível padrão: região imediata
    python -m src.build_dataset estado     # série única do estado
    python -m src.build_dataset municipio  # município (muitos zeros!)

Por que região? Internação por transtorno de humor num município pequeno é
quase sempre 0 ou 1 por mês. Agregar em regiões imediatas do IBGE dá séries
com sinal, mantendo vários "painéis" para o modelo aprender.

Saída: data/processed/painel_<nivel>.parquet, uma linha por unidade x mês:
  unidade, mes, y (alvo: internações do grupo config.ALVO),
  int_<grupo> para cada grupo, urgencias, dias_perm,
  leitos_sm_sus, leitos_tot_sus, pop, taxa_100k
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config, io_utils

NIVEIS = {"regiao": "reg_imediata_id", "estado": None, "municipio": "munic"}


def montar_painel(sih: pd.DataFrame, cnes: pd.DataFrame, mun: pd.DataFrame,
                  pop: pd.DataFrame, nivel: str = "regiao") -> pd.DataFrame:
    if nivel not in NIVEIS:
        raise ValueError(f"nível inválido: {nivel}. Use {list(NIVEIS)}")
    chave = NIVEIS[nivel]
    mapa = mun[["munic", "reg_imediata_id"]].copy()
    mapa["estado"] = str(config.UF_IBGE)
    if chave is None:
        chave = "estado"

    def para_unidade(df: pd.DataFrame, col_munic: str) -> pd.DataFrame:
        df = df.merge(mapa, left_on=col_munic, right_on="munic",
                      how="inner", suffixes=("", "_m"))  # descarta fora da UF
        return df.rename(columns={chave: "unidade"})

    # --- SIH: internações por grupo (colunas int_<grupo>)
    s = para_unidade(sih, "munic_res")
    ints = s.pivot_table(index=["unidade", "mes"], columns="grupo",
                         values="internacoes", aggfunc="sum", fill_value=0)
    ints.columns = [f"int_{c}" for c in ints.columns]
    alvo = s[s["grupo"] == config.ALVO].groupby(["unidade", "mes"]).agg(
        urgencias=("urgencias", "sum"), dias_perm=("dias_perm", "sum"))
    base = ints.join(alvo, how="left")

    # --- grade completa: toda unidade x todo mês (sem internação = 0)
    meses = pd.date_range(f"{config.ANO_INI}-01-01", f"{config.ANO_FIM}-12-01",
                          freq="MS")
    unidades = sorted(mapa[chave].dropna().unique())
    grade = pd.MultiIndex.from_product([unidades, meses],
                                       names=["unidade", "mes"])
    painel = base.reindex(grade).fillna(0).reset_index()
    for g in config.GRUPOS_CID:
        if f"int_{g}" not in painel:
            painel[f"int_{g}"] = 0
    painel["y"] = painel[f"int_{config.ALVO}"]

    # --- CNES: leitos (NÃO preenche com zero: ausência = sem informação)
    c = para_unidade(cnes, "munic")
    leitos = c.groupby(["unidade", "mes"], as_index=False)[
        ["leitos_sm_sus", "leitos_tot_sus"]].sum()
    painel = painel.merge(leitos, on=["unidade", "mes"], how="left")

    # --- IBGE: população anual
    p = para_unidade(pop, "munic")
    p = p.groupby(["unidade", "ano"], as_index=False)["pop"].sum()
    painel["ano"] = painel["mes"].dt.year
    painel = painel.merge(p, on=["unidade", "ano"], how="left").drop(columns="ano")
    painel["taxa_100k"] = painel["y"] / painel["pop"] * 1e5

    painel["unidade"] = painel["unidade"].astype(str)
    return painel.sort_values(["unidade", "mes"]).reset_index(drop=True)


def main() -> None:
    nivel = sys.argv[1] if len(sys.argv) > 1 else "regiao"
    P = config.PROCESSED
    painel = montar_painel(
        io_utils.ler(P / "sih_mensal_municipio.parquet"),
        io_utils.ler(P / "cnes_leitos_municipio.parquet"),
        io_utils.ler(P / "ibge_municipios.parquet"),
        io_utils.ler(P / "ibge_populacao.parquet"),
        nivel,
    )
    io_utils.salvar(painel, P / f"painel_{nivel}.parquet")
    print(f"painel_{nivel}: {painel['unidade'].nunique()} unidades x "
          f"{painel['mes'].nunique()} meses = {len(painel):,} linhas")
    print(f"meses com y = 0: {(painel['y'] == 0).mean():.1%}")
    print(f"leitos ausentes: {painel['leitos_sm_sus'].isna().mean():.1%}")
    print(painel.groupby(painel["mes"].dt.year)["y"].sum().to_string())


if __name__ == "__main__":
    main()
