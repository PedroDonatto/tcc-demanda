"""Etapa 5 - Engenharia de atributos temporais SEM vazamento.

Regra de ouro: para prever y no mês t (origem t-1), só entra informação
disponível até t-1. Por isso todo atributo derivado de y ou de variáveis
observadas usa shift(>=1) DENTRO de cada unidade (groupby), e janelas móveis
são calculadas sobre a série já deslocada: shift(1).rolling(w).

Atributos de calendário (mês, ano) e o indicador de pandemia são conhecidos
antecipadamente e podem usar o próprio mês t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)
JANELAS = (3, 6, 12)
# leitos do CNES são publicados com atraso: usamos 2 meses de defasagem
DEFASAGEM_LEITOS = 2
# população anual: usa a do ano anterior (já publicada na origem)
DEFASAGEM_POP = 12


def criar_atributos(painel: pd.DataFrame, lags=LAGS, janelas=JANELAS,
                    exogenas_lag=("int_ansiedade",)) -> pd.DataFrame:
    df = painel.sort_values(["unidade", "mes"]).copy()
    g = df.groupby("unidade", sort=False)

    # --- defasagens do alvo
    for k in lags:
        df[f"y_lag{k}"] = g["y"].shift(k)

    # --- janelas móveis e expansiva (sempre sobre o passado)
    passado = g["y"].shift(1)
    gp = passado.groupby(df["unidade"], sort=False)
    for w in janelas:
        df[f"y_media{w}"] = gp.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"y_dp{w}"] = gp.transform(lambda s, w=w: s.rolling(w).std())
    df["y_media_exp"] = gp.transform(lambda s: s.expanding().mean())

    # --- diferenças disponíveis na origem t-1 (Equação 4 do texto)
    df["y_diff1"] = df["y_lag1"] - df["y_lag2"]
    df["y_diff12"] = df["y_lag1"] - g["y"].shift(13)

    # --- outras séries (multivariado), sempre defasadas
    for col in exogenas_lag:
        if col in df:
            for k in (1, 12):
                df[f"{col}_lag{k}"] = g[col].shift(k)
    for col in ("leitos_sm_sus", "leitos_tot_sus"):
        if col in df:
            df[f"{col}_lag{DEFASAGEM_LEITOS}"] = g[col].shift(DEFASAGEM_LEITOS)
    if "pop" in df:
        df["pop_lag"] = g["pop"].shift(DEFASAGEM_POP)

    # --- calendário (conhecido de antemão)
    mes = df["mes"].dt.month
    df["mes_sin"] = np.sin(2 * np.pi * (mes - 1) / 12)
    df["mes_cos"] = np.cos(2 * np.pi * (mes - 1) / 12)
    df["ano"] = df["mes"].dt.year
    df["pandemia"] = ((df["mes"] >= "2020-03-01")
                      & (df["mes"] <= "2021-12-01")).astype(int)
    df["unidade_cod"] = df["unidade"].astype("category").cat.codes

    # regra uniforme para o início da série: descarta linhas sem histórico
    # suficiente para o maior lag (não preenche com valores futuros)
    df = df[df[f"y_lag{max(lags)}"].notna() & df["y_diff12"].notna()]
    return df.reset_index(drop=True)


def colunas_atributos(df: pd.DataFrame) -> list[str]:
    """Colunas de entrada do modelo (tudo que foi criado aqui)."""
    prefixos = ("y_lag", "y_media", "y_dp", "y_diff", "int_ansiedade_lag",
                "leitos_", "pop_lag", "mes_sin", "mes_cos", "ano",
                "pandemia", "unidade_cod")
    return [c for c in df.columns
            if c.startswith(prefixos)
            and c not in ("leitos_sm_sus", "leitos_tot_sus")]
