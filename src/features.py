"""Etapa 5 - Engenharia de atributos temporais SEM vazamento.

Regra de ouro: para prever y no mês t (origem t-1), só entra informação
disponível até t-1. Por isso todo atributo derivado de y ou de variáveis
observadas usa shift(>=1) DENTRO de cada unidade (groupby), e janelas móveis
são calculadas sobre a série já deslocada: shift(1).rolling(w).

Atributos de calendário (mês, ano) e o indicador de pandemia são conhecidos
antecipadamente e podem usar o próprio mês t.

Horizonte h (estratégia DIRETA): para prever h meses à frente, a origem é
t-h. Todos os deslocamentos ganham +(h-1), de modo que "y_lag1" passa a ser
o último valor observado na origem. Assim um mesmo conjunto de nomes de
atributos serve para qualquer horizonte, sem previsões recursivas.
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
                    exogenas_lag=("int_ansiedade",), h: int = 1) -> pd.DataFrame:
    if h < 1:
        raise ValueError("h deve ser >= 1")
    d = h - 1  # deslocamento extra por causa do horizonte
    df = painel.sort_values(["unidade", "mes"]).copy()
    g = df.groupby("unidade", sort=False)

    # --- defasagens do alvo (relativas à origem t-h)
    for k in lags:
        df[f"y_lag{k}"] = g["y"].shift(k + d)

    # --- janelas móveis e expansiva (sempre sobre o passado da origem)
    passado = g["y"].shift(1 + d)
    gp = passado.groupby(df["unidade"], sort=False)
    for w in janelas:
        df[f"y_media{w}"] = gp.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"y_dp{w}"] = gp.transform(lambda s, w=w: s.rolling(w).std())
    df["y_media_exp"] = gp.transform(lambda s: s.expanding().mean())

    # --- diferenças disponíveis na origem (Equação 4 do texto)
    df["y_diff1"] = df["y_lag1"] - df["y_lag2"]
    df["y_diff12"] = df["y_lag1"] - g["y"].shift(13 + d)

    # --- baseline sazonal ingênuo: mesmo mês do ano anterior (h <= 12)
    df["y_sazonal"] = g["y"].shift(12) if h <= 12 else np.nan

    # --- outras séries (multivariado), sempre defasadas
    for col in exogenas_lag:
        if col in df:
            for k in (1, 12):
                df[f"{col}_lag{k}"] = g[col].shift(k + d)
    for col in ("leitos_sm_sus", "leitos_tot_sus"):
        if col in df:
            df[f"{col}_lag{DEFASAGEM_LEITOS}"] = g[col].shift(DEFASAGEM_LEITOS + d)
    if "pop" in df:
        df["pop_lag"] = g["pop"].shift(DEFASAGEM_POP + d)

    # --- calendário do mês-alvo (conhecido de antemão)
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
