"""Exporta os resultados para a experiência web (outputs/web_data.json).

Uso:
    python -m src.exportar_web

Junta painel, previsões, backtests e importâncias num único JSON compacto e
baixa do IBGE o mapa dos municípios de SP (GeoJSON, qualidade mínima); a
página junta os municípios em regiões imediatas (mapa munic -> região).
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from . import config, io_utils

URL_MAPA = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/"
            f"{config.UF_IBGE}?formato=application/vnd.geo+json"
            "&intrarregiao=municipio&qualidade=minima")

NOMES_ATRIB = {
    "y_lag1": "Internações do último mês", "y_lag2": "Há 2 meses",
    "y_lag3": "Há 3 meses", "y_lag6": "Há 6 meses",
    "y_lag12": "Mesmo mês do ano anterior", "y_media3": "Média dos últimos 3 meses",
    "y_media6": "Média dos últimos 6 meses", "y_media12": "Média dos últimos 12 meses",
    "y_dp3": "Oscilação (3 meses)", "y_dp6": "Oscilação (6 meses)",
    "y_dp12": "Oscilação (12 meses)", "y_media_exp": "Média de todo o histórico",
    "y_diff1": "Subiu ou caiu no último mês", "y_diff12": "Mudança em 1 ano",
    "int_ansiedade_lag1": "Internações por ansiedade", "int_ansiedade_lag12":
    "Ansiedade há 1 ano", "leitos_sm_sus_lag2": "Leitos de saúde mental",
    "leitos_tot_sus_lag2": "Total de leitos SUS", "pop_lag": "População",
    "mes_sin": "Época do ano (sazonalidade)", "mes_cos": "Época do ano (sazonalidade)",
    "ano": "Ano (tendência)", "pandemia": "Período da pandemia",
    "unidade_cod": "Qual é a região",
}


def r1(x):
    return [None if pd.isna(v) else round(float(v), 1) for v in x]


def main() -> None:
    P, O = config.PROCESSED, config.OUTPUTS
    painel = io_utils.ler(P / "painel_regiao.parquet")
    mun = io_utils.ler(P / "ibge_municipios.parquet")
    nomes = (mun.drop_duplicates("reg_imediata_id")
                .set_index("reg_imediata_id")["reg_imediata"].to_dict())
    pr = pd.read_csv(O / "previsao_regiao.csv", parse_dates=["mes"])
    pt = pd.read_csv(O / "previsao_total.csv", parse_dates=["mes"])
    bt25 = pd.read_csv(O / "previsoes_regiao.csv", parse_dates=["mes"])
    bth = pd.read_csv(O / "backtest_horizontes.csv")
    imp = pd.read_csv(O / "importancias_regiao.csv")
    meta_m = json.loads((config.ROOT / "models" / "meta.json").read_text(
        encoding="utf-8"))

    modelos = [c for c in pt.columns if f"{c}_lo" in pt.columns]
    ml = bth[bth["modelo"].isin(modelos)]
    melhor = ml.groupby("modelo")["erro_pct_total"].mean().idxmin()
    meses = sorted(painel["mes"].unique())
    meses_prev = sorted(pr["mes"].unique())
    mods_bt = [c for c in bt25.columns if c not in ("unidade", "mes", "y")]

    def bloco(hist: pd.DataFrame, prev: pd.DataFrame, b25: pd.DataFrame):
        hist = hist.set_index("mes").reindex(meses)
        u12 = hist.tail(12)
        internacoes12 = u12["y"].sum()
        los = (u12["dias_perm"].sum() / internacoes12) if internacoes12 else None
        prev = prev.set_index("mes").reindex(meses_prev)
        b25 = b25.set_index("mes").sort_index()
        return {
            "y": r1(hist["y"]),
            "pop": round(float(hist["pop"].iloc[-1])) if pd.notna(hist["pop"].iloc[-1]) else None,
            "leitos_sm": r1([hist["leitos_sm_sus"].iloc[-1]])[0],
            "leitos_tot": r1([hist["leitos_tot_sus"].iloc[-1]])[0],
            "los": round(los, 1) if los else None,
            "prev": {m: {"p": r1(prev[m]), "lo": r1(prev[f"{m}_lo"]),
                         "hi": r1(prev[f"{m}_hi"])} for m in modelos},
            "real26": r1(prev["real"]) if "real" in prev else [],
            "status26": [s if isinstance(s, str) else None
                         for s in prev.get("status", [])],
            "bt25": {"meses": [m.strftime("%Y-%m") for m in b25.index],
                     "y": r1(b25["y"]),
                     **{m: r1(b25[m]) for m in mods_bt}},
        }

    regioes = []
    for u, h in painel.groupby("unidade"):
        b = bloco(h, pr[pr["unidade"].astype(str) == str(u)],
                  bt25[bt25["unidade"].astype(str) == str(u)])
        b.update({"id": str(u), "nome": nomes.get(str(u), str(u))})
        regioes.append(b)

    cols = ["y", "dias_perm", "pop", "leitos_sm_sus", "leitos_tot_sus"]
    hist_t = painel.groupby("mes", as_index=False)[cols].sum(min_count=1)
    bt_t = bt25.groupby("mes", as_index=False)[["y"] + mods_bt].sum()
    total = bloco(hist_t, pt, bt_t)
    total.update({"id": "total", "nome": "Estado de São Paulo"})

    imp_top = {}
    for m, g in imp.groupby("modelo"):
        g = g.assign(nome=g["atributo"].map(NOMES_ATRIB).fillna(g["atributo"]))
        g = g.groupby("nome", as_index=False)["importancia"].sum()
        g["importancia"] = g["importancia"] / g["importancia"].sum()
        g = g.sort_values("importancia", ascending=False).head(10)
        imp_top[m] = [[n, round(float(v), 4)] for n, v in
                      zip(g["nome"], g["importancia"])]

    try:
        geo = requests.get(URL_MAPA, timeout=120).json()
        print(f"Mapa IBGE: {len(geo.get('features', []))} municípios")
    except Exception as e:  # noqa: BLE001
        geo = None
        print(f"AVISO: não consegui baixar o mapa do IBGE ({e})")

    dados = {
        "meta": {"gerado_em": datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "ultimo_treino": meta_m.get("ultimo_mes_treino"),
                 "melhor": melhor, "modelos": modelos, "modelos_bt": mods_bt},
        "meses": [pd.Timestamp(m).strftime("%Y-%m") for m in meses],
        "meses_prev": [pd.Timestamp(m).strftime("%Y-%m") for m in meses_prev],
        "total": total, "regioes": regioes,
        "backtest_h": bth.round(2).to_dict(orient="records"),
        "importancias": imp_top, "geo": geo,
        "mun_regiao": dict(zip(mun["cod_ibge7"].astype(str),
                               mun["reg_imediata_id"].astype(str))),
    }
    saida = O / "web_data.json"
    saida.write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")),
                     encoding="utf-8")
    print(f"OK: {saida} ({saida.stat().st_size / 1e3:.0f} KB), "
          f"{len(regioes)} regiões, melhor modelo {melhor}")

    # monta a página da Sala de Situação com os dados embutidos
    modelo_html = config.ROOT / "app" / "web" / "template.html"
    if modelo_html.exists():
        js = json.dumps(dados, ensure_ascii=False, separators=(",", ":"))
        html = modelo_html.read_text(encoding="utf-8").replace(
            "__DADOS__", js.replace("</", "<\\/"))
        destino = config.ROOT / "app" / "web" / "index.html"
        destino.write_text(html, encoding="utf-8")
        print(f"Sala de Situação: {destino} (abra com ver_sala.bat)")


if __name__ == "__main__":
    main()
