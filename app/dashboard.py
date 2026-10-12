"""Dashboard do TCC - previsão de internações por transtornos de humor (SP).

Uso (na pasta do projeto, com o .venv ativo):
    streamlit run app/dashboard.py

Lê só os arquivos já gerados (data/processed e outputs). Não baixa nada e não
treina nada: para atualizar, rode antes `python -m src.prever`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC, OUT, MODELS = ROOT / "data" / "processed", ROOT / "outputs", ROOT / "models"
TOTAL = "Estado de SP (total)"
COR = {"hist": "#3b3b3b", "prev": "#2a7de1", "real": "#d9480f",
       "base": "#9aa0a6"}

st.set_page_config(page_title="Demanda de internações – Saúde mental SP",
                   page_icon="🧠", layout="wide")


# ------------------------------------------------------------------ dados
@st.cache_data
def carregar():
    d = {}
    d["painel"] = pd.read_parquet(PROC / "painel_regiao.parquet")
    mun = pd.read_parquet(PROC / "ibge_municipios.parquet")
    d["nomes"] = (mun.drop_duplicates("reg_imediata_id")
                     .set_index("reg_imediata_id")["reg_imediata"].to_dict())
    for nome in ("previsao_total", "previsao_regiao", "backtest_horizontes",
                 "metricas_regiao", "importancias_regiao"):
        arq = OUT / f"{nome}.csv"
        d[nome] = (pd.read_csv(arq, parse_dates=["mes"] if nome.startswith(
            "previsao") else None) if arq.exists() else None)
    meta = MODELS / "meta.json"
    d["meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return d


try:
    D = carregar()
except FileNotFoundError as e:
    st.error(f"Arquivo não encontrado: {e}. Rode o pipeline antes "
             "(rodar_tudo.bat ou os comandos do README).")
    st.stop()

painel, nomes = D["painel"], D["nomes"]
prev_tot, prev_reg = D["previsao_total"], D["previsao_regiao"]
tem_previsao = prev_tot is not None and prev_reg is not None

modelos_prev = []
if tem_previsao:
    modelos_prev = [c for c in prev_tot.columns
                    if f"{c}_lo" in prev_tot.columns]


def melhor_modelo() -> str | None:
    bt = D["backtest_horizontes"]
    if bt is None or not modelos_prev:
        return modelos_prev[0] if modelos_prev else None
    ml = bt[bt["modelo"].isin(modelos_prev)]
    return ml.groupby("modelo")["erro_pct_total"].mean().idxmin()


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("Filtros")
    regioes = sorted(nomes.items(), key=lambda kv: kv[1])
    opcoes = [TOTAL] + [n for _, n in regioes]
    escolha = st.selectbox("Território", opcoes)
    unidade = None if escolha == TOTAL else next(
        k for k, v in nomes.items() if v == escolha)
    modelo = None
    if modelos_prev:
        padrao = melhor_modelo()
        modelo = st.selectbox("Modelo da previsão", modelos_prev,
                              index=modelos_prev.index(padrao))
        st.caption(f"Melhor no backtest: **{padrao}**")
    meses_hist = st.slider("Meses de histórico no gráfico", 12, 132, 36, 12)
    st.divider()
    st.caption("Fonte: DATASUS (SIH-SUS, CNES) e IBGE. Internações SUS com "
               "diagnóstico principal F30–F39, por residência do paciente.")

# ------------------------------------------------------------------ séries do território
if unidade is None:
    hist = painel.groupby("mes", as_index=False)[["y", "pop"]].sum()
    prev = prev_tot if tem_previsao else None
else:
    hist = painel.loc[painel["unidade"] == str(unidade), ["mes", "y", "pop"]]
    prev = (prev_reg[prev_reg["unidade"].astype(str) == str(unidade)]
            if tem_previsao else None)
hist = hist.sort_values("mes")

st.title("🧠 Demanda de internações por transtornos de humor")
st.markdown(f"**{escolha}** · internações SUS (CID-10 F30–F39) · "
            f"dados até {hist['mes'].max():%m/%Y}")

# ------------------------------------------------------------------ indicadores
ult12 = hist.tail(12)["y"].sum()
ant12 = hist.iloc[-24:-12]["y"].sum()
pop_ult = hist["pop"].iloc[-1]
c1, c2, c3, c4 = st.columns(4)
if prev is not None and modelo:
    p1 = prev.sort_values("mes").iloc[0]
    c1.metric(f"Previsão {p1['mes']:%m/%Y}", f"{p1[modelo]:,.0f}".replace(",", "."),
              help="Faixa de 80%: há 8 em 10 chances do valor real cair "
                   "dentro dela, segundo os erros do backtest.")
    c1.caption(f"faixa provável: {p1[f'{modelo}_lo']:,.0f} a "
               f"{p1[f'{modelo}_hi']:,.0f}".replace(",", "."))
else:
    c1.metric("Previsão", "—")
    c1.caption("rode `python -m src.prever`")
c2.metric("Internações (últimos 12 meses)", f"{ult12:,.0f}".replace(",", "."),
          f"{(ult12 / ant12 - 1) * 100:+.1f}% vs 12 meses anteriores"
          if ant12 else None)
c3.metric("Taxa por 100 mil hab./ano",
          f"{ult12 / pop_ult * 1e5:,.1f}".replace(".", ",") if pop_ult else "—")
bt = D["backtest_horizontes"]
if bt is not None and modelo:
    e1 = bt[(bt["modelo"] == modelo) & (bt["h"] == 1)]
    if unidade is None:
        c4.metric("Erro médio do modelo (1 mês à frente)",
                  f"{e1['erro_pct_total'].iloc[0]:.1f}%".replace(".", ","))
    else:
        c4.metric("Erro médio por região (1 mês à frente)",
                  f"±{e1['MAE_regiao'].iloc[0]:.1f} internações")

aba1, aba2, aba3, aba4 = st.tabs(
    ["📈 Previsão", "🗓️ Histórico", "🤖 Modelos", "ℹ️ Sobre"])

# ------------------------------------------------------------------ aba previsão
with aba1:
    fig = go.Figure()
    h = hist.tail(meses_hist)
    fig.add_trace(go.Scatter(x=h["mes"], y=h["y"], name="Histórico",
                             line=dict(color=COR["hist"], width=2)))
    if prev is not None and modelo:
        p = prev.sort_values("mes")
        fig.add_trace(go.Scatter(
            x=pd.concat([p["mes"], p["mes"][::-1]]),
            y=pd.concat([p[f"{modelo}_hi"], p[f"{modelo}_lo"][::-1]]),
            fill="toself", fillcolor="rgba(42,125,225,0.18)",
            line=dict(width=0), name="Faixa de 80%", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p["mes"], y=p[modelo], name=f"Previsão ({modelo})",
                                 mode="lines+markers",
                                 line=dict(color=COR["prev"], width=2.5)))
        if "real" in p and p["real"].notna().any():
            for status, simbolo in (("completo", "circle"), ("parcial", "x")):
                r = p[p["status"] == status]
                if len(r):
                    fig.add_trace(go.Scatter(
                        x=r["mes"], y=r["real"], mode="markers",
                        name=f"Real ({status})",
                        marker=dict(color=COR["real"], size=10, symbol=simbolo)))
    fig.update_layout(height=430, hovermode="x unified",
                      yaxis_title="Internações por mês",
                      legend=dict(orientation="h", y=-0.15),
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if prev is not None and modelo:
        st.subheader("Previsão mês a mês")
        t = prev.sort_values("mes")[["mes", modelo, f"{modelo}_lo", f"{modelo}_hi"]].copy()
        t.columns = ["Mês", "Previsto", "Mínimo provável", "Máximo provável"]
        if "real" in prev:
            t["Real"] = prev.sort_values("mes")["real"].to_numpy()
            t["Situação do dado real"] = prev.sort_values("mes")["status"].to_numpy()
            t["Dentro da faixa?"] = [
                ("✅" if lo <= r <= hi else "❌") if pd.notna(r) else "—"
                for r, lo, hi in zip(t["Real"], t["Mínimo provável"],
                                     t["Máximo provável"])]
        t["Mês"] = t["Mês"].dt.strftime("%m/%Y")
        st.dataframe(t.round(0), hide_index=True, use_container_width=True)
        st.caption("“Parcial”: o SUS ainda pode lançar internações desse mês "
                   "(AIHs chegam com atraso de até alguns meses).")
    else:
        st.info("Ainda não há previsão. Rode `python -m src.prever` e recarregue.")

# ------------------------------------------------------------------ aba histórico
with aba2:
    col_a, col_b = st.columns(2)
    saz = hist.assign(m=hist["mes"].dt.month).groupby("m")["y"].mean()
    nomes_mes = ["jan", "fev", "mar", "abr", "mai", "jun",
                 "jul", "ago", "set", "out", "nov", "dez"]
    fig_s = go.Figure(go.Bar(x=nomes_mes, y=saz.values, marker_color=COR["prev"]))
    fig_s.update_layout(height=320, title="Média de internações por mês do ano",
                        margin=dict(l=10, r=10, t=40, b=10))
    col_a.plotly_chart(fig_s, use_container_width=True)

    anual = hist.assign(ano=hist["mes"].dt.year).groupby("ano")["y"].sum()
    fig_a = go.Figure(go.Bar(x=anual.index.astype(str), y=anual.values,
                             marker_color=COR["hist"]))
    fig_a.update_layout(height=320, title="Total por ano",
                        margin=dict(l=10, r=10, t=40, b=10))
    col_b.plotly_chart(fig_a, use_container_width=True)
    st.caption("A queda em 2020–2021 coincide com a pandemia de COVID-19.")

    if unidade is None:
        st.subheader("Regiões com maior taxa (últimos 12 meses)")
        u12 = painel[painel["mes"] > painel["mes"].max() - pd.DateOffset(months=12)]
        rk = u12.groupby("unidade").agg(y=("y", "sum"), pop=("pop", "last"))
        rk["taxa"] = rk["y"] / rk["pop"] * 1e5
        rk["Região"] = rk.index.map(lambda u: nomes.get(u, u))
        rk = rk.sort_values("taxa", ascending=False).head(15)
        fig_r = go.Figure(go.Bar(x=rk["taxa"], y=rk["Região"], orientation="h",
                                 marker_color=COR["real"],
                                 customdata=rk["y"],
                                 hovertemplate="%{y}: %{x:.1f} por 100 mil "
                                               "(%{customdata:.0f} internações)"))
        fig_r.update_layout(height=480, yaxis=dict(autorange="reversed"),
                            xaxis_title="Internações por 100 mil habitantes",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_r, use_container_width=True)

# ------------------------------------------------------------------ aba modelos
with aba3:
    st.markdown("**Como os modelos foram testados:** para cada mês de teste, "
                "o modelo é treinado do zero só com o passado e prevê o futuro. "
                "Nenhum dado do mês previsto é usado no treino.")
    if bt is not None:
        st.subheader("Erro no total do estado por distância da previsão")
        fig_b = go.Figure()
        for m, g in bt.groupby("modelo"):
            fig_b.add_trace(go.Scatter(
                x=g["h"], y=g["erro_pct_total"], name=m, mode="lines+markers",
                line=dict(dash="dot" if m in ("Ingenuo", "Sazonal_ingenuo")
                          else "solid")))
        fig_b.update_layout(height=380, xaxis_title="Meses à frente (h)",
                            yaxis_title="Erro médio (%)",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_b, use_container_width=True)
        st.caption("Linhas pontilhadas: chutes simples (repetir o último mês; "
                   "repetir o mesmo mês do ano anterior).")
    met = D["metricas_regiao"]
    if met is not None:
        st.subheader("Comparação no teste de 2025 (1 mês à frente, por região)")
        st.dataframe(met.round(3), hide_index=True, use_container_width=True)
        st.caption("O R² fica alto até para os chutes simples porque as regiões "
                   "têm tamanhos muito diferentes; compare pelo MAE e RMSE.")
    imp = D["importancias_regiao"]
    if imp is not None:
        st.subheader("O que mais pesa na previsão")
        mod_imp = st.selectbox("Modelo", sorted(imp["modelo"].unique()))
        top = (imp[imp["modelo"] == mod_imp]
               .sort_values("importancia", ascending=False).head(15))
        fig_i = go.Figure(go.Bar(x=top["importancia"], y=top["atributo"],
                                 orientation="h", marker_color=COR["prev"]))
        fig_i.update_layout(height=450, yaxis=dict(autorange="reversed"),
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_i, use_container_width=True)

# ------------------------------------------------------------------ aba sobre
with aba4:
    meta = D["meta"]
    st.markdown(f"""
**Objetivo:** prever quantas internações por transtornos de humor (CID-10
F30–F39) o SUS terá em cada região de São Paulo nos próximos meses, para
apoiar o planejamento de leitos e equipes.

**Dados**
- **SIH-SUS (DATASUS):** internações, contadas pela data da internação e pelo
  município de residência do paciente. Só AIHs novas (exclui continuações de
  longa permanência).
- **CNES:** leitos de psiquiatria e saúde mental (códigos 47 e 87).
- **IBGE:** população e regiões geográficas imediatas (53 em SP).

**Modelos:** Random Forest e LightGBM (gradient boosting), comparados com
chutes simples. A previsão usa só informação disponível na data da previsão:
meses anteriores, médias móveis, sazonalidade, leitos (com 2 meses de atraso)
e população.

**Margem de erro:** a faixa de 80% vem dos erros que o próprio modelo cometeu
no teste com dados passados (backtest).

Modelos treinados até: **{meta.get('ultimo_mes_treino', '—')}** ·
horizontes: **{meta.get('horizontes', '—')} meses**
""")
