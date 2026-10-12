"""Configurações centrais do projeto. Mude aqui o recorte (UF, período, CIDs)."""
from pathlib import Path

# ---------------------------------------------------------------- recorte
UF = "SP"            # sigla usada nos arquivos do DATASUS (RDSP, LTSP...)
UF_IBGE = 35         # código IBGE da UF (SP = 35)
ANO_INI = 2015       # primeiro ano da série (mês de internação)
ANO_FIM = 2025       # último ano da série

# Internações são lançadas no SIH pela competência de processamento, que pode
# ser meses depois da data de internação. Baixamos alguns meses além do fim
# para capturar essas AIHs atrasadas.
MESES_EXTRAS_SIH = 6

# Grupos de diagnóstico (CID-10, diagnóstico principal)
GRUPOS_CID = {
    "humor": ("F30", "F39"),      # alvo principal: transtornos de humor
    "ansiedade": ("F40", "F48"),  # análise complementar / variável exógena
}
ALVO = "humor"

# Códigos de leito do CNES considerados "saúde mental".
# 47 = Psiquiatria | 87 = Saúde mental (Portaria SAS/MS 706/2012).
# Confira no dicionário de variáveis do CNES; o ingest_cnes imprime os
# códigos encontrados para você validar.
CODLEITOS_SAUDE_MENTAL = ["47", "87"]

# ---------------------------------------------------------------- FTP DATASUS
FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR_SIH = "/dissemin/publicos/SIHSUS/200801_/Dados"
FTP_DIR_CNES_LT = "/dissemin/publicos/CNES/200508_/Dados/LT"

# ---------------------------------------------------------------- caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"              # .dbc baixados
INTERIM = DATA / "interim"      # parquet filtrado por arquivo (cache)
PROCESSED = DATA / "processed"  # tabelas finais
OUTPUTS = ROOT / "outputs"      # métricas, gráficos
DB_PATH = DATA / "tcc.duckdb"

for _p in (RAW, INTERIM, PROCESSED, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

# Apagar o .dbc depois de processar (economiza ~2 GB de disco)
APAGAR_DBC_APOS_PROCESSAR = True
