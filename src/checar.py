"""Checagem rápida do ambiente: bibliotecas + acesso ao FTP do DATASUS.

Uso:
    python -m src.checar
"""
import ftplib
import importlib
import sys

from . import config

LIBS = ["numpy", "pandas", "pyarrow", "duckdb", "datasus_dbc", "requests",
        "sklearn", "xgboost", "lightgbm", "statsmodels", "matplotlib"]


def main() -> None:
    print(f"Python {sys.version.split()[0]}")
    ok = True
    for nome in LIBS:
        try:
            m = importlib.import_module(nome)
            print(f"  [ok] {nome} {getattr(m, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [FALTA] {nome}: {e}")

    print(f"\nTestando FTP {config.FTP_HOST} ...")
    try:
        with ftplib.FTP(config.FTP_HOST, timeout=60) as ftp:
            ftp.login()
            ftp.cwd(config.FTP_DIR_SIH)
            arq = f"RD{config.UF}{config.ANO_INI % 100:02d}01.dbc"
            print(f"  [ok] conectado; {arq} tem {ftp.size(arq) / 1e6:.1f} MB")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [ERRO] FTP: {e}")
        print("  Verifique internet/firewall. Redes de empresa/faculdade às vezes "
              "bloqueiam FTP; tente em casa ou no 4G.")

    print("\nTudo certo! Pode rodar o pipeline." if ok
          else "\nCorrija os itens acima antes de seguir.")


if __name__ == "__main__":
    main()
