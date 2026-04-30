import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Carica credenziali
load_dotenv()
DB_URL = os.getenv("DB_URL")

# Fix per SQLAlchemy (asyncpg non piace a pandas, usiamo psycopg2 per l'export)
# Se DB_URL è "postgresql+asyncpg://...", lo trasformiamo in "postgresql://..."
SYNC_DB_URL = DB_URL.replace("+asyncpg", "")

def export_data():
    print("📥 Scaricamento dati dal DB...")
    engine = create_engine(SYNC_DB_URL)
    
    try:
        # 1. Scarica Tokens
        df_tokens = pd.read_sql("SELECT * FROM tokens", engine)
        df_tokens.to_csv("tokens_new.csv", index=False)
        print(f"✅ tokens_new.csv salvato ({len(df_tokens)} righe)")

        # 2. Scarica Prezzi
        df_prices = pd.read_sql("SELECT * FROM market_data ORDER BY timestamp ASC", engine)
        df_prices.to_csv("prices_new.csv", index=False)
        print(f"✅ prices_new.csv salvato ({len(df_prices)} righe)")

        # 3. Scarica Holders
        df_holders = pd.read_sql("SELECT * FROM holders_snapshot", engine)
        df_holders.to_csv("holders_new.csv", index=False)
        print(f"✅ holders_new.csv salvato ({len(df_holders)} righe)")

    except Exception as e:
        print(f"❌ Errore export: {e}")
        print("Suggerimento: Installa psycopg2 se manca: pip install psycopg2-binary")

if __name__ == "__main__":
    export_data()