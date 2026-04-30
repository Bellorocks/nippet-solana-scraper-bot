import os
import sys
from dotenv import load_dotenv
from solders.keypair import Keypair
from solders.pubkey import Pubkey

# Load environment variables
load_dotenv()

# --- SENSITIVE DATA ---
PRIVATE_KEY_STR = os.getenv("PRIVATE_KEY")
if not PRIVATE_KEY_STR:
    # Optional for scraping, required for trading.
    # We set PAYER_KEYPAIR to None so execution logic can check later if needed.
    PAYER_KEYPAIR = None
else:
    try:
        PAYER_KEYPAIR = Keypair.from_base58_string(PRIVATE_KEY_STR)
    except Exception as e:
        print(f"CRITICAL: Invalid PRIVATE_KEY format. {e}")
        sys.exit(1)

RPC_HTTP_URL = os.getenv("RPC_HTTP_URL")
RPC_WSS_URL = os.getenv("RPC_WSS_URL")

if not RPC_HTTP_URL:
    RPC_ENDPOINT = os.getenv("RPC_ENDPOINT")
    if RPC_ENDPOINT:
        RPC_HTTP_URL = RPC_ENDPOINT
        RPC_WSS_URL = RPC_ENDPOINT.replace("https", "wss").replace("http", "ws")

if not RPC_HTTP_URL or not RPC_WSS_URL:
    print("CRITICAL: RPC URLs not found in .env")
    sys.exit(1)

# --- STRATEGY PARAMETERS ---
ENTRY_SIZE_SOL = 0.2
ENTRY_DELAY_SECONDS = 30
ENTRY_WINDOW_START = 30
ENTRY_WINDOW_END = 120

TP_PERCENT = 100  # +100%
SL_PERCENT = 20   # -20%
TIME_STOP_SECONDS = 300 # 5 Minutes

# Gatekeeper Rules
MAX_TOP_10_PERCENT = 60.0
MAX_SINGLE_HOLDER_PERCENT = 20.0

# --- EXECUTION & JITO ---
JITO_BLOCK_ENGINE_URL = os.getenv("JITO_BLOCK_ENGINE_URL", "https://amsterdam.mainnet.block-engine.jito.wtf")
JITO_TIP_AMOUNT_SOL = 0.01
JITO_TIP_HIGH_CONGESTION_SOL = 0.02
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5", 
    "HgTtJusZe2NtcZEeKS0oZ3mXe1rMPr2d7Qf25Q8d7q9",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopXSjb3uCBBU19ICjWojaw39Dx",
    "DfXygSm4jCyNCyb3qzK698k1rraXS9f8G9stXk7w8c9"
]

MAX_RETRIES = 3

# --- PAPER TRADING / SIMULATION SETTINGS ---
# Set to False ONLY when ready to burn real money
PAPER_TRADING = True 

# Simulation Costs (Hyper-Realistic)
SIM_SLIPPAGE_PCT = 0.05      # 5% Slippage on Entry AND Exit
SIM_JITO_TIP = 0.01          # 0.01 SOL Jito Tip
SIM_TX_FEE = 0.000005        # Solana Base Fee

# --- CONSTANTS ---
WSOL_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")

# --- METEORA DLMM ---
# Ref: Meteora DLMM Mainnet Program ID
METEORA_DLMM_PROGRAM_ID = Pubkey.from_string("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo")
# Ref: Meteora DLMM Whitepaper - Price formula uses basis points
BASIS_POINT_MAX = 10000

# --- JUPITER ULTRA API ---
# Ultra API is faster and gasless, free during beta period
# Ref: https://station.jup.ag/docs/apis/ultra-api
# Note: All Jupiter APIs now require x-api-key header (get free key at portal.jup.ag)
JUPITER_ORDER_URL = "https://api.jup.ag/ultra/v1/order"   # Returns unsigned transaction directly (Quote+Swap combined)
JUPITER_EXECUTE_URL = "https://api.jup.ag/ultra/v1/execute" # Executes signed transaction
JUPITER_PRICE_URL = "https://api.jup.ag/price/v3"  # Price API is separate
JUPITER_API_KEY = os.getenv("JUPITER_API_KEY", "")  # Required for all Jupiter APIs