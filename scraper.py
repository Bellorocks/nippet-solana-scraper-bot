import asyncio
import os
import sys
import datetime
import traceback
import struct
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, deque

import aiohttp
import numpy as np
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, desc
from loguru import logger
from dotenv import load_dotenv

from models import Base, Token, MarketData, HoldersSnapshot
from rate_limiter import TokenBucketLimiter
from config import JUPITER_API_KEY
import gatekeeper
from math_utils import (
    # V1 Metrics
    gini_coefficient, 
    shannon_entropy, 
    calculate_velocity_acceleration_jerk,
    calculate_holder_dump_percentage,
    # V2: Chaos & Memory Theory
    hurst_exponent,
    lyapunov_exponent,
    fractal_dimension,
    # V2: Statistical Anomaly
    benford_deviation,
    jarque_bera_test,
    calculate_autocorrelation,
    calculate_kurtosis,
    calculate_skewness,
    # V2: Spectral Analysis
    spectral_analysis,
    # V2: Market Microstructure
    kyle_lambda,
    amihud_illiquidity,
    roll_spread,
    # V2: Holder Network
    herfindahl_index,
    top3_vs_rest,
    # V2: Technical Indicators
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_position,
    calculate_vpt,
    # V2: Risk Metrics
    calculate_max_drawdown,
    calculate_var_95,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    # V2: Time Patterns
    calculate_price_age_correlation,
    detect_pump_duration
)

# Load configuration
load_dotenv()

DB_URL = os.getenv("DB_URL")
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
NEW_PAIRS_API = "https://api.geckoterminal.com/api/v2/networks/solana/new_pools?page=1"
METEORA_DLMM_API = "https://dlmm.datapi.meteora.ag"

if not DB_URL:
    logger.error("DB_URL not found in environment variables.")
    sys.exit(1)

# Configure Logging
logger.add("logs/scraper.log", rotation="10 MB", level="INFO")
logger.add("logs/error.log", rotation="2 MB", level="ERROR")

# Database Setup
engine = create_async_engine(DB_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Rate Limiter for RPC (10 requests per second, capacity 10)
rpc_limiter = TokenBucketLimiter(rate=10.0, capacity=10.0)

# =============================================================================
# TICK-BY-TICK TRANSACTION FETCHER (For Institutional Microstructure Metrics)
# =============================================================================
# Fetches actual swap transactions from Meteora DLMM pools to calculate:
# - Kyle's Lambda (price impact per unit volume)
# - Amihud Illiquidity (|return|/volume)
# - VPT (Volume Price Trend)
# - Real buy/sell classification (CVD)

# Meteora DLMM Program ID
METEORA_DLMM_PROGRAM_ID = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo"

# Token transfers need to be tracked to calculate swap amounts
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


class TickData:
    """
    Represents a single tick (trade) for microstructure analysis.
    """
    def __init__(self, timestamp: float, price: float, volume: float, is_buy: bool):
        self.timestamp = timestamp
        self.price = price
        self.volume = volume  # Actual trade volume, not hourly aggregate
        self.is_buy = is_buy  # True if swap X->Y, False if Y->X
    
    def __repr__(self):
        return f"TickData(t={self.timestamp:.0f}, p=${self.price:.8f}, v={self.volume:.2f}, buy={self.is_buy})"


# Tick history storage: {pair_address: deque([TickData, ...])}
tick_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))


async def fetch_meteora_swaps_rpc(
    session: aiohttp.ClientSession,
    pair_address: str,
    limit: int = 50
) -> List[TickData]:
    """
    Fetch recent swap transactions from Meteora DLMM pool via RPC.
    
    This provides TICK-BY-TICK data for accurate microstructure calculations.
    
    Args:
        session: aiohttp session
        pair_address: The DLMM pool address
        limit: Max transactions to fetch (1 RPC call = 1 signature batch)
    
    Returns:
        List of TickData objects with actual trade volumes
    """
    # Step 1: Get transaction signatures for the pool
    payload_sigs = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [
            pair_address,
            {
                "limit": min(limit, 100),  # Max 100 per call
                "commitment": "confirmed"
            }
        ]
    }
    
    try:
        await rpc_limiter.acquire()
        
        async with session.post(RPC_ENDPOINT, json=payload_sigs, timeout=10) as resp:
            if resp.status == 429:
                logger.debug(f"RPC rate limit hit for {pair_address}")
                return []
            
            if resp.status != 200:
                logger.debug(f"RPC error {resp.status} fetching signatures for {pair_address}")
                return []
            
            data = await resp.json()
            signatures = data.get("result", [])
            
            if not signatures:
                return []
    
    except asyncio.TimeoutError:
        logger.debug(f"RPC timeout fetching signatures for {pair_address}")
        return []
    except Exception as e:
        logger.debug(f"RPC error fetching signatures: {type(e).__name__}: {e}")
        return []
    
    # Step 2: Fetch transaction details in batches of 10
    ticks = []
    
    for i in range(0, len(signatures), 10):
        batch = signatures[i:i+10]
        
        batch_payload = [
            {
                "jsonrpc": "2.0",
                "id": j,
                "method": "getTransaction",
                "params": [
                    sig["signature"],
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "confirmed"
                    }
                ]
            }
            for j, sig in enumerate(batch)
        ]
        
        try:
            await rpc_limiter.acquire()
            
            async with session.post(RPC_ENDPOINT, json=batch_payload, timeout=15) as resp:
                if resp.status == 429:
                    await asyncio.sleep(0.5)  # Brief backoff
                    continue
                
                if resp.status != 200:
                    continue
                
                tx_responses = await resp.json()
                
                # Parse each transaction
                for tx_resp in tx_responses:
                    tick = _parse_meteora_swap_transaction(tx_resp, pair_address)
                    if tick:
                        ticks.append(tick)
        
        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.debug(f"Error fetching transaction batch: {type(e).__name__}: {e}")
            continue
    
    return ticks


def _parse_meteora_swap_transaction(tx_resp: dict, pair_address: str) -> Optional[TickData]:
    """
    Parse a Meteora DLMM swap transaction to extract tick data.
    
    Meteora DLMM swaps involve:
    - Instruction to DLMM program (swap)
    - Inner instructions showing token transfers
    - We need to determine direction (X->Y or Y->X) and amounts
    """
    if not tx_resp or "result" not in tx_resp:
        return None
    
    tx = tx_resp["result"]
    if not tx:
        return None
    
    # Skip failed transactions
    meta = tx.get("meta", {})
    if meta.get("err"):
        return None
    
    # Get timestamp
    block_time = tx.get("blockTime")
    if not block_time:
        return None
    timestamp = float(block_time)
    
    # Look for Meteora DLMM swap in instructions
    transaction = tx.get("transaction", {})
    message = transaction.get("message", {})
    instructions = message.get("instructions", [])
    
    is_meteora_swap = False
    
    for ix in instructions:
        if ix.get("programId") == METEORA_DLMM_PROGRAM_ID:
            # Could be swap, initialize, etc. Check parsed data if available
            parsed = ix.get("parsed", {})
            if parsed.get("type") in ["swap", "Swap"]:
                is_meteora_swap = True
                break
    
    if not is_meteora_swap:
        # Check if inner instructions contain DLMM swap
        inner_instructions = meta.get("innerInstructions", [])
        for inner in inner_instructions:
            for ix in inner.get("instructions", []):
                if ix.get("programId") == METEORA_DLMM_PROGRAM_ID:
                    is_meteora_swap = True
                    break
            if is_meteora_swap:
                break
    
    if not is_meteora_swap:
        return None
    
    # Parse token transfers from inner instructions to get amounts
    token_transfers = []
    
    inner_instructions = meta.get("innerInstructions", [])
    for inner in inner_instructions:
        for ix in inner.get("instructions", []):
            parsed = ix.get("parsed", {})
            if parsed.get("type") in ["transfer", "transferChecked"]:
                info = parsed.get("info", {})
                amount = info.get("tokenAmount", {}).get("uiAmount")
                if amount is None:
                    # Try raw amount
                    try:
                        amount = float(info.get("amount", 0))
                        decimals = info.get("tokenAmount", {}).get("decimals", 0)
                        if decimals:
                            amount = amount / (10 ** decimals)
                    except (ValueError, TypeError):
                        amount = 0
                
                if amount and amount > 0:
                    token_transfers.append({
                        "amount": float(amount),
                        "mint": info.get("mint"),
                        "destination": info.get("destination"),
                        "source": info.get("source")
                    })
    
    if len(token_transfers) < 2:
        # Need at least 2 transfers for a swap
        return None
    
    # For a swap, we typically have:
    # - User sends token X to pool
    # - Pool sends token Y to user
    # The larger amount in USD terms is usually what we track
    
    # Sort by amount (largest first)
    token_transfers.sort(key=lambda x: x["amount"], reverse=True)
    
    # Take the larger transfer as the "volume"
    volume = token_transfers[0]["amount"]
    
    # Determine if it's a "buy" (arbitrary for DLMM, but we can use first transfer direction)
    # In DLMM, there's no clear buy/sell - it's just swapping X for Y
    # We'll use True if first transfer is to the pool (user selling), False otherwise
    is_buy = False  # Default, can be refined with pool state analysis
    
    # We don't have price from transaction alone - will be matched with price data later
    # For now, return with volume and timestamp
    return TickData(
        timestamp=timestamp,
        price=0.0,  # Will be populated from price feed
        volume=volume,
        is_buy=is_buy
    )


def calculate_microstructure_metrics_from_ticks(
    pair_address: str,
    price: float,
    prev_price: float
) -> dict:
    """
    Calculate microstructure metrics using ACTUAL tick-by-tick transaction data.
    
    This replaces the inaccurate hourly-volume-based calculations.
    """
    ticks = list(tick_history.get(pair_address, []))
    
    if len(ticks) < 2:
        return {
            "kyle_lambda": None,
            "amihud_illiquidity": None,
            "vpt": None,
            "tick_count": 0,
            "real_volume_1m": None,
        }
    
    # Calculate metrics from recent ticks (last 60 seconds)
    cutoff = asyncio.get_event_loop().time() - 60
    recent_ticks = [t for t in ticks if t.timestamp > cutoff]
    
    if len(recent_ticks) < 2:
        return {
            "kyle_lambda": None,
            "amihud_illiquidity": None,
            "vpt": None,
            "tick_count": len(ticks),
            "real_volume_1m": sum(t.volume for t in ticks[-20:]) if ticks else None,
        }
    
    # Extract arrays
    prices = np.array([t.price if t.price > 0 else price for t in recent_ticks])
    volumes = np.array([t.volume for t in recent_ticks])
    
    # Calculate price changes
    price_changes = np.diff(prices)
    
    # Kyle's Lambda: price impact per unit volume
    kyle = kyle_lambda(price_changes, volumes[1:])
    
    # Amihud Illiquidity: |return|/volume
    returns = np.diff(np.log(prices + 1e-10))
    amihud = amihud_illiquidity(returns, volumes[1:])
    
    # VPT calculation using actual tick volumes
    vpt_val = None
    if len(prices) >= 5 and len(volumes) >= 5:
        vpt_val = calculate_vpt(prices, volumes)
    
    # Real volume in last minute
    real_volume_1m = np.sum(volumes)
    
    return {
        "kyle_lambda": kyle,
        "amihud_illiquidity": amihud,
        "vpt": vpt_val,
        "tick_count": len(recent_ticks),
        "real_volume_1m": float(real_volume_1m),
    }


def _estimate_volume_from_ticks(pair_address: str, token_age_sec: float) -> float:
    """
    Stima il volume orario (1h) dai tick raccolti per token neonati.
    
    Per token appena nati (0-120s), Meteora API non ha ancora il volume_1h.
    Questa funzione calcola una stima basata sulle transazioni reali
    raccolte dal tick_fetcher_task.
    
    Args:
        pair_address: Indirizzo della pool DLMM
        token_age_sec: Età del token in secondi
        
    Returns:
        Volume stimato per 1h, o 0.0 se non ci sono dati sufficienti
    """
    ticks = list(tick_history.get(pair_address, []))
    if not ticks or token_age_sec <= 0:
        return 0.0
    
    # Calcola il periodo di raccolta (dal primo all'ultimo tick)
    timestamps = [t.timestamp for t in ticks]
    volume_total = sum(t.volume for t in ticks)
    
    time_span_sec = max(timestamps) - min(timestamps)
    if time_span_sec < 1:  # Evita divisione per zero
        time_span_sec = token_age_sec
    
    # Estrapola a 1 ora (3600 secondi)
    # Esempio: se in 30s abbiamo 1000 USDC di volume, in 1h = 1000 * (3600/30) = 120k
    volume_1h_estimate = volume_total * (3600.0 / time_span_sec)
    
    # Applica un fattore di confidenza: meno dati = più incertezza
    # Con pochi tick (es. 1-2), la stima è poco affidabile, la riduciamo
    if len(ticks) < 5:
        confidence_factor = len(ticks) / 5.0  # 0.2 con 1 tick, 0.4 con 2 tick...
        volume_1h_estimate *= confidence_factor
    
    return float(volume_1h_estimate)


async def tick_fetcher_task(pair_address: str, base_symbol: str, session: aiohttp.ClientSession):
    """
    Background task: Fetches swap transactions every 15 seconds.
    
    This provides tick-by-tick data for accurate institutional-grade
    microstructure calculations (Kyle's Lambda, Amihud, VPT).
    
    Uses free RPC with rate limiting - fetches ~50 transactions every 15s
    which is sufficient for microstructure analysis on $20k+ liquidity pools.
    """
    logger.info(f"[{base_symbol}] Starting tick fetcher for microstructure analysis")
    
    fetch_count = 0
    
    while True:
        try:
            # Check if we should stop (monitor no longer active)
            if pair_address not in tick_history and fetch_count > 0:
                # Cleanup has been called
                break
            
            # Fetch recent swaps
            ticks = await fetch_meteora_swaps_rpc(session, pair_address, limit=50)
            
            if ticks:
                fetch_count += 1
                
                # Get current price from price history for tick price assignment
                # Find a mint that maps to this pair_address
                current_price = None
                for mint, hist in price_history.items():
                    if hist:
                        current_price = hist[-1][0]  # Latest price
                        break
                
                # Update ticks with current price
                for tick in ticks:
                    if tick.price == 0.0 and current_price:
                        tick.price = current_price
                    tick_history[pair_address].append(tick)
                
                if fetch_count % 4 == 0:  # Log every minute
                    logger.info(
                        f"[{base_symbol}] Tick fetch #{fetch_count}: "
                        f"{len(ticks)} swaps, total history: {len(tick_history[pair_address])}"
                    )
            
        except asyncio.CancelledError:
            logger.debug(f"[{base_symbol}] Tick fetcher cancelled")
            raise
        except Exception as e:
            logger.debug(f"[{base_symbol}] Tick fetch error: {type(e).__name__}: {e}")
        
        # Fetch every 15 seconds (conservative to respect rate limits)
        await asyncio.sleep(15)


def _chunk(lst: list, size: int):
    """Split list into chunks of given size."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# --- ADVANCED ALPHA METRICS V2: Global State ---
# Price history buffer: {mint_address: deque([(price, timestamp), ...], maxlen=100)}
# Extended to 100 for better Hurst/Lyapunov calculations
price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

# Volume history buffer: {mint_address: deque([volume, ...], maxlen=100)}
volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

# Volatility history buffer: {mint_address: deque([(volatility, timestamp), ...], maxlen=20)}
volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))

# Holder tracking state: {mint_address: {top_holder, balance, gini, hhi, top3_ratio, holder_counts, ...}}
holder_state: Dict[str, dict] = {}

# Token creation times: {mint_address: creation_timestamp}
token_creation_times: Dict[str, float] = {}

# Active token monitors: {mint_address: True} - tracks which tokens are being monitored
active_monitors: Dict[str, bool] = {}


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def fetch_new_pairs(session: aiohttp.ClientSession) -> List[dict]:
    """
    Scout Component: Poll GeckoTerminal for new Meteora DLMM pairs.
    
    API Docs: https://api.geckoterminal.com/api/v2/networks/solana/new_pools
    Rate Limit: 30 calls/minute per IP (free tier)
    
    Filters for Meteora DLMM pools only and extracts relevant pair data.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    
    try:
        async with session.get(NEW_PAIRS_API, headers=headers, timeout=15) as response:
            if response.status == 429:
                logger.warning("GeckoTerminal API rate limit (429) - sleeping 60s")
                await asyncio.sleep(60)
                return []
            
            if response.status != 200:
                logger.warning(f"GeckoTerminal API returned {response.status}: {await response.text()}")
                return []
            
            # Check content type
            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                logger.warning(f"GeckoTerminal returned non-JSON content: {content_type}")
                return []

            data = await response.json()
            raw_pools = data.get("data", [])
            
            pairs = []
            for pool in raw_pools:
                attrs = pool.get("attributes", {})
                
                # Filter: Meteora DLMM only (not AMM or other Meteora products)
                # Filter: Meteora DLMM only (not AMM or other Meteora products)
                dex_rels = pool.get("relationships", {}).get("dex", {}) or {}
                dex_data = dex_rels.get("data") or {}
                dex_id = (dex_data.get("id") or "").lower()
                name = attrs.get("name", "").lower()

                # Only accept Meteora DLMM/DAMM variants, reject Meteora AMM
                is_meteora_dlmm = (
                    "meteora-damm" in dex_id or 
                    "meteora-dlmm" in dex_id or
                    ("meteora" in dex_id and "dlmm" in name) or
                    ("meteora" in name and "dlmm" in name)
                )
                
                if not is_meteora_dlmm:
                    continue
                
                # --- FIX CRITICO QUI SOTTO ---
                base_token_rels = pool.get("relationships", {}).get("base_token", {}) or {}
                base_token_data = base_token_rels.get("data") or {}
                raw_id = base_token_data.get("id", "")
                # Rimuove sia SOLANA_ che solana_
                base_token_address = raw_id.replace("SOLANA_", "").replace("solana_", "")
                # -----------------------------
                
                pairs.append({
                    "baseToken": {
                        "address": base_token_address,
                        "symbol": attrs.get("name", "UNKNOWN").split("/")[0].strip()
                    },
                    "pairAddress": attrs.get("address"),
                    "dexId": "meteora",
                    "pairCreatedAt": None, 
                    "liquidity": {"usd": attrs.get("reserve_in_usd")},
                    "fdv": attrs.get("fdv_usd")
                })

            return pairs

    except asyncio.TimeoutError:
        logger.warning("GeckoTerminal API timeout - will retry next cycle")
        return []
    except aiohttp.ClientError as e:
        logger.error(f"GeckoTerminal API network error: {e}")
        return []
    except ValueError as e:
        logger.error(f"GeckoTerminal API JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching new pairs: {type(e).__name__}: {e}")
        return []


async def get_token_largest_accounts(session: aiohttp.ClientSession, mint_address: str) -> List[dict]:
    """Call Solana RPC to get token largest accounts with rate limiting."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [
            mint_address,
            {"commitment": "finalized"}
        ]
    }
    
    retries = 3
    base_delay = 20 # Seconds for 429 backoff

    for attempt in range(retries):
        try:
            await rpc_limiter.acquire()
            async with session.post(RPC_ENDPOINT, json=payload, timeout=10) as response:
                if response.status == 429:
                    logger.warning(f"RPC 429 Too Many Requests for {mint_address}. Sleeping {base_delay}s.")
                    await asyncio.sleep(base_delay * (2 ** attempt)) # Exponential backoff
                    continue
                
                if response.status != 200:
                    logger.error(f"RPC Error {response.status} for {mint_address}")
                    return []

                if "application/json" not in response.content_type:
                    logger.warning(f"RPC returned invalid content type for {mint_address}: {response.content_type}")
                    # Likely a cloudflare page or similar 5xx disguised as 200 or just bad proxy
                    return []

                try:
                    json_resp = await response.json()
                except Exception as e:
                    logger.error(f"RPC JSON decode error for {mint_address}: {e}")
                    return []

                if "error" in json_resp:
                    logger.error(f"RPC API Error for {mint_address}: {json_resp['error']}")
                    return []
                
                return json_resp.get("result", {}).get("value", [])

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching holders for {mint_address}: {e}")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Exception fetching holders for {mint_address}: {e}")
            await asyncio.sleep(1)
            
    return []


async def get_token_accounts_by_owner(session: aiohttp.ClientSession, owner_address: str) -> List[dict]:
    """Get all token accounts for a specific owner (used to find vaults)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            owner_address,
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
            {"encoding": "jsonParsed"}
        ]
    }
    
    try:
        await rpc_limiter.acquire()
        async with session.post(RPC_ENDPOINT, json=payload, timeout=10) as response:
            if response.status != 200:
                logger.error(f"RPC Error fetching accounts for {owner_address}")
                return []
            
            data = await response.json()
            return data.get("result", {}).get("value", [])
    except Exception as e:
        logger.error(f"Error fetching token accounts for owner {owner_address}: {e}")
        return []


# Global cache for LbPair offsets: {pair_address: (bin_step_offset, active_id_offset, vol_offset)}
# Global cache for LbPair offsets: {pair_address: (bin_step_offset, active_id_offset, vol_offset)}
LBPAIR_OFFSET_CACHE = {}

async def decode_lb_pair(data: bytes, pair_address: str = None) -> Optional[Tuple[int, int, float]]:
    """
    Decodes the Meteora DLMM LbPair account data with DYNAMIC offset detection.
    
    FIX: The active_id offset varies between pools due to variable-length fields,
    so we scan for valid values instead of using fixed offset 677.
    
    Returns: (active_id, bin_step, volatility_accumulator) or None
    """
    try:
        # Validate minimum data length
        if len(data) < 700:
            logger.debug(f"[{pair_address}] Insufficient data: {len(data)} bytes")
            return None
        
        # Verify discriminator (identifies this as an LbPair account)
        discriminator = data[:8]
        expected_discriminator = bytes.fromhex("f19a6d0411b16dbc")
        
        if discriminator != expected_discriminator:
            logger.debug(
                f"[{pair_address}] Wrong discriminator: "
                f"expected {expected_discriminator.hex()}, got {discriminator.hex()}"
            )
            return None
        
        BASIS_POINT_MAX = 10000
        ACTIVE_ID_BASE = 8388608  # 2^23
        
        # STEP 1: Extract bin_step (this offset is stable)
        bin_step_offset = 48
        try:
            bin_step = struct.unpack_from('<H', data, bin_step_offset)[0]
            
            # Validate bin_step (should be reasonable basis points)
            if not (1 <= bin_step <= 5000):
                logger.debug(f"[{pair_address}] Invalid bin_step: {bin_step}")
                return None
                
        except struct.error:
            logger.debug(f"[{pair_address}] Failed to extract bin_step at offset {bin_step_offset}")
            return None
        
        # STEP 2: FIX - Scan for active_id dynamically (offset varies between pools)
        # Based on diagnostic: valid active_ids are in range 5M-15M
        # Search in the first 700 bytes, checking every 4-byte boundary
        
        valid_candidates = []
        
        for offset in range(100, min(700, len(data) - 4), 4):
            try:
                active_id = struct.unpack_from('<i', data, offset)[0]
                
                # Check if this looks like a valid active_id
                if 5_000_000 <= active_id <= 15_000_000:
                    # Test if price calculation would work
                    try:
                        adjusted_id = active_id - ACTIVE_ID_BASE
                        test_price = (1 + bin_step / BASIS_POINT_MAX) ** adjusted_id
                        
                        import numpy as np
                        if np.isfinite(test_price) and (1e-15 < test_price < 1e15):
                            valid_candidates.append((offset, active_id, test_price))
                    except (OverflowError, ValueError):
                        pass
                        
            except struct.error:
                pass
        
        if not valid_candidates:
            logger.debug(f"[{pair_address}] No valid active_id found in scan")
            return None
        
        # STEP 3: Pick the best candidate (first valid one)
        offset, active_id, price = valid_candidates[0]
        
        logger.debug(
            f"[{pair_address}] Decoded successfully: "
            f"bin_step={bin_step}, active_id={active_id} (offset={offset}), "
            f"price={price:.9f}"
        )
        
        # Volatility accumulator (placeholder - offset unknown)
        volatility_accumulator = 0.0
        
        return (active_id, bin_step, volatility_accumulator)
        
    except Exception as e:
        logger.error(f"[{pair_address}] Unexpected decode error: {e}")
        return None


# ============================================================================
# METEORA DLMM REST API - Primary data source (Feb 2026)
# ============================================================================

# Cache for Meteora pool data: {pair_address: (data_dict, timestamp)}
_meteora_pool_cache: Dict[str, Tuple[dict, float]] = {}
METEORA_CACHE_TTL = 5.0  # 5 seconds cache TTL

# Meteora DLMM API rate limiter: 30 RPS as per official docs
# https://docs.meteora.ag/api-reference/dlmm/overview
meteora_limiter = TokenBucketLimiter(rate=30.0, capacity=30.0)


async def fetch_meteora_pool_data(session: aiohttp.ClientSession, pair_address: str) -> Optional[dict]:
    """
    Fetch pool data from Meteora DLMM REST API.
    
    Official Docs: https://docs.meteora.ag/api-reference/dlmm/overview
    Endpoint: GET https://dlmm.datapi.meteora.ag/pools/{address}
    Rate limit: 30 RPS (shared across all calls - enforced via meteora_limiter)
    
    Returns dict with: current_price, volume_1h, bin_step, tvl,
    token_x_amount, token_y_amount, dynamic_fee_pct, or None if unavailable.
    """
    import time
    
    # Check cache first
    if pair_address in _meteora_pool_cache:
        cached_data, cached_time = _meteora_pool_cache[pair_address]
        if time.time() - cached_time < METEORA_CACHE_TTL:
            return cached_data
    
    # Apply global rate limiting (30 RPS across all concurrent calls)
    await meteora_limiter.acquire()
    
    try:
        url = f"{METEORA_DLMM_API}/pools/{pair_address}"
        async with session.get(url, timeout=5) as response:
            if response.status == 429:
                logger.warning(f"Meteora API rate limit hit (429) for {pair_address}")
                return None
            
            if response.status != 200:
                logger.debug(f"Meteora API returned {response.status} for {pair_address}")
                return None
            
            data = await response.json()
            
            if not data or not data.get("address"):
                return None
            
            # Extract fields from response
            pool_config = data.get("pool_config", {})
            volume_data = data.get("volume", {})
            
            result = {
                "current_price": float(data.get("current_price", 0) or 0),
                "volume_1h": float(volume_data.get("1h", 0) or 0),
                "volume_30m": float(volume_data.get("30m", 0) or 0),
                "bin_step": int(pool_config.get("bin_step", 0) or 0),
                "tvl": float(data.get("tvl", 0) or 0),
                "token_x_amount": float(data.get("token_x_amount", 0) or 0),
                "token_y_amount": float(data.get("token_y_amount", 0) or 0),
                "dynamic_fee_pct": float(data.get("dynamic_fee_pct", 0) or 0),
            }
            
            # Validate price
            if result["current_price"] <= 0:
                logger.debug(f"Meteora API returned zero/negative price for {pair_address}")
                return None
            
            # Cache it
            _meteora_pool_cache[pair_address] = (result, time.time())
            return result
            
    except asyncio.TimeoutError:
        logger.debug(f"Meteora API timeout for {pair_address}")
        return None
    except aiohttp.ClientError as e:
        logger.debug(f"Meteora API network error for {pair_address}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Meteora API fetch failed for {pair_address}: {type(e).__name__}: {e}")
        return None


# Jupiter price cache: {mint_address: (price_data, timestamp)}
# Price data now includes: usd_price, decimals, block_id, price_change_24h
_jupiter_price_cache: Dict[str, Tuple[dict, float]] = {}
JUPITER_CACHE_TTL = 5.0  # 5 seconds cache TTL

# Jupiter API rate limiter: 10 RPS with burst of 20 (conservative for free tier)
jupiter_limiter = TokenBucketLimiter(rate=10.0, capacity=20.0)


async def fetch_jupiter_prices_batch(
    session: aiohttp.ClientSession, 
    mint_addresses: List[str]
) -> Dict[str, dict]:
    """
    Fetch prices for multiple tokens from Jupiter Price API V3 in a SINGLE request.
    
    Official Docs: https://dev.jup.ag/docs/price/v3
    - Can query up to 50 ids at once
    - Returns: usdPrice, blockId, decimals, priceChange24h
    - Requires x-api-key header for production use
    
    Args:
        mint_addresses: List of mint addresses (max 50 per Jupiter docs)
        
    Returns:
        Dict mapping mint_address -> price_data dict with keys:
        - usd_price: float
        - decimals: int
        - block_id: int
        - price_change_24h: float
    """
    import time
    
    if not mint_addresses:
        return {}
    
    # Jupiter limit: max 50 ids per request
    if len(mint_addresses) > 50:
        logger.warning(f"Batch size {len(mint_addresses)} exceeds Jupiter limit of 50, truncating")
        mint_addresses = mint_addresses[:50]
    
    # Check cache first and filter out cached addresses
    results = {}
    uncached_addresses = []
    
    for addr in mint_addresses:
        if addr in _jupiter_price_cache:
            cached_data, cached_time = _jupiter_price_cache[addr]
            if time.time() - cached_time < JUPITER_CACHE_TTL:
                results[addr] = cached_data
                continue
        uncached_addresses.append(addr)
    
    if not uncached_addresses:
        return results
    
    # Rate limit the API call
    await jupiter_limiter.acquire()
    
    try:
        # Official Jupiter Price API V3 endpoint
        ids_param = ",".join(uncached_addresses)
        url = f"https://api.jup.ag/price/v3?ids={ids_param}"
        
        # API key is REQUIRED for production (free at portal.jup.ag)
        headers = {}
        if JUPITER_API_KEY:
            headers["x-api-key"] = JUPITER_API_KEY
        else:
            logger.warning("JUPITER_API_KEY not set - using public tier (lower rate limits)")
        
        async with session.get(url, headers=headers, timeout=10) as response:
            if response.status == 429:
                logger.warning(f"Jupiter API rate limit hit (429) for batch of {len(uncached_addresses)} tokens")
                return results
            
            if response.status != 200:
                logger.debug(f"Jupiter API V3 returned {response.status}")
                return results
            
            data = await response.json()
            
            # Response format: {"<mint>": {"usdPrice": 0.123, "blockId": 123, "decimals": 6, "priceChange24h": 1.23}, ...}
            for mint in uncached_addresses:
                token_data = data.get(mint)
                if not token_data:
                    continue
                
                usd_price = token_data.get("usdPrice")
                if usd_price is None:
                    continue
                
                price_data = {
                    "usd_price": float(usd_price),
                    "decimals": token_data.get("decimals", 0),
                    "block_id": token_data.get("blockId", 0),
                    "price_change_24h": token_data.get("priceChange24h", 0.0),
                }
                
                _jupiter_price_cache[mint] = (price_data, time.time())
                results[mint] = price_data
                
                logger.debug(f"Jupiter V3 price for {mint}: ${price_data['usd_price']:.9f} "
                           f"(block {price_data['block_id']}, 24h: {price_data['price_change_24h']:+.2f}%)")
            
    except asyncio.TimeoutError:
        logger.warning(f"Jupiter API timeout for batch of {len(uncached_addresses)} tokens")
    except Exception as e:
        logger.debug(f"Jupiter V3 batch fetch failed: {type(e).__name__}: {e}")
    
    return results


async def fetch_jupiter_price(session: aiohttp.ClientSession, mint_address: str) -> Optional[float]:
    """
    Fetch single token price from Jupiter Price API V3 (convenience wrapper).
    
    NOTE: For fetching multiple prices, use fetch_jupiter_prices_batch() instead
    to reduce API calls by up to 50x (Jupiter allows 50 ids per request).
    
    Returns: price in USD or None if unavailable.
    """
    results = await fetch_jupiter_prices_batch(session, [mint_address])
    if mint_address in results:
        return results[mint_address]["usd_price"]
    return None


async def fetch_bin_arrays(
    session: aiohttp.ClientSession, 
    pair_address: str, 
    active_id: int,
    bin_range: int = 10
) -> Tuple[float, float]:
    """
    Fetch BinArray account data from Meteora DLMM to calculate liquidity depth.
    
    BinArray Layout:
    - Each BinArray holds 70 bins
    - Each bin contains: amountX (u128, 16 bytes), amountY (u128, 16 bytes), plus other fields
    - amountX = liquidity on ask side (sell orders)
    - amountY = liquidity on bid side (buy orders)
    
    We sum liquidity from bins around the active_id (+/- bin_range).
    
    Returns: (liquidity_depth_ask, liquidity_depth_bid) as raw lamport sums.
    
    Note: For full implementation, we would need to derive BinArray PDAs.
    This is a simplified version that estimates liquidity from the LbPair directly.
    """
    # For now, return placeholder while we implement full BinArray fetching
    # This requires deriving the BinArray PDA addresses which depends on:
    # - LbPair address
    # - bin_array_index = floor(bin_id / 70)
    # - seeds = [b"bin_array", lb_pair_bytes, index_bytes]
    
    # TODO: Implement full BinArray PDA derivation and fetching
    # For MVP, we'll use an alternative approach: fetch from Jupiter API
    
    try:
        # Alternative: Use Jupiter swap simulation to estimate liquidity
        # This gives us an indirect measure of available liquidity
        # by simulating a small swap and looking at price impact
        
        # For now, return non-zero placeholder to indicate "unknown but present"
        # This is better than returning 0.0 which implies "no liquidity"
        return (1.0, 1.0)  # Placeholder indicating unknown liquidity
        
    except Exception as e:
        logger.debug(f"BinArray fetch failed for {pair_address}: {e}")
        return (0.0, 0.0)


def _calculate_all_metrics(
    mint_address: str,
    pair_address: str,
    prices_array: np.ndarray,
    timestamps_array: np.ndarray,
    volumes_array: np.ndarray,
    creation_timestamp: float,
    current_time: float,
) -> dict:
    """
    Calculate ALL available metrics from price/volume/timestamp history.
    
    🔥 V3 OPTIMIZED: Reduced thresholds for faster dump detection
    - Phase 1 (n >= 2): Critical signals (price_jerk, pump_duration)  
    - Phase 2 (n >= 5): Tier A predictors (benford, price_age_corr)
    - Phase 3 (n >= 10): Technical indicators (RSI, periodicity)
    - Phase 4+ (n >= 15-30): Advanced metrics
    
    🔥 INSTITUTIONAL GRADE: Uses tick-by-tick transaction data for microstructure
    metrics (Kyle's Lambda, Amihud, VPT) when available, falling back to hourly
    volume estimates only when necessary.
    
    Returns a dict of metric_name -> value (or None if not enough data).
    """
    n = len(prices_array)
    metrics = {}
    
    if n < 2:
        return metrics
    
    # Calculate returns (used by many metrics)
    returns = np.diff(prices_array) / (prices_array[:-1] + 1e-15)
    price_changes = np.diff(prices_array)
    
    # Volume array for microstructure metrics
    if len(volumes_array) > 0 and np.any(volumes_array > 0):
        vols_for_micro = volumes_array[1:] if len(volumes_array) == n else volumes_array
    else:
        vols_for_micro = np.ones_like(price_changes)
    
    # Ensure volume arrays match return length
    min_len = min(len(returns), len(vols_for_micro))
    returns_trimmed = returns[:min_len]
    vols_trimmed = vols_for_micro[:min_len]
    
    prices_list = prices_array.tolist()
    timestamps_list = timestamps_array.tolist()
    
    # ========== PHASE 0: INSTANT (n >= 1) ==========
    try:
        metrics['time_since_creation_sec'] = int(current_time - creation_timestamp)
    except Exception:
        pass
    
    # ========== PHASE 1: ULTRA FAST (n >= 2) - TIER S DUMP SIGNALS ==========
    try:
        velocity, acceleration, jerk = calculate_velocity_acceleration_jerk(prices_list, timestamps_list)
        metrics['price_velocity'] = velocity
        metrics['price_acceleration'] = acceleration
        metrics['price_jerk'] = jerk  # 🔥 TIER S: Critical dump signal
        
        # Alert on high negative jerk
        if jerk is not None and jerk < -0.01:
            logger.warning(f"[{mint_address[:8]}] High negative jerk: {jerk:.6f} (dump signal!)")
    except Exception:
        pass
    
    try:
        metrics['max_drawdown'] = calculate_max_drawdown(prices_array)
    except Exception:
        pass
    
    try:
        # Pump duration - TIER B predictor
        pump_ticks = detect_pump_duration(prices_array)
        if pump_ticks is not None:
            avg_interval = np.mean(np.diff(timestamps_array)) if len(timestamps_array) > 1 else 1.0
            metrics['pump_duration_sec'] = int(pump_ticks * avg_interval)
            
            if metrics['pump_duration_sec'] < 60:
                logger.warning(f"[{mint_address[:8]}] Fast pump: {metrics['pump_duration_sec']}s")
    except Exception:
        pass
    
    # ========== PHASE 2: FAST (n >= 5) - TIER A PREDICTORS ==========
    # 🔥 Reduced from n >= 10 for faster detection
    if n >= 5:
        try:
            # Benford Deviation - TIER A: Fake volume detector
            metrics['benford_deviation'] = benford_deviation(prices_array)
            if metrics['benford_deviation'] > 15:
                logger.warning(f"[{mint_address[:8]}] Benford violation: {metrics['benford_deviation']:.1f}")
        except Exception:
            pass
        
        try:
            # Price-Age Correlation - TIER A: Dump trajectory
            metrics['price_age_correlation'] = calculate_price_age_correlation(prices_array, timestamps_array)
            if metrics['price_age_correlation'] and metrics['price_age_correlation'] < -0.5:
                logger.warning(f"[{mint_address[:8]}] Dump trajectory: {metrics['price_age_correlation']:.2f}")
        except Exception:
            pass
        
        try:
            # Autocorrelation - TIER B: Bot detection
            metrics['autocorrelation_lag1'] = calculate_autocorrelation(prices_array)
        except Exception:
            pass
        
        try:
            metrics['volume_price_trend'] = calculate_vpt(prices_array, vols_for_micro[:n] if len(vols_for_micro) >= n else np.ones(n))
        except Exception:
            pass
        
        try:
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
        except Exception:
            pass
        
        try:
            metrics['kurtosis'] = calculate_kurtosis(prices_array)
            metrics['skewness'] = calculate_skewness(prices_array)
        except Exception:
            pass
    
    # ========== PHASE 3: MEDIUM (n >= 10) - TECHNICAL INDICATORS ==========
    if n >= 10:
        try:
            # RSI - TIER A: Overbought detection (works fine with 10 samples)
            metrics['rsi_14'] = calculate_rsi(prices_array)
            if metrics['rsi_14'] and metrics['rsi_14'] > 85:
                logger.warning(f"[{mint_address[:8]}] Extreme overbought: RSI={metrics['rsi_14']:.0f}")
        except Exception:
            pass
        
        try:
            # Spectral Analysis - TIER A: Periodicity/bot detection  
            spectral = spectral_analysis(prices_array, sample_rate=1.0)
            metrics['dominant_frequency'] = spectral.get("dominant_frequency")
            metrics['spectral_entropy'] = spectral.get("spectral_entropy")
            metrics['periodicity_score'] = spectral.get("periodicity_score")
            
            if metrics['periodicity_score'] and metrics['periodicity_score'] > 0.6:
                logger.warning(f"[{mint_address[:8]}] Bot activity: {metrics['periodicity_score']:.2f}")
        except Exception:
            pass
        
        try:
            metrics['fractal_dimension'] = fractal_dimension(prices_array)
        except Exception:
            pass
        
        try:
            metrics['shannon_entropy'] = shannon_entropy(prices_array)
        except Exception:
            pass
        
        try:
            metrics['jarque_bera_stat'] = jarque_bera_test(returns)
        except Exception:
            pass
        
        try:
            metrics['var_95'] = calculate_var_95(returns)
            metrics['calmar_ratio'] = calculate_calmar_ratio(returns, prices_array)
        except Exception:
            pass
        
        try:
            # Roll spread uses only price data - always accurate
            metrics['roll_spread'] = roll_spread(prices_array)
        except Exception:
            pass
        
        # ========== INSTITUTIONAL MICROSTRUCTURE (TICK DATA) ==========
        # Use actual transaction data if available, fallback to hourly estimates
        try:
            ticks = list(tick_history.get(pair_address, []))
            
            if len(ticks) >= 5:
                # We have tick data! Calculate microstructure metrics with ACTUAL volumes
                tick_prices = np.array([t.price if t.price > 0 else prices_array[-1] for t in ticks])
                tick_volumes = np.array([t.volume for t in ticks])
                
                # Calculate returns from tick prices
                tick_returns = np.diff(np.log(tick_prices + 1e-10))
                tick_price_changes = np.diff(tick_prices)
                
                # Kyle's Lambda with ACTUAL per-trade volumes
                if len(tick_price_changes) >= 5 and len(tick_volumes) >= 6:
                    metrics['kyle_lambda'] = kyle_lambda(tick_price_changes, tick_volumes[1:])
                
                # Amihud with ACTUAL per-trade volumes
                if len(tick_returns) >= 5 and len(tick_volumes) >= 6:
                    metrics['amihud_illiquidity'] = amihud_illiquidity(tick_returns, tick_volumes[1:])
                
                # VPT with ACTUAL per-trade volumes
                if len(tick_prices) >= 5 and len(tick_volumes) >= 5:
                    metrics['volume_price_trend'] = calculate_vpt(tick_prices, tick_volumes)
                
                # Store tick count for monitoring
                metrics['tick_count_1m'] = len(ticks)
                
                logger.debug(
                    f"[{mint_address[:8]}] Microstructure from {len(ticks)} ticks: "
                    f"Kyle={metrics.get('kyle_lambda', 0):.6f}, "
                    f"Amihud={metrics.get('amihud_illiquidity', 0):.2e}"
                )
            else:
                # Fallback to hourly volume estimates (less accurate)
                metrics['amihud_illiquidity'] = amihud_illiquidity(returns_trimmed, vols_trimmed)
                metrics['tick_count_1m'] = 0
                
        except Exception as e:
            logger.debug(f"[{mint_address[:8]}] Tick microstructure error: {e}")
            # Fallback to hourly estimates
            try:
                metrics['amihud_illiquidity'] = amihud_illiquidity(returns_trimmed, vols_trimmed)
            except:
                pass
    
    # ========== PHASE 4: SLOW (n >= 15) - RISK METRICS ==========
    # 🔥 Reduced from n >= 20
    if n >= 15:
        try:
            metrics['bollinger_position'] = calculate_bollinger_position(prices_array)
        except Exception:
            pass
        
        try:
            metrics['hurst_exponent'] = hurst_exponent(prices_array)
        except Exception:
            pass
        
        # Kyle's Lambda - only calculate if not already done with tick data
        if 'kyle_lambda' not in metrics or metrics['kyle_lambda'] is None:
            try:
                metrics['kyle_lambda'] = kyle_lambda(price_changes, vols_for_micro[:len(price_changes)])
            except Exception:
                pass
    
    # ========== PHASE 5: ADVANCED (n >= 26) - MACD ==========
    if n >= 26:
        try:
            metrics['macd_signal'] = calculate_macd(prices_array)
        except Exception:
            pass
    
    # ========== PHASE 6: VERY ADVANCED (n >= 30) - LYAPUNOV ==========
    if n >= 30:
        try:
            metrics['lyapunov_exponent'] = lyapunov_exponent(prices_array)
        except Exception:
            pass
    
    return metrics


def calculate_dump_probability(
    metrics: dict,
    holder_state: dict,
    current_price: float = None,
    current_volume: float = None
) -> float:
    """
    Calculate real-time dump probability using weighted scoring.
    
    🔥 HOTFIXED: Now safely handles None values everywhere
    
    Returns: 0.0-1.0 (0% to 100% dump probability)
    """
    score = 0.0
    signals = []
    
    # Helper function to safely get numeric values
    def safe_get(d, key, default=0):
        """Safely get a numeric value, return default if None or missing"""
        val = d.get(key, default)
        return val if val is not None else default
    
    # ========== TIER S: Immediate Dump Signals (40% weight) ==========
    
    # S1: max_holder_dump - THE #1 PREDICTOR (20% max)
    max_dump = safe_get(holder_state, 'max_holder_dump', 0)
    if max_dump < -10:
        score += 0.20
        signals.append(f"MAJOR_DUMP({abs(max_dump):.1f}%)")
    elif max_dump < -5:
        score += 0.10
        signals.append(f"DUMP({abs(max_dump):.1f}%)")
    elif max_dump < -2:
        score += 0.05
        signals.append(f"MINOR_DUMP({abs(max_dump):.1f}%)")
    
    # S2: price_jerk - Momentum breaking (10% max)
    price_jerk = safe_get(metrics, 'price_jerk', 0)
    if price_jerk < -0.1:
        score += 0.10
        signals.append(f"EXTREME_JERK({price_jerk:.3f})")
    elif price_jerk < -0.01:
        score += 0.05
        signals.append(f"HIGH_JERK({price_jerk:.3f})")
    
    # S3: holder_count_trend - Retail exodus (5% max)
    holder_trend = safe_get(holder_state, 'holder_count_trend', 0)
    if holder_trend < -0.2:
        score += 0.05
        signals.append(f"EXODUS({holder_trend:.2f}/s)")
    elif holder_trend < -0.1:
        score += 0.02
        signals.append(f"LEAVING({holder_trend:.2f}/s)")
    
    # S4: volatility_speed (5% max)
    vol_speed = safe_get(metrics, 'volatility_speed', 0)
    if vol_speed > 0.5:
        score += 0.05
        signals.append(f"VOL_SPIKE({vol_speed:.2f})")
    
    # ========== TIER A: Pre-Dump Conditions (30% weight) ==========
    
    # A1: rsi_14 - Extreme overbought (10% max)
    rsi = safe_get(metrics, 'rsi_14', 50)  # Default to neutral 50
    if rsi > 90:
        score += 0.10
        signals.append(f"RSI_EXTREME({rsi:.0f})")
    elif rsi > 80:
        score += 0.06
        signals.append(f"RSI_HIGH({rsi:.0f})")
    elif rsi > 70:
        score += 0.03
        signals.append(f"RSI_OB({rsi:.0f})")
    
    # A2: benford_deviation - Fake volume (8% max)
    benford = safe_get(metrics, 'benford_deviation', 0)
    if benford > 50:
        score += 0.08
        signals.append(f"FAKE_VOL_SEVERE({benford:.1f})")
    elif benford > 15:
        score += 0.04
        signals.append(f"FAKE_VOL({benford:.1f})")
    
    # A3: price_age_correlation - Dump trajectory (7% max)
    price_age = safe_get(metrics, 'price_age_correlation', 0)
    if price_age < -0.7:
        score += 0.07
        signals.append(f"DUMP_TRAJ({price_age:.2f})")
    elif price_age < -0.5:
        score += 0.04
        signals.append(f"DECLINING({price_age:.2f})")
    
    # A4: periodicity_score - Bot exit (5% max)
    periodicity = safe_get(metrics, 'periodicity_score', 0)
    if periodicity > 0.6:
        score += 0.05
        signals.append(f"BOT({periodicity:.2f})")
    
    # ========== TIER B: Structural Risks (20% weight) ==========
    
    # B1: pump_duration_sec - Fast pump indicator (10% max)
    pump_duration = safe_get(metrics, 'pump_duration_sec', 300)  # Default to 5 min = safe
    if pump_duration < 30:
        score += 0.10
        signals.append(f"FLASH_PUMP({pump_duration}s)")
    elif pump_duration < 60:
        score += 0.06
        signals.append(f"FAST_PUMP({pump_duration}s)")
    elif pump_duration < 120:
        score += 0.03
        signals.append(f"QUICK_PUMP({pump_duration}s)")
    
    # B2: gini_coefficient - Concentration (5% max)
    gini = safe_get(holder_state, 'gini_coefficient', 0)
    if gini > 0.9:
        score += 0.05
        signals.append(f"GINI_EXTREME({gini:.3f})")
    elif gini > 0.85:
        score += 0.03
        signals.append(f"GINI_HIGH({gini:.3f})")
    
    # B3: top3_vs_rest_ratio - Whale dominance (3% max)
    top3_ratio = safe_get(holder_state, 'top3_vs_rest_ratio', 0)
    if top3_ratio > 10:
        score += 0.03
        signals.append(f"WHALE_DOM({top3_ratio:.1f})")
    
    # B4: autocorrelation_lag1 - Bot pattern (2% max)
    autocorr = safe_get(metrics, 'autocorrelation_lag1', 0)
    if autocorr > 0.5:
        score += 0.02
        signals.append(f"BOT_PATTERN({autocorr:.2f})")
    
    # ========== TIER C: Volume/Momentum (10% weight) ==========
    
    # C1: max_drawdown - Already dumping (5% max)
    max_dd = safe_get(metrics, 'max_drawdown', 0)
    if max_dd > 0.8:
        score += 0.05
        signals.append(f"DD_SEVERE({max_dd*100:.0f}%)")
    elif max_dd > 0.5:
        score += 0.03
        signals.append(f"DD_HIGH({max_dd*100:.0f}%)")
    
    # C2: Bollinger position extreme (2% max)
    bollinger = safe_get(metrics, 'bollinger_position', 0.5)  # Default to middle
    if bollinger > 0.95:
        score += 0.02
        signals.append(f"BB_EXTREME({bollinger:.2f})")
    
    # Cap at 1.0 (100%)
    final_score = min(1.0, score)
    
    # Log if significant risk detected
    if final_score >= 0.5 and signals:
        logger.warning(
            f"DUMP RISK: {final_score*100:.0f}% probability - "
            f"Signals: {', '.join(signals)}"
        )
    
    return final_score


def get_dump_recommendation(dump_prob: float, current_position: bool = False) -> dict:
    """
    Get trading recommendation based on dump probability.
    
    Args:
        dump_prob: Probability from calculate_dump_probability (0.0-1.0)
        current_position: Whether we currently hold this token
    
    Returns:
        dict with 'action', 'urgency', and 'reason'
    """
    if dump_prob >= 0.9:
        return {
            'action': 'SELL_ALL' if current_position else 'DO_NOT_BUY',
            'urgency': 'IMMEDIATE',
            'reason': 'Dump imminent (90%+ probability)',
            'exit_percentage': 100 if current_position else None
        }
    elif dump_prob >= 0.7:
        return {
            'action': 'SELL_PARTIAL' if current_position else 'DO_NOT_BUY',
            'urgency': 'HIGH',
            'reason': 'High dump risk (70-90% probability)',
            'exit_percentage': 75 if current_position else None
        }
    elif dump_prob >= 0.5:
        return {
            'action': 'REDUCE' if current_position else 'SMALL_SIZE_ONLY',
            'urgency': 'MEDIUM',
            'reason': 'Elevated dump risk (50-70% probability)',
            'exit_percentage': 50 if current_position else None
        }
    elif dump_prob >= 0.3:
        return {
            'action': 'MONITOR' if current_position else 'NORMAL_SIZE',
            'urgency': 'LOW',
            'reason': 'Moderate risk (30-50% probability)',
            'exit_percentage': None
        }
    else:
        return {
            'action': 'HOLD' if current_position else 'BUY_OK',
            'urgency': 'NONE',
            'reason': 'Low dump risk (<30% probability)',
            'exit_percentage': None
        }


async def fast_price_monitor(mint_address: str, pair_address: str, base_symbol: str, session: aiohttp.ClientSession, created_at: datetime.datetime = None):
    """
    High-frequency monitor (1s interval) for 5 minutes.
    
    🔥 STRATEGIA 2026: Adaptive Data Source based on token age
    
    FASE 1 - NEONATO (0-120s): RPC-FIRST
        Per token appena nati, le API (Meteora/Jupiter) non hanno ancora indicizzato
        il token (docs 2026: Jupiter V3 richiede scambi entro 7 giorni).
        L'RPC on-chain con commitment "confirmed" è l'unica fonte affidabile.
        
    FASE 2 - MATURO (120s-300s): HYBRID
        1. Meteora DLMM REST API — prezzo + volume + liquidità
        2. Jupiter Price API V3 — fallback prezzo solo
        3. RPC On-Chain — fallback critico se API falliscono + volatility_accumulator
    
    Docs 2026: 
    - Solana RPC: commitment "confirmed" bilancia velocità e sicurezza
    - Jupiter V3: non restituisce prezzi per token senza storia di trading
    
    ALL metrics are calculated on EVERY tick regardless of data source.
    """
    logger.info(f"Starting Fast Monitor for {base_symbol} ({pair_address}) [METEORA API + ALPHA V2]")
    
    # Register this token as actively monitored
    active_monitors[mint_address] = True
    
    # Store creation time for time-based metrics
    creation_timestamp = created_at.timestamp() if created_at else asyncio.get_event_loop().time()
    token_creation_times[mint_address] = creation_timestamp
    
    # Launch holder check task for this token
    holder_task = asyncio.create_task(holder_check_task(mint_address, base_symbol, session))
    
    # Launch tick fetcher task for accurate microstructure metrics
    # Fetches swap transactions every 15 seconds to populate tick_history
    tick_task = asyncio.create_task(tick_fetcher_task(pair_address, base_symbol, session))
    
    try:
        # Monitoring Loop (300 seconds = 5 minutes)
        for tick in range(300):
            start_time = asyncio.get_event_loop().time()
            
            try:
                price = None
                volume_1h = 0.0
                active_bin_id = None
                bin_step_val = None
                volatility_accumulator = None
                volatility_speed = None
                liquidity_depth_ask = None
                liquidity_depth_bid = None
                data_source = "none"
                
                # Calcola età del token per strategia dati adattiva
                token_age_sec = asyncio.get_event_loop().time() - creation_timestamp
                is_newborn = token_age_sec < 120  # Primi 2 minuti: RPC-first
                jupiter_block_id = None
                
                # ==========================================
                # STRATEGIA 2026: RPC-FIRST per Token Neonati (0-120s)
                # ==========================================
                # Per token appena nati, le API (Meteora/Jupiter) non hanno ancora indicizzato
                # il token. L'RPC on-chain è l'UNICA fonte affidabile in questa fase.
                # Docs 2026: Jupiter V3 non restituisce prezzi per token non scambiati da 7+ giorni
                
                if is_newborn:
                    # NEONATO: Prova RPC PRIMO (strategia aggressiva)
                    try:
                        RPC_EP = os.getenv("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
                        # 2026 Best Practice: commitment "confirmed" bilancia velocità e sicurezza
                        payload = {
                            "jsonrpc": "2.0", "id": 1,
                            "method": "getAccountInfo",
                            "params": [pair_address, {"encoding": "base64", "commitment": "confirmed"}]
                        }
                        
                        await rpc_limiter.acquire()
                        
                        async with session.post(RPC_EP, json=payload, timeout=5) as rpc_resp:
                            if rpc_resp.status == 200:
                                rpc_data = await rpc_resp.json()
                                account_data = rpc_data.get("result", {}).get("value", {})
                                if account_data and account_data.get("data"):
                                    import base64
                                    raw_data = base64.b64decode(account_data["data"][0])
                                    decoded = await decode_lb_pair(raw_data, pair_address)
                                    
                                    if decoded:
                                        aid, bs, vol_acc = decoded
                                        active_bin_id = aid
                                        bin_step_val = bs
                                        volatility_accumulator = vol_acc
                                        
                                        # Calcolo manuale prezzo DLMM (2026 formula)
                                        if active_bin_id is not None and bin_step_val is not None:
                                            ACTIVE_ID_BASE = 8388608  # 2^23
                                            bp = bin_step_val / 10000.0
                                            true_active_id = active_bin_id - ACTIVE_ID_BASE
                                            calculated_price = (1 + bp) ** true_active_id
                                            
                                            if calculated_price > 0 and 1e-15 < calculated_price < 1e15:
                                                price = calculated_price
                                                data_source = "rpc_decode_newborn"
                                                # Stima volume_1h dai tick raccolti (se disponibili)
                                                volume_1h = _estimate_volume_from_ticks(pair_address, token_age_sec)
                    except Exception:
                        pass  # Fallback alle API se RPC fallisce
                
                # ==========================================
                # SOURCE 1: Meteora DLMM REST API (PRIMARY per token maturi)
                # ==========================================
                # Per token >120s, Meteora è la fonte migliore (prezzo + volume + liquidità)
                if not is_newborn and (price is None or price <= 0):
                    meteora_data = await fetch_meteora_pool_data(session, pair_address)
                    if meteora_data and meteora_data["current_price"] > 0:
                        price = meteora_data["current_price"]
                        volume_1h = meteora_data["volume_1h"]
                        bin_step_val = meteora_data["bin_step"] if meteora_data["bin_step"] > 0 else None
                        liquidity_depth_ask = meteora_data["token_x_amount"] if meteora_data["token_x_amount"] > 0 else None
                        liquidity_depth_bid = meteora_data["token_y_amount"] if meteora_data["token_y_amount"] > 0 else None
                        data_source = "meteora_api"
                
                # ==========================================
                # SOURCE 2: Jupiter Price API V3 (FALLBACK)
                # ==========================================
                # Docs 2026: Jupiter V3 restituisce prezzi solo per token con scambi recenti
                # NOTA: Per token <7 giorni, Jupiter probabilmente NON avrà il prezzo
                if price is None or price <= 0:
                    jupiter_results = await fetch_jupiter_prices_batch(session, [mint_address])
                    if mint_address in jupiter_results:
                        jupiter_data = jupiter_results[mint_address]
                        price = jupiter_data["usd_price"]
                        jupiter_block_id = jupiter_data["block_id"]
                        data_source = "jupiter_v3"
                
                # ==========================================
                # SOURCE 3: RPC On-Chain Decode (CRITICAL FALLBACK / AUXILIARY)
                # ==========================================
                # Per token maturi: RPC ogni 10 tick per volatility_accumulator
                # Per token senza prezzo: RPC immediato come fallback
                should_run_rpc = (not is_newborn and (price is None or price <= 0)) or (tick % 10 == 0)

                if should_run_rpc:
                    try:
                        RPC_EP = os.getenv("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
                        payload = {
                            "jsonrpc": "2.0", "id": 1,
                            "method": "getAccountInfo",
                            "params": [pair_address, {"encoding": "base64"}]
                        }
                        
                        # Only use rate limiter if we are actually making the request
                        await rpc_limiter.acquire()
                        
                        async with session.post(RPC_EP, json=payload, timeout=5) as rpc_resp:
                            if rpc_resp.status == 200:
                                rpc_data = await rpc_resp.json()
                                account_data = rpc_data.get("result", {}).get("value", {})
                                if account_data and account_data.get("data"):
                                    import base64
                                    raw_data = base64.b64decode(account_data["data"][0])
                                    decoded = await decode_lb_pair(raw_data, pair_address)
                                    
                                    if decoded:
                                        aid, bs, vol_acc = decoded
                                        
                                        # Use decoded values if we don't have them yet
                                        if active_bin_id is None:
                                            active_bin_id = aid
                                        if bin_step_val is None:
                                            bin_step_val = bs
                                        if volatility_accumulator is None:
                                            volatility_accumulator = vol_acc
                                            
                                        # If we still don't have a price, calculate it from active bin
                                        if (price is None or price <= 0) and active_bin_id is not None and bin_step_val is not None:
                                            # Price = (1 + bin_step/10000) ^ (active_id - zero_bin_id)
                                            # DLMM active_id bias is 2^23 (8,388,608)
                                            ACTIVE_ID_BASE = 8388608
                                            
                                            bp = bin_step_val / 10000.0
                                            # Fix: Subtract bias from active_id
                                            true_active_id = active_bin_id - ACTIVE_ID_BASE
                                            calculated_price = (1 + bp) ** true_active_id
                                            
                                            # Sanity check price
                                            if calculated_price > 0 and 1e-15 < calculated_price < 1e15:
                                                price = calculated_price
                                                data_source = "rpc_decode"
                                                # Also populate volume with 0 if missing (better than None)
                                                if volume_1h == 0:
                                                    volume_1h = 0.0
                                                
                    except Exception as e:
                       # logger.debug(f"RPC Decode failed: {e}") 
                       pass

                # ==========================================
                # FINAL CHECK: Ora possiamo scartare se manca ancora il prezzo
                # ==========================================
                if price is None or price <= 0:
                    if tick % 30 == 0:
                        logger.warning(f"[{base_symbol}] No price available from any source (API+RPC). Mint: {mint_address}")
                    continue
                
                # ==========================================
                # RECORD DATA & CALCULATE METRICS
                # ==========================================
                current_time = asyncio.get_event_loop().time()
                
                # Track volatility for volatility_speed
                if volatility_accumulator is not None:
                    volatility_history[mint_address].append((volatility_accumulator, current_time))
                vol_hist = volatility_history[mint_address]
                if len(vol_hist) >= 2:
                    v1, t1 = vol_hist[-2]
                    v2, t2 = vol_hist[-1]
                    dt = t2 - t1
                    if dt > 0:
                        volatility_speed = (v2 - v1) / dt
                
                # Add to price & volume history buffers
                price_history[mint_address].append((price, current_time))
                volume_history[mint_address].append(volume_1h)
                history = price_history[mint_address]
                
                # Extract arrays for calculations
                prices_list = [p[0] for p in history]
                timestamps_list = [p[1] for p in history]
                prices_array = np.array(prices_list)
                timestamps_array = np.array(timestamps_list)
                volumes_array = np.array(list(volume_history[mint_address]))
                
                # ========== FILTRO #3: PREZZO CONGELATO (STD_DEV = 0) ==========
                if tick >= 60:
                    price_std = np.std(prices_array)
                    if price_std < 1e-10:
                        logger.warning(f"⚠ ZOMBIE DETECTED: {base_symbol} prezzo congelato dopo 60s (std_dev=0). KILL.")
                        if mint_address in active_monitors:
                            active_monitors[mint_address] = False
                        return
                
                # ========== CALCULATE ALL METRICS ==========
                metrics = _calculate_all_metrics(
                    mint_address=mint_address,
                    pair_address=pair_address,
                    prices_array=prices_array,
                    timestamps_array=timestamps_array,
                    volumes_array=volumes_array,
                    creation_timestamp=creation_timestamp,
                    current_time=current_time,
                )
                
                # Get holder metrics from state
                holder_metrics = holder_state.get(mint_address, {})
                gini = holder_metrics.get("gini")
                max_holder_dump_val = holder_metrics.get("dump_pct")
                hhi = holder_metrics.get("hhi")
                top3_ratio = holder_metrics.get("top3_ratio")
                holder_trend = holder_metrics.get("holder_trend")
                
                # Calculate approximate latency for tracking
                price_latency_ms = None
                if jupiter_block_id:
                    # Rough estimate: 400ms per slot difference
                    current_slot = 0  # We don't have this, will be NULL
                    price_latency_ms = None  # Can't calculate without current slot
                
                # ========== SAVE TO DATABASE ==========
                async with AsyncSessionLocal() as db_session:
                    # Compatta i dati per il DB - gestisce colonne che potrebbero non esistere
                    market_data_dict = {
                        'mint_address': mint_address,
                        'timestamp': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                        'price_usd': price,
                        'volume_h1': volume_1h,
                        'buys_h1': 0,
                        'sells_h1': 0,
                    }
                    
                    # Prova ad aggiungere campi opzionali se il DB li supporta
                    try:
                        market_data_dict['data_source'] = data_source
                        market_data_dict['block_id'] = jupiter_block_id
                        market_data_dict['price_latency_ms'] = price_latency_ms
                        market_data_dict['active_bin_id'] = active_bin_id
                        market_data_dict['bin_step'] = bin_step_val
                        market_data_dict['volatility_accumulator'] = volatility_accumulator
                        market_data_dict['liquidity_depth_ask'] = liquidity_depth_ask
                        market_data_dict['liquidity_depth_bid'] = liquidity_depth_bid
                        market_data_dict['volatility_speed'] = volatility_speed
                        market_data_dict['price_velocity'] = metrics.get('price_velocity')
                        market_data_dict['price_acceleration'] = metrics.get('price_acceleration')
                        market_data_dict['price_jerk'] = metrics.get('price_jerk')
                        market_data_dict['gini_coefficient'] = gini
                        market_data_dict['shannon_entropy'] = metrics.get('shannon_entropy')
                        market_data_dict['max_holder_dump'] = max_holder_dump_val
                        market_data_dict['hurst_exponent'] = metrics.get('hurst_exponent')
                        market_data_dict['lyapunov_exponent'] = metrics.get('lyapunov_exponent')
                        market_data_dict['fractal_dimension'] = metrics.get('fractal_dimension')
                        market_data_dict['benford_deviation'] = metrics.get('benford_deviation')
                        market_data_dict['jarque_bera_stat'] = metrics.get('jarque_bera_stat')
                        market_data_dict['autocorrelation_lag1'] = metrics.get('autocorrelation_lag1')
                        market_data_dict['kurtosis'] = metrics.get('kurtosis')
                        market_data_dict['skewness'] = metrics.get('skewness')
                        market_data_dict['dominant_frequency'] = metrics.get('dominant_frequency')
                        market_data_dict['spectral_entropy'] = metrics.get('spectral_entropy')
                        market_data_dict['periodicity_score'] = metrics.get('periodicity_score')
                        market_data_dict['kyle_lambda'] = metrics.get('kyle_lambda')
                        market_data_dict['amihud_illiquidity'] = metrics.get('amihud_illiquidity')
                        market_data_dict['roll_spread'] = metrics.get('roll_spread')
                        market_data_dict['holder_herfindahl_index'] = hhi
                        market_data_dict['top3_vs_rest_ratio'] = top3_ratio
                        market_data_dict['holder_count_trend'] = holder_trend
                        market_data_dict['rsi_14'] = metrics.get('rsi_14')
                        market_data_dict['max_drawdown'] = metrics.get('max_drawdown')
                        market_data_dict['time_since_creation_sec'] = metrics.get('time_since_creation_sec')
                        market_data_dict['pump_duration_sec'] = metrics.get('pump_duration_sec')
                        market_data_dict['price_age_correlation'] = metrics.get('price_age_correlation')
                    except Exception:
                        pass  # Se il modello non ha questi campi, ignora
                    
                    market_entry = MarketData(**market_data_dict)
                    db_session.add(market_entry)
                    await db_session.commit()
                
                # 🔥 Calculate real-time dump probability
                dump_prob = calculate_dump_probability(
                    metrics,
                    holder_state.get(mint_address, {}),
                    price,
                    volume_1h
                )
                
                # Alert based on 90% quantile threshold
                if dump_prob >= 0.9:
                    logger.critical(
                        f"🔴🔴🔴 DUMP IMMINENT: {base_symbol}\n"
                        f"   Probability: {dump_prob*100:.0f}%"
                    )
                elif dump_prob >= 0.7:
                    logger.error(f"🔴 HIGH DUMP RISK: {base_symbol} - {dump_prob*100:.0f}%")
                elif dump_prob >= 0.5:
                    logger.warning(f"🟠 ELEVATED RISK: {base_symbol} - {dump_prob*100:.0f}%")
                elif dump_prob >= 0.3 and tick % 30 == 0:
                    logger.info(f"🟡 MODERATE RISK: {base_symbol} - {dump_prob*100:.0f}%")

                # Log every 30 ticks (reduced spam)
                if tick % 30 == 0:
                    n_metrics = len([v for v in metrics.values() if v is not None])
                    h_str = f"{metrics.get('hurst_exponent', 0):.2f}" if metrics.get('hurst_exponent') is not None else "?"
                    r_str = f"{metrics.get('rsi_14', 0):.0f}" if metrics.get('rsi_14') is not None else "?"
                    g_str = f"{gini:.2f}" if gini is not None else "?"
                    logger.info(
                        f"[{base_symbol}] tick={tick} price={price:.8f} "
                        f"H={h_str} RSI={r_str} Gini={g_str} "
                        f"metrics={n_metrics} src={data_source}"
                    )

            except Exception as e:
                logger.error(f"Error in fast monitor for {base_symbol}: {e}")

            finally:
                # Precise Timing (Executes even on 'continue')
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0.0, 1.0 - elapsed)
                await asyncio.sleep(sleep_time)
    
    finally:
        # Cleanup
        active_monitors.pop(mint_address, None)
        holder_task.cancel()
        tick_task.cancel()
        price_history.pop(mint_address, None)
        volume_history.pop(mint_address, None)
        volatility_history.pop(mint_address, None)
        tick_history.pop(pair_address, None)  # Clear tick data
        holder_state.pop(mint_address, None)
        token_creation_times.pop(mint_address, None)
        _meteora_pool_cache.pop(pair_address, None)
        volume_history.pop(mint_address, None)
        volatility_history.pop(mint_address, None)
        holder_state.pop(mint_address, None)
        token_creation_times.pop(mint_address, None)
        _meteora_pool_cache.pop(pair_address, None)
        
        logger.info(f"Fast Monitor Finished for {base_symbol}")


async def holder_check_task(mint_address: str, base_symbol: str, session: aiohttp.ClientSession):
    """
    Background task that checks holder distribution with ADAPTIVE intervals.
    
    🔥 V3 OPTIMIZED: Adaptive intervals for better max_holder_dump detection
    - 0-120s: Every 10 seconds (12 snapshots - critical early detection)
    - 120-300s: Every 30 seconds (6 snapshots - pattern confirmation)
    Total: 18 snapshots (vs old: 5)
    
    Calculates:
    - Gini coefficient on top 20 holder balances
    - Herfindahl-Hirschman Index (HHI) 
    - Top 3 vs rest ratio
    - Holder count trend (change in number of holders)
    - max_holder_dump: % change in top holder's balance (TIER S predictor!)
    """
    logger.info(f"[{base_symbol}] Starting Adaptive Holder Monitor V3")
    
    start_time = asyncio.get_event_loop().time()
    check_count = 0
    prev_holder_count = None
    
    # Initialize holder_count_history for this token
    if mint_address not in holder_state:
        holder_state[mint_address] = {
            'holder_count_history': deque(maxlen=10)
        }
    
    try:
        while mint_address in active_monitors:
            # 🔥 ADAPTIVE INTERVAL based on elapsed time
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if elapsed < 120:  # First 2 minutes - CRITICAL PERIOD
                interval = 10  # High frequency (every 10 seconds)
            else:  # Minutes 2-5
                interval = 30  # Medium frequency (every 30 seconds)
            
            try:
                # Fetch top holders
                holders = await get_token_largest_accounts(session, mint_address)
                
                if holders and len(holders) > 0:
                    check_count += 1
                    current_time = asyncio.get_event_loop().time()
                    
                    # Extract balances (top 20)
                    top_20 = holders[:20]
                    balances = []
                    
                    for holder in top_20:
                        amount = holder.get("uiAmount")
                        if amount is None:
                            raw_amount = holder.get("amount", "0")
                            amount = float(raw_amount)
                        balances.append(amount)
                    
                    balances_array = np.array(balances)
                    
                    # V1: Gini coefficient
                    gini = gini_coefficient(balances_array)
                    
                    # ========== KILL SWITCH: GINI > 0.85 ==========
                    if gini is not None and gini > 0.85:
                        logger.warning(f"⛔ KILL SWITCH: {base_symbol} Gini={gini:.2f} > 0.85 (Extreme concentration = Rug Pull)")
                        if mint_address in active_monitors:
                            active_monitors[mint_address] = False
                        return  # Exit early
                    
                    # V2: Herfindahl-Hirschman Index
                    hhi = herfindahl_index(balances_array)
                    
                    # V2: Top 3 vs rest ratio
                    top3_ratio = top3_vs_rest(balances_array)
                    
                    # V2: Holder count trend
                    current_holder_count = len(holders)
                    holder_trend = None
                    if prev_holder_count is not None:
                        holder_trend = float(current_holder_count - prev_holder_count)
                    prev_holder_count = current_holder_count
                    
                    # Track holder count history for trend calculation
                    holder_state[mint_address]['holder_count_history'].append(
                        (current_time, current_holder_count)
                    )
                    
                    # Calculate holder_count_trend (rate of change)
                    count_history = list(holder_state[mint_address]['holder_count_history'])
                    if len(count_history) >= 3:
                        times = np.array([t for t, _ in count_history])
                        counts = np.array([c for _, c in count_history])
                        time_diffs = times - times[0]
                        
                        if time_diffs[-1] > 0:
                            # Simple slope calculation: holders per second
                            slope = (counts[-1] - counts[0]) / time_diffs[-1]
                            holder_trend = slope
                            
                            # Alert on rapid exodus
                            if slope < -0.1:  # Losing >0.1 holders per second
                                logger.warning(
                                    f"⚠️ RETAIL EXODUS: {base_symbol} - "
                                    f"{abs(slope)*60:.1f} holders/minute leaving"
                                )
                    
                    # Track top holder changes for dump detection
                    top_holder_address = top_20[0].get("address") if top_20 else None
                    top_holder_balance = balances[0] if balances else 0
                    
                    dump_pct = None
                    prev_state = holder_state.get(mint_address)
                    
                    # 🔥 TIER S: Calculate max_holder_dump
                    if prev_state and prev_state.get("top_holder") == top_holder_address:
                        prev_balance = prev_state.get("balance", 0)
                        dump_pct = calculate_holder_dump_percentage(top_holder_balance, prev_balance)
                        
                        # CRITICAL ALERT on significant dumps
                        if dump_pct is not None:
                            if dump_pct < -10:
                                logger.critical(
                                    f"🚨🚨🚨 MAJOR DUMP ALERT: {base_symbol}\n"
                                    f"   Top holder sold {abs(dump_pct):.1f}%\n"
                                    f"   Previous: {prev_balance:,.0f}\n"
                                    f"   Current:  {top_holder_balance:,.0f}\n"
                                    f"   Time since last check: {interval}s"
                                )
                            elif dump_pct < -5:
                                logger.warning(f"⚠️ DUMP WARNING: {base_symbol} - Top holder sold {abs(dump_pct):.1f}%")
                            elif dump_pct > 5:
                                logger.info(f"📈 ACCUMULATION: {base_symbol} - Top holder bought {dump_pct:.1f}%")
                    
                    # Update holder state with all metrics
                    holder_state[mint_address].update({
                        "top_holder": top_holder_address,
                        "balance": top_holder_balance,
                        "gini_coefficient": gini,  # Changed key name for consistency
                        "holder_herfindahl_index": hhi,  # Changed key name
                        "top3_vs_rest_ratio": top3_ratio,  # Changed key name
                        "holder_count_trend": holder_trend,
                        "max_holder_dump": dump_pct,  # Changed key name
                        "holder_count": current_holder_count,
                        "all_balances": balances,
                        "last_check_time": current_time
                    })
                    
                    # Log progress every few checks
                    if check_count % 5 == 0:
                        logger.info(
                            f"[{base_symbol}] Holder Check #{check_count} (interval={interval}s): "
                            f"Holders={current_holder_count}, Gini={gini:.3f}, "
                            f"Dump={dump_pct:+.1f}% if dump_pct else 'N/A', "
                            f"Trend={holder_trend:+.3f}/s" if holder_trend else "Trend=N/A"
                        )
                    
            except Exception as e:
                logger.error(f"[{base_symbol}] Holder check error: {e}")
            
            # 🔥 ADAPTIVE SLEEP
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.debug(f"[{base_symbol}] Holder Check Task cancelled")
        raise
    except Exception as e:
        logger.error(f"[{base_symbol}] Holder check fatal error: {e}")




async def process_new_pair(db_session: AsyncSession, pair: dict, session: aiohttp.ClientSession):
    """Process a single pair: insert DB and trigger snapshot."""
    mint = pair.get("baseToken", {}).get("address")
    if not mint:
        return

    # Filter: Meteora only
    if pair.get("dexId") != "meteora":
        return

    # --- FIX: LOGICA SEMPLIFICATA (Niente pi loop infiniti) ---
    stmt = select(Token).where(Token.mint_address == mint)
    result = await db_session.execute(stmt)
    if result.scalar_one_or_none():
        return # Se esiste gi, NON fare nulla. Ci pensa resume_active_monitors all'avvio.
    # -----------------------------------------------------------

    # Determine creation time
    created_at_ts = pair.get("pairCreatedAt")
    if created_at_ts:
        # pairCreatedAt is in milliseconds
        created_at = datetime.datetime.fromtimestamp(created_at_ts / 1000, tz=datetime.timezone.utc).replace(tzinfo=None)
    else:
        created_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)

    # ========== FILTRO #1: LIQUIDIT MINIMA ($20000) ==========
    # Raised from $2k to $20k to reduce wasted RPC calls on low-liquidity tokens
    # Institutional-grade analysis requires sufficient market depth
    initial_liquidity = float(pair.get("liquidity", {}).get("usd", 0) or 0)
    if initial_liquidity < 20000:
        logger.info(f" SCARTATO {pair.get('baseToken', {}).get('symbol', 'UNKNOWN')}: Liquidit troppo bassa (${initial_liquidity:.0f} < $20000)")
        return
    
    # ========== GATEKEEPER SECURITY CHECKS ==========
    try:
        # 1. Check Mint Authority & Freeze Authority
        is_safe_mint = await gatekeeper.check_mint_security(mint)
        if not is_safe_mint:
            logger.warning(f" Security Check Failed: {mint} has Verify/Freeze Authority enabled.")
            return

        # 2. Check Liquidity Lock
        # Note: pairAddress might not be an LP mint for DLMM, handled inside gatekeeper
        is_liquidity_locked = await gatekeeper.check_liquidity_lock(pair.get("pairAddress"), mint)
        if not is_liquidity_locked:
            logger.warning(f" Security Check Failed: {mint} Liquidity not locked/burned.")
            return

        # 3. Check Supply Concentration (Optional but recommended)
        is_supply_safe = await gatekeeper.check_supply_concentration(mint)
        if not is_supply_safe:
            logger.warning(f" Security Check Failed: {mint} Supply too concentrated.")
            return
            
    except Exception as e:
        logger.error(f"Gatekeeper check error for {mint}: {e}")
        # Fail safe: Skip token on error
        return

    # Insert new token
    try:
        token = Token(
            mint_address=mint,
            pair_address=pair.get("pairAddress"),
            symbol=pair.get("baseToken", {}).get("symbol", "UNKNOWN"),
            created_at=created_at,
            initial_liquidity_usd=initial_liquidity,
            fdv=float(pair.get("fdv", 0) or 0)
        )
        db_session.add(token)
        await db_session.commit()
        logger.info(f" New Token Discovered: {token.symbol} ({mint}) - Liquidity: ${initial_liquidity:.0f}")

        # Trigger Initial Snapshot (Blocking here to ensure it's done before managing too many logic branches,
        # but could be offloaded to a queue in a larger system)
        await take_holder_snapshot(db_session, session, mint)

        # Launch Fast Price Monitor V2 (Non-blocking, passes creation time for time metrics)
        asyncio.create_task(fast_price_monitor(mint, pair.get("pairAddress"), token.symbol, session, created_at))

    except Exception as e:
        logger.error(f"Error saving token {mint}: {e}")
        await db_session.rollback()


async def take_holder_snapshot(db_session: AsyncSession, session: aiohttp.ClientSession, mint_address: str):
    """Forensics Snapshot: Fetch and store top holders."""
    holders = await get_token_largest_accounts(session, mint_address)
    if not holders:
        return

    # Store top 20
    top_20 = holders[:20]
    snapshot_objects = []
    
    for idx, holder in enumerate(top_20):
        # Prefer uiAmount if available for correct decimal scaling
        amount = holder.get("uiAmount")
        if amount is None:
            raw_amount = holder.get("amount", "0")
            amount = float(raw_amount)
            logger.warning(f"Using raw amount for holder {holder.get('address')} of {mint_address}: {raw_amount} (uiAmount missing)")
            
        snapshot_objects.append(HoldersSnapshot(
            mint_address=mint_address,
            holder_address=holder.get("address"),
            amount=amount, 
            rank=idx + 1,
            snapshot_time=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        ))
    
    if snapshot_objects:
        try:
            db_session.add_all(snapshot_objects)
            await db_session.commit()
            logger.info(f"Snapshot taken for {mint_address} ({len(snapshot_objects)} holders)")
        except Exception as e:
            logger.error(f"Error saving snapshot for {mint_address}: {e}")
            await db_session.rollback()


async def scout_loop():
    """Discovery Loop."""
    logger.info("Starting Scout Loop...")
    async with aiohttp.ClientSession() as http_session:
        while True:
            try:
                pairs = await fetch_new_pairs(http_session)
                
                async with AsyncSessionLocal() as db_session:
                    for pair in pairs:
                        # Logic to check age < 1 hour could be done here if pair data has creation time, 
                        # but DexScreener often just gives "recent". We process all new ones we haven't seen.
                        await process_new_pair(db_session, pair, http_session)

            except Exception as e:
                logger.critical(f"Scout Loop Crashed (Retrying): {e}")
                traceback.print_exc()
            
            await asyncio.sleep(30)


async def historian_loop():
    """Historian Loop."""
    logger.info("Starting Historian Loop...")
    async with aiohttp.ClientSession() as http_session:
        while True:
            try:
                async with AsyncSessionLocal() as db_session:
                    # Find tokens < 24h old
                    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)).replace(tzinfo=None)
                    stmt = select(Token).where(Token.created_at >= cutoff)
                    result = await db_session.execute(stmt)
                    tokens = result.scalars().all()
                    
                    if not tokens:
                        await asyncio.sleep(300)
                        continue

                    # Process in chunks to respect API limits if needed, though DexScreener allows multiple pairs
                    # Max 30 tokens per request is a safe bet for generic fetch
                    chunk_size = 30
                    for i in range(0, len(tokens), chunk_size):
                        chunk = tokens[i:i+chunk_size]
                        pairs_str = ",".join([t.pair_address for t in chunk])
                        url = f"https://api.dexscreener.com/latest/dex/pairs/solana/{pairs_str}"
                        
                        try:
                            async with http_session.get(url, timeout=10) as response:
                                if response.status == 200:
                                    if "application/json" not in response.content_type:
                                        logger.warning(f"DexScreener (Historian) invalid content type: {response.content_type}")
                                        continue

                                    try:
                                        data = await response.json()
                                    except Exception as json_err:
                                        logger.error(f"JSON decode error in historian loop: {json_err}")
                                        continue

                                    pairs_data = data.get("pairs", [])
                                    
                                    for p_data in pairs_data:
                                        # Match back to token
                                        base_monitor = next((t for t in chunk if t.pair_address == p_data.get("pairAddress")), None)
                                        if base_monitor:
                                            market_entry = MarketData(
                                                mint_address=base_monitor.mint_address,
                                                timestamp=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
                                                price_usd=float(p_data.get("priceUsd", 0) or 0),
                                                volume_h1=float(p_data.get("volume", {}).get("h1", 0) or 0),
                                                buys_h1=int(p_data.get("txns", {}).get("h1", {}).get("buys", 0) or 0),
                                                sells_h1=int(p_data.get("txns", {}).get("h1", {}).get("sells", 0) or 0)
                                            )
                                            db_session.add(market_entry)
                                    
                                    await db_session.commit()
                                    logger.info(f"Updated market data for {len(pairs_data)} tokens.")
                                else:
                                    logger.warning(f"DexScreener batch returned {response.status}")

                        except aiohttp.ClientError as e:
                            logger.error(f"Network error in historian loop chunk: {e}")
                        except Exception as e:
                            logger.error(f"Error updating market data chunk: {e}")
                
            except Exception as e:
                logger.critical(f"Historian Loop Crashed (Retrying): {e}")
                traceback.print_exc()

            await asyncio.sleep(300) # 5 minutes


async def resume_active_monitors(session: aiohttp.ClientSession):
    """Resume monitoring for tokens created recently (ONCE at startup)."""
    logger.info(" Checking for interrupted monitors...")
    async with AsyncSessionLocal() as db_session:
        # Cerca token pi giovani di 10 minuti
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10)).replace(tzinfo=None)
        stmt = select(Token).where(Token.created_at >= cutoff)
        result = await db_session.execute(stmt)
        tokens = result.scalars().all()
        
        count = 0
        for token in tokens:
             logger.info(f" Resuming Fast Monitor for: {token.symbol}")
             asyncio.create_task(fast_price_monitor(token.mint_address, token.pair_address, token.symbol, session))
             count += 1
        
        logger.info(f" Resumed {count} monitors.")


async def main():
    # Disable DEBUG logs
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    await init_db()
    
    # Creiamo una sessione "Master" per i monitoraggi ripresi
    async with aiohttp.ClientSession() as main_session:
        # 1. Recupera i monitoraggi vecchi (UNA VOLTA SOLA)
        await resume_active_monitors(main_session)
        
        # 2. Lancia i loop normali
        await asyncio.gather(
            scout_loop(),
            historian_loop()
        )

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
