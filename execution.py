import logging
import asyncio
import aiohttp
import base64
import time
from typing import Optional, List
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Processed
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import MessageV0
from solders.instruction import Instruction
from solders.system_program import TransferParams, transfer

import config

logger = logging.getLogger("ExecutionManager")


class ExecutionManager:
    """
    Handles swap execution via Jupiter V6 API and Jito bundles.
    Includes HYPER-REALISTIC Paper Trading Simulation.
    
    Execution Stack (2026):
    - Data: Direct RPC for reading Meteora DLMM LbPair accounts
    - Execution: Jupiter V6 API for swap routing through DLMM bins
    - Speed: Jito bundles for fast transaction landing
    """
    
    def __init__(self, rpc_client: AsyncClient):
        self.rpc_client = rpc_client
        # Safely handle missing key in simulation
        if hasattr(config, 'PAYER_KEYPAIR'):
            self.payer = config.PAYER_KEYPAIR
        else:
            self.payer = None

    async def _get_tip_account(self) -> Pubkey:
        import random
        return Pubkey.from_string(random.choice(config.JITO_TIP_ACCOUNTS))

    async def _create_tip_instruction(self, tip_amount_sol: float) -> Instruction:
        tip_account = await self._get_tip_account()
        lamports = int(tip_amount_sol * 1_000_000_000)
        return transfer(
            TransferParams(
                from_pubkey=self.payer.pubkey(),
                to_pubkey=tip_account,
                lamports=lamports
            )
        )

    async def _send_jito_bundle(self, transactions: List[VersionedTransaction]) -> bool:
        """
        Sends a bundle to Jito Block Engine.
        """
        encoded_txs = [str(base64.b64encode(bytes(tx)), "utf-8") for tx in transactions]
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [
                encoded_txs,
                {"encoding": "base64"}
            ]
        }

        async with aiohttp.ClientSession() as session:
            try:
                url = f"{config.JITO_BLOCK_ENGINE_URL}/api/v1/bundles"
                async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as response:
                    if response.status != 200:
                        logger.error(f"Jito Error {response.status}: {await response.text()}")
                        return False
                    
                    data = await response.json()
                    if "result" in data:
                        bundle_id = data["result"]
                        logger.info(f"Jito Bundle Sent! ID: {bundle_id}")
                        return True
                    else:
                        logger.error(f"Jito Failed: {data}")
                        return False
            except Exception as e:
                logger.error(f"Jito Exception: {e}")
                return False

    async def execute_ultra_swap(
        self, 
        input_mint: str, 
        output_mint: str, 
        amount_lamports: int, 
        slippage_bps: int = 50
    ) -> Optional[str]:
        """
        Execute a swap using Jupiter Ultra API (Beta/Free).
        
        Ref: https://station.jup.ag/docs/apis/ultra-api
        
        Flow:
        1. GET /order -> Get optimized unsigned transaction (Quote + Swap combined)
        2. Sign transaction locally
        3. POST /execute -> Submit to Jupiter's gasless/RPC-less engine
        
        Returns:
            Transaction Signature (txid) string on success, None on failure
        """
        if not self.payer:
            logger.error("No Private Key loaded. Cannot execute Jupiter swap.")
            return None
        
        headers = {"Content-Type": "application/json"}
        if config.JUPITER_API_KEY:
            headers["x-api-key"] = config.JUPITER_API_KEY
        else:
            logger.error("Missing JUPITER_API_KEY for Ultra API")
            return None
        
        async with aiohttp.ClientSession() as session:
            try:
                # Step 1: Get Order (Unsigned Tx)
                # This endpoint replaces both Quote and Swap from v6
                order_params = {
                    "inputMint": input_mint,
                    "outputMint": output_mint,
                    "amount": str(amount_lamports),
                    "slippageBps": slippage_bps,
                    "userPublicKey": str(self.payer.pubkey()),
                    # Ultra optimizations
                    "computeUnitPriceMicroLamports": "auto",
                    "asLegacyTransaction": "false"
                }
                
                # JUPITER_ORDER_URL = https://api.jup.ag/ultra/v1/order
                order_url = f"{config.JUPITER_ORDER_URL}?" + "&".join(f"{k}={v}" for k, v in order_params.items())
                
                async with session.get(order_url, headers=headers, timeout=10) as order_response:
                    if order_response.status != 200:
                        error_text = await order_response.text()
                        logger.error(f"Jupiter Ultra Order Failed ({order_response.status}): {error_text}")
                        return None
                    
                    order_data = await order_response.json()
                    
                    # Ultra API returns 'transaction' field directly
                    swap_tx_base64 = order_data.get("transaction")
                    if not swap_tx_base64:
                        logger.error("No 'transaction' in Jupiter Ultra response")
                        return None
                        
                    logger.info(f"Jupiter Ultra: Order received for {amount_lamports} input")
                
                # Step 2: Sign Transaction
                tx_bytes = base64.b64decode(swap_tx_base64)
                tx = VersionedTransaction.from_bytes(tx_bytes)
                
                # Sign with payer
                signed_tx = VersionedTransaction(tx.message, [self.payer])
                signed_tx_base64 = base64.b64encode(bytes(signed_tx)).decode("utf-8")
                
                # Step 3: Execute (Gasless Submission)
                # JUPITER_EXECUTE_URL = https://api.jup.ag/ultra/v1/execute
                execute_payload = {
                    "transaction": signed_tx_base64
                }
                
                async with session.post(
                    config.JUPITER_EXECUTE_URL, 
                    json=execute_payload, 
                    headers=headers, 
                    timeout=15
                ) as exec_response:
                    if exec_response.status != 200:
                        error_text = await exec_response.text()
                        logger.error(f"Jupiter Ultra Execute Failed ({exec_response.status}): {error_text}")
                        return None
                    
                    exec_data = await exec_response.json()
                    signature = exec_data.get("signature") or exec_data.get("txid")
                    
                    if signature:
                        logger.info(f"Jupiter Ultra Executed! Sig: {signature}")
                        return signature
                    else:
                        logger.info("Jupiter Ultra executed (no signature returned immediately)")
                        return "submitted"
                
            except Exception as e:
                logger.error(f"Jupiter Ultra swap exception: {e}")
                return None

    async def buy(self, mint: Pubkey, amount_sol: float, current_price: float = 0.0) -> Optional[float]:
        """
        Executes buy via Jupiter Ultra API (Gasless/Fast).
        In PAPER_TRADING: Returns a Simulated Execution Price.
        """
        logger.info(f"Preparing BUY for {mint} with {amount_sol} SOL")
        
        # --- SIMULATION BLOCK ---
        if config.PAPER_TRADING:
            # We assume current_price is passed from Strategy. If 0 (fallback), use mock.
            ref_price = current_price if current_price > 0 else 0.0001
            
            # Slippage: We buy HIGHER
            executed_price = ref_price * (1 + config.SIM_SLIPPAGE_PCT)
            
            est_jito = config.SIM_JITO_TIP
            est_fee = config.SIM_TX_FEE
            
            logger.info(f"[SIMULATION] BUY EXECUTED on {mint}")
            logger.info(f"   Market Price: {ref_price:.8f} SOL")
            logger.info(f"   Exec Price:   {executed_price:.8f} SOL (+{config.SIM_SLIPPAGE_PCT*100}% Slip)")
            logger.info(f"   Est. Costs:   {est_jito + est_fee:.6f} SOL")
            
            return executed_price
        # ------------------------

        if not self.payer:
            logger.error("No Private Key loaded. Cannot trade LIVE.")
            return None

        # LIVE LOGIC: Jupiter Ultra Swap
        try:
            # Convert SOL to lamports
            amount_lamports = int(amount_sol * 1_000_000_000)
            
            # Execute Jupiter Ultra swap: WSOL -> Target Token
            signature = await self.execute_ultra_swap(
                input_mint=str(config.WSOL_MINT),
                output_mint=str(mint),
                amount_lamports=amount_lamports,
                slippage_bps=50  # 0.5% slippage
            )
            
            if signature:
                logger.info(f"BUY successful for {mint} (Sig: {signature})")
                return current_price  # Return rough price since we don't wait for confirmation
            
            logger.error("Buy Failed (Jupiter Ultra)")
            return None

        except Exception as e:
            logger.error(f"Buy Exception: {e}")
            return None

    async def sell(self, mint: Pubkey, percent: int, entry_price: float, current_price: float, position_size_sol: float) -> Optional[float]:
        """
        Executes sell via Jupiter Ultra API (Gasless/Fast).
        In PAPER_TRADING: Returns Simulated Execution Price and logs REAL NET PNL.
        """
        logger.info(f"Preparing SELL {percent}% for {mint}")

        # --- SIMULATION BLOCK ---
        if config.PAPER_TRADING:
            # Slippage: We sell LOWER
            executed_price = current_price * (1 - config.SIM_SLIPPAGE_PCT)
            
            # Gross ROI
            roi_raw = (executed_price - entry_price) / entry_price
            
            # Net PnL Calculation
            exit_value = position_size_sol * (1 + roi_raw)
            
            # Total Round-Trip Costs (Buy + Sell Jito & Fees)
            total_costs = (config.SIM_JITO_TIP * 2) + (config.SIM_TX_FEE * 2)
            
            net_pnl = (exit_value - position_size_sol) - total_costs
            
            logger.info(f"[SIMULATION] SELL EXECUTED on {mint}")
            logger.info(f"   Market Price: {current_price:.8f} SOL")
            logger.info(f"   Exec Price:   {executed_price:.8f} SOL (-{config.SIM_SLIPPAGE_PCT*100}% Slip)")
            logger.info(f"   GROSS ROI:    {roi_raw*100:.2f}%")
            logger.info(f"   NET PnL:      {net_pnl:.4f} SOL (After {total_costs} SOL fees)")
            
            return executed_price
        # ------------------------

        if not self.payer:
            return None

        # LIVE LOGIC: Jupiter Ultra Swap (Token -> WSOL)
        try:
            # Estimate token amount - in production use get_token_accounts_by_owner
            estimated_token_amount = int((position_size_sol / current_price) * (percent / 100) * 1_000_000_000)
            
            signature = await self.execute_ultra_swap(
                input_mint=str(mint),
                output_mint=str(config.WSOL_MINT),
                amount_lamports=estimated_token_amount,
                slippage_bps=100  # 1% slippage for sells
            )
            
            if signature:
                logger.info(f"SELL successful for {mint} (Sig: {signature})")
                return current_price
            
            logger.error("Sell Failed (Jupiter Ultra)")
            return current_price

        except Exception as e:
            logger.error(f"Sell Exception: {e}")
            return current_price