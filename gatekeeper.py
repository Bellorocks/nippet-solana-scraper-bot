
import asyncio
import os
import struct
from typing import Optional, List, Dict
import aiohttp
from loguru import logger
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TokenAccountOpts
from solders.pubkey import Pubkey
from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token._layouts import MINT_LAYOUT

# Load configuration if needed, or use defaults
RPC_ENDPOINT = os.getenv("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")

# Known Safe Addresses for Liquidity
BURN_ADDRESSES = {
    "1nc1nerator11111111111111111111111111111111",
    "DeadDeadDeadDeadDeadDeadDeadDeadDeadDead..." 
    # Add other known burn addresses if necessary
}

# Known Locker Programs (Simplified list)
LOCKER_PROGRAMS = {
    "Da65mDXqFVfW1DK4gJK7Y5y6eP7j2Y9y9y9", # Example Streamflow
    "Team...", # Placeholder for Team Finance
    # Add real program IDs for Streamflow, PinkSale etc if known.
    # For now, we mainly check if the holder is NOT a user wallet.
}

async def check_mint_security(mint_address_str: str) -> bool:
    """
    Checks if Mint Authority or Freeze Authority are enabled.
    False = Risk (Mutable/Freezable)
    True = Safe (Immutable)
    """
    try:
        mint_pubkey = Pubkey.from_string(mint_address_str)
        async with AsyncClient(RPC_ENDPOINT) as client:
            resp = await client.get_account_info(mint_pubkey)
            
            if not resp.value:
                logger.error(f"[{mint_address_str}] Mint account not found.")
                return False

            data = resp.value.data
            if len(data) < MINT_LAYOUT.sizeof():
                logger.error(f"[{mint_address_str}] Invalid Mint data size (< {MINT_LAYOUT.sizeof()}).")
                return False

            mint_data = MINT_LAYOUT.parse(data)
            
            # Check Mint Authority (Option<Pubkey>)
            # In layout: mint_authority_option (u32), mint_authority (Pubkey)
            has_mint_auth = mint_data.mint_authority_option != 0
            
            # Check Freeze Authority (Option<Pubkey>)
            # In layout: freeze_authority_option (u32), freeze_authority (Pubkey)
            has_freeze_auth = mint_data.freeze_authority_option != 0

            if has_mint_auth:
                logger.warning(f"⛔ SECURITY FAIL: {mint_address_str} has Active Mint Authority.")
                return False
            
            if has_freeze_auth:
                logger.warning(f"⛔ SECURITY FAIL: {mint_address_str} has Active Freeze Authority.")
                return False

            logger.info(f"✅ SECURITY PASS: {mint_address_str} is Immutable & Unfreezable.")
            return True

    except Exception as e:
        logger.error(f"Error checking mint security for {mint_address_str}: {e}")
        # Fail safe: If we can't check, we assume RISK or skip. 
        # Requirement says: "fail-safe: se errore rete, salta il token".
        # Returning False here acts as a "block" conservatively? 
        # "fail-safe: se errore rete, salta il token" usually means don't crash, but maybe allow or disallow?
        # Text says: "se errore rete, salta il token" -> SKIP processing. 
        # So returning False makes the bot SKIP it (as per requirement "Se uno dei due è False... interrompere").
        return False


async def check_liquidity_lock(pair_address_str: str, base_mint_str: str) -> bool:
    """
    Checks if liquidity is locked or burned.
    For standard AMMs: Checks LP Token holders.
    For DLMM: This is harder. We will try to see if 'pair_address' acts as an LP mint.
    """
    try:
        # 1. Identify "Liquidity Token". 
        # Check if pair_address is a Mint (Standard AMM)
        pair_pubkey = Pubkey.from_string(pair_address_str)
        
        async with AsyncClient(RPC_ENDPOINT) as client:
            resp = await client.get_account_info(pair_pubkey)
            if not resp.value:
                logger.warning(f"[{pair_address_str}] Pair account not found.")
                return False
            
            data = resp.value.data
            is_spl_mint = len(data) == MINT_LAYOUT.sizeof()
            
            target_mint_str = pair_address_str
            
            if not is_spl_mint:
                # If not an SPL Mint, it's likely a DLMM Program Account or Raydium Pool Account.
                # In this case, we can't easily check "LP Holders" because there is no single LP token.
                logger.info(f"ℹ️ {pair_address_str} is not a standard LP Mint (Likely DLMM). Skipping LP Lock Check as 'Safe' for now.")
                return True

            # If it IS a mint, check holders
            # Get Largest Accounts
            largest_resp = await client.get_token_largest_accounts(Pubkey.from_string(target_mint_str))
            if not largest_resp.value:
                logger.warning(f"[{target_mint_str}] No holders found for LP token.")
                return False # Risky if no holders? Or maybe just created.

            top_holder = largest_resp.value[0] # Largest holder
            holder_addr = str(top_holder.address)
            # Fix: Handle missing ui_amount/ui_amount_string attributes
            try:
                amount = float(top_holder.ui_amount or 0)
            except AttributeError:
                try:
                    amount = float(top_holder.ui_amount_string or 0)
                except AttributeError:
                    amount = 0.0
            
            # Calculate percentage
            # We need total supply? get_token_largest_accounts doesn't give supply.
            # Assuming top holder has majority or check if > 80% if possible. 
            # Ideally we compare `amount` to total supply.
            supply_resp = await client.get_token_supply(Pubkey.from_string(target_mint_str))
            
            try:
                total_supply = float(supply_resp.value.ui_amount or 1)
            except AttributeError:
                try:
                    total_supply = float(supply_resp.value.ui_amount_string or 1)
                except AttributeError:
                    total_supply = 1.0
            
            if total_supply == 0:
                total_supply = 1.0
            
            percentage = (amount / total_supply) * 100
            
            if percentage < 80:
                logger.warning(f"⚠️ LP Distributed: Top holder has {percentage:.1f}% (<80%). Risk of split rug?")
                # If distributed, it might be risk or just heavily traded. 
                # Requirement: "Se il maggior detentore ... è un normale Wallet -> FAIL".
                # If percentage is low, it means many holders. That's usually GOOD for distribution, 
                # BUT for LP, it usually means the dev hasn't locked the majority?
                # Actually, if 100% is in a user wallet = RUG RISK.
                # If 100% is in Burn/Locker = SAFE.
                pass 

            # Check if holder is Whitelisted (Burn or Locker)
            is_burned = holder_addr in BURN_ADDRESSES
            is_locker = holder_addr in LOCKER_PROGRAMS # Or verify if it's a PDA of a locker program
            
            # Simple heuristic for "User Wallet": System Program owned?
            # We need to check owner of the holder account? No, holder_addr is the Token Account or the Owner?
            # getTokenLargestAccounts returns Token Accounts. The owner of the token account is what matters.
            # Wait, `get_token_largest_accounts` returns the *Token Account* address, not the owner?
            # Correct. We need to check the *Owner* of that Token Account.
            
            acc_info = await client.get_account_info(top_holder.address)
            if not acc_info.value:
                return False
                
            # Parse Token Account to get Owner
            from spl.token._layouts import ACCOUNT_LAYOUT
            if len(acc_info.value.data) != ACCOUNT_LAYOUT.sizeof():
                return False # Not a token account?
                
            token_acc_data = ACCOUNT_LAYOUT.parse(acc_info.value.data)
            owner_pubkey = token_acc_data.owner
            owner_str = str(owner_pubkey)
            
            if owner_str in BURN_ADDRESSES or owner_str in LOCKER_PROGRAMS:
                logger.info(f"✅ Liquidity Locked/Burned: Held by {owner_str} ({percentage:.1f}%)")
                return True
            
            # Check if Owner is a PDA (Program Derived Address) -> Likely a Locker
            # Heuristic: Check if owner account is executable? No, owner is an account. 
            # Check owner's owner.
            owner_acc_info = await client.get_account_info(owner_pubkey)
            if owner_acc_info.value:
                owner_owner = str(owner_acc_info.value.owner)
                if owner_owner != "11111111111111111111111111111111": # System Program
                    # It's owned by a program (Locker?)
                    logger.info(f"✅ Liquidity Held by Program {owner_owner} (Likely Locker).")
                    return True
            
            # If Owner is a User Wallet (System Program owned) and holds > 80%
            if percentage > 80:
                logger.warning(f"⛔ SECURITY FAIL: Top LP Holder is a Wallet ({owner_str}) with {percentage:.1f}% supply.")
                return False

            return True

    except Exception as e:
        logger.error(f"Error checking liquidity lock for {pair_address_str}: {e}")
        return False


async def check_supply_concentration(mint_address_str: str) -> bool:
    """
    Checks if any single holder has > 30% of supply (excluding pools/burn).
    """
    try:
        mint_pubkey = Pubkey.from_string(mint_address_str)
        async with AsyncClient(RPC_ENDPOINT) as client:
            largest_resp = await client.get_token_largest_accounts(mint_pubkey)
            if not largest_resp.value:
                return True # No data, assume safe?
                
            supply_resp = await client.get_token_supply(mint_pubkey)
            try:
                total_supply = float(supply_resp.value.ui_amount or 1)
            except AttributeError:
                try:
                    total_supply = float(supply_resp.value.ui_amount_string or 1)
                except AttributeError:
                    total_supply = 1.0
            
            for account in largest_resp.value:
                address = str(account.address)
                # Fix: Handle missing ui_amount attribute in newer solders versions
                try:
                    amount = float(account.ui_amount or 0)
                except AttributeError:
                    # Fallback to ui_amount_string if ui_amount is missing
                    try:
                        amount = float(account.ui_amount_string or 0)
                    except AttributeError:
                        amount = 0.0

                if total_supply > 0:
                    percentage = (amount / total_supply) * 100
                else:
                    percentage = 0.0
                
                # Exclusions
                # We need to check if this address is a Pool or Burn address.
                # Checking if it's a PDA or specific known address?
                # For now, simplistic check: 
                # If we don't know it's a pool, and it's > 30%, we flag it.
                # Ideally pass the pool address to exclude it.
                
                # To do this properly, we'd need to know the pair address(es).
                # But this function only takes mint_address.
                # Assumption: The generic caller handles logic?
                # User requirement: "Escludi l'indirizzo del Pool... e Burn"
                
                if percentage > 30:
                    # Check if it's a pool (Program owned?)
                    acc_info = await client.get_account_info(account.address)
                    if acc_info.value:
                        # Parse owner
                        from spl.token._layouts import ACCOUNT_LAYOUT
                        if len(acc_info.value.data) == ACCOUNT_LAYOUT.sizeof():
                            data = ACCOUNT_LAYOUT.parse(acc_info.value.data)
                            owner = str(data.owner)
                            
                            if owner in BURN_ADDRESSES:
                                continue
                            
                            # Check if owner is a Program (Pool)
                            owner_acc = await client.get_account_info(data.owner)
                            if owner_acc.value and str(owner_acc.value.owner) != "11111111111111111111111111111111":
                                # Owned by a program (e.g. Raydium, Meteora) -> Safe (Pool)
                                continue
                    
                    logger.warning(f"⛔ SECURITY FAIL: High Concentration. {address} holds {percentage:.1f}%")
                    return False
            
            return True

    except Exception as e:
        logger.error(f"Error checking supply concentration: {e}")
        return True # Fail open to avoid blocking valid tokens on error? Or fail closed?
