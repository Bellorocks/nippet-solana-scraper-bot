# 🕸️ Nippet - Advanced Solana MEV & Scraper Bot

*(GitHub Topics: `solana`, `mev-bot`, `meteora-dlmm`, `jupiter-api`, `jito-block-engine`, `python-trading-bot`, `crypto-trading`, `microstructure-analysis`, `rug-pull-detection`, `algorithmic-trading`, `defi`, `quantitative-analysis`)*

Nippet is an advanced Solana MEV & Scraper Trading Bot designed specifically for the Meteora DLMM ecosystem. It monitors the creation of new pools, analyzes liquidity patterns and holder behaviors in real-time, and executes high-speed trades via Jupiter and Jito Block Engine.

> ⚠️ **FINANCIAL DISCLAIMER:**
> This software is for **educational and experimental purposes only**. Trading cryptocurrencies, especially newly created meme coins, carries an extremely high level of risk. You can and likely will lose money. The creators, contributors, and authors of Nippet are not responsible for any financial losses incurred from using this software. Always test with paper trading enabled (`PAPER_TRADING = True`) before risking real capital.

## ✨ Features

*   **Meteora DLMM Focus**: Extracts and decodes Meteora DLMM (Dynamic Liquidity Market Maker) pool data instantly.
*   **Advanced Microstructure Metrics**: Calculates Kyle's Lambda, Amihud Illiquidity, and Volume Price Trends directly from tick-by-tick RPC data. It also includes advanced Alpha Metrics V2 using Chaos Theory (Hurst, Lyapunov), Statistical Anomaly Detection (Benford's Law), and Spectral Analysis (FFT).
*   **Gatekeeper & Security Checks**: Protects against rug pulls by analyzing token distribution (Gini coefficient), holder snapshots, and liquidity locks. It actively verifies Mint Authority and Freeze Authority to ensure token immutability.
*   **Lightning Execution**: Integration with Jupiter Ultra API and Jito Block Engine for MEV-protected, high-priority transaction execution.
*   **Simulated Trading Mode**: Built-in paper trading to test strategies hyper-realistically without spending SOL.

## 🏗️ Architecture & Modules

*   `scraper.py`: Main entry point handling asynchronous discovery and historian loops.
*   `execution.py`: Manages trade execution via Jupiter V6/Ultra and Jito bundles.
*   `math_utils.py`: Core mathematical engine for technical indicators, risk metrics, and chaotic pattern recognition.
*   `gatekeeper.py`: Strict security filters preventing interaction with honeypots or highly concentrated supplies.
*   `models.py`: SQLAlchemy asynchronous models mapped to a PostgreSQL database for comprehensive data retention.

## 🛠️ Prerequisites

*   **Python 3.9+**
*   **PostgreSQL Database**
*   **Solana RPC URL**: A reliable endpoint such as QuickNode or Helius (public endpoints are heavily rate-limited).
*   **Jupiter API Key**: Available for free at portal.jup.ag.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Nippet.git](https://github.com/yourusername/Nippet.git)
   cd Nippet
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and fill in your `JUPITER_API_KEY`, `RPC_ENDPOINT`, `DB_URL`, and optionally your `PRIVATE_KEY` for live trading.

## ⚙️ Configuration & Usage

All core strategy parameters, execution endpoints, and paper trading toggles are centrally managed in `config.py`. Modify variables such as `ENTRY_SIZE_SOL`, `TP_PERCENT`, and `SL_PERCENT` to fit your risk profile.

To run the scraper and bot, ensure your PostgreSQL database is running, then execute:
```bash
python scraper.py
```

To export collected database metrics for external analysis, you can run:
```bash
python export.py
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details. Copyright (c) 2026 Nippet Contributors.
