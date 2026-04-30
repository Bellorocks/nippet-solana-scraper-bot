import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(AsyncAttrs, DeclarativeBase):
    pass

class Token(Base):
    __tablename__ = "tokens"

    mint_address: Mapped[str] = mapped_column(String, primary_key=True)
    pair_address: Mapped[str] = mapped_column(String, nullable=False)
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, nullable=False)
    initial_liquidity_usd: Mapped[float] = mapped_column(Float, nullable=True)
    fdv: Mapped[float] = mapped_column(Float, nullable=True)
    is_analyzed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    market_data: Mapped[list["MarketData"]] = relationship("MarketData", back_populates="token", cascade="all, delete-orphan")
    holders_snapshots: Mapped[list["HoldersSnapshot"]] = relationship("HoldersSnapshot", back_populates="token", cascade="all, delete-orphan")

class MarketData(Base):
    __tablename__ = "market_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mint_address: Mapped[str] = mapped_column(ForeignKey("tokens.mint_address"), nullable=False)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    
    # --- DATA SOURCE TRACKING (For model training quality control) ---
    data_source: Mapped[str] = mapped_column(String, nullable=True)  # "meteora_api", "jupiter_v3", "rpc_decode"
    block_id: Mapped[int] = mapped_column(Integer, nullable=True)  # Jupiter block ID for timestamp verification
    price_latency_ms: Mapped[int] = mapped_column(Integer, nullable=True)  # Approximate price data latency
    
    price_usd: Mapped[float] = mapped_column(Float, nullable=True)
    volume_h1: Mapped[float] = mapped_column(Float, nullable=True)
    buys_h1: Mapped[int] = mapped_column(Integer, nullable=True)
    sells_h1: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # --- METEORA DLMM AI DATA ---
    # Ref: Meteora DLMM Whitepaper - These fields enable ML prediction of liquidity walls
    active_bin_id: Mapped[int] = mapped_column(Integer, nullable=True)
    bin_step: Mapped[int] = mapped_column(Integer, nullable=True)
    volatility_accumulator: Mapped[float] = mapped_column(Float, nullable=True)
    # Liquidity depth in 5 bins above/below active bin (in quote token units)
    liquidity_depth_ask: Mapped[float] = mapped_column(Float, nullable=True)
    liquidity_depth_bid: Mapped[float] = mapped_column(Float, nullable=True)
    # Volatility dynamics
    volatility_speed: Mapped[float] = mapped_column(Float, nullable=True)  # dVolatility/dt - liquidity shock precursor
    
    # --- ADVANCED ALPHA METRICS ---
    # Price dynamics (calculated from rolling window)
    price_velocity: Mapped[float] = mapped_column(Float, nullable=True)  # dPrice/dt
    price_acceleration: Mapped[float] = mapped_column(Float, nullable=True)  # dVelocity/dt
    price_jerk: Mapped[float] = mapped_column(Float, nullable=True)  # dAcceleration/dt (3rd derivative)
    
    # Holder concentration metrics
    gini_coefficient: Mapped[float] = mapped_column(Float, nullable=True)  # 0-1, higher = more concentrated
    
    # Price manipulation detection
    shannon_entropy: Mapped[float] = mapped_column(Float, nullable=True)  # Low = possible manipulation
    
    # Volume delta tracking
    cvd_integral: Mapped[float] = mapped_column(Float, nullable=True)  # Cumulative (buys - sells)
    
    # Whale dump detection
    max_holder_dump: Mapped[float] = mapped_column(Float, nullable=True)  # Negative % = holder sold
    
    # --- V2: CHAOS & MEMORY THEORY ---
    hurst_exponent: Mapped[float] = mapped_column(Float, nullable=True)  # 0-1, >0.5=trending
    lyapunov_exponent: Mapped[float] = mapped_column(Float, nullable=True)  # Chaos measure
    fractal_dimension: Mapped[float] = mapped_column(Float, nullable=True)  # Price curve complexity
    
    # --- V2: STATISTICAL ANOMALY DETECTION ---
    benford_deviation: Mapped[float] = mapped_column(Float, nullable=True)  # Chi-sq from Benford's Law
    jarque_bera_stat: Mapped[float] = mapped_column(Float, nullable=True)  # Normality test
    autocorrelation_lag1: Mapped[float] = mapped_column(Float, nullable=True)  # Self-correlation
    kurtosis: Mapped[float] = mapped_column(Float, nullable=True)  # Fat tails
    skewness: Mapped[float] = mapped_column(Float, nullable=True)  # Distribution asymmetry
    
    # --- V2: SPECTRAL ANALYSIS (FFT) ---
    dominant_frequency: Mapped[float] = mapped_column(Float, nullable=True)  # Hz - bot detection
    spectral_entropy: Mapped[float] = mapped_column(Float, nullable=True)  # Pattern randomness
    periodicity_score: Mapped[float] = mapped_column(Float, nullable=True)  # Cyclic pattern strength
    
    # --- V2: MARKET MICROSTRUCTURE ---
    kyle_lambda: Mapped[float] = mapped_column(Float, nullable=True)  # Price impact per volume
    amihud_illiquidity: Mapped[float] = mapped_column(Float, nullable=True)  # |Return|/Volume
    roll_spread: Mapped[float] = mapped_column(Float, nullable=True)  # Implied spread
    
    # --- V2: HOLDER NETWORK ---
    holder_herfindahl_index: Mapped[float] = mapped_column(Float, nullable=True)  # HHI concentration
    top3_vs_rest_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # Top 3 dominance
    holder_count_trend: Mapped[float] = mapped_column(Float, nullable=True)  # dHolders/dt
    
    # --- V2: TECHNICAL INDICATORS ---
    rsi_14: Mapped[float] = mapped_column(Float, nullable=True)  # 0-100
    macd_signal: Mapped[float] = mapped_column(Float, nullable=True)  # -1/0/1 crossover
    bollinger_position: Mapped[float] = mapped_column(Float, nullable=True)  # 0-1 band position
    volume_price_trend: Mapped[float] = mapped_column(Float, nullable=True)  # VPT indicator
    
    # --- V2: RISK METRICS ---
    max_drawdown: Mapped[float] = mapped_column(Float, nullable=True)  # Max drop from peak
    var_95: Mapped[float] = mapped_column(Float, nullable=True)  # Value at Risk 95%
    sharpe_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # Risk-adjusted return
    calmar_ratio: Mapped[float] = mapped_column(Float, nullable=True)  # Return/MaxDrawdown
    
    # --- V2: TIME PATTERNS ---
    time_since_creation_sec: Mapped[int] = mapped_column(Integer, nullable=True)  # Token age
    pump_duration_sec: Mapped[int] = mapped_column(Integer, nullable=True)  # Length of initial pump
    price_age_correlation: Mapped[float] = mapped_column(Float, nullable=True)  # Price vs age corr

    token: Mapped["Token"] = relationship("Token", back_populates="market_data")

class HoldersSnapshot(Base):
    __tablename__ = "holders_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mint_address: Mapped[str] = mapped_column(ForeignKey("tokens.mint_address"), nullable=False)
    holder_address: Mapped[str] = mapped_column(String, nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False) # storing as float for simplicity, but BigInt might be better depending on precision needs
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_time: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    token: Mapped["Token"] = relationship("Token", back_populates="holders_snapshots")
