"""
Math utilities for Advanced Alpha Metrics V2.
Uses NumPy and SciPy for efficient calculations.

Includes:
- Chaos Theory: Hurst, Lyapunov, Fractal Dimension
- Statistical: Benford, Jarque-Bera, Autocorrelation
- Spectral: FFT analysis
- Market Microstructure: Kyle's Lambda, Amihud
- Technical: RSI, MACD, Bollinger
- Risk: VaR, Drawdown, Sharpe
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import stats
from scipy.fft import fft, fftfreq


# =============================================================================
# V1: BASIC METRICS
# =============================================================================

def gini_coefficient(balances: np.ndarray) -> float:
    """
    Calculate the Gini coefficient for a set of holder balances.
    
    Args:
        balances: Array of holder balances (must be non-negative)
        
    Returns:
        float: Gini coefficient between 0 (perfect equality) and 1 (maximum inequality)
    """
    if len(balances) == 0:
        return 0.0
    
    balances = np.asarray(balances, dtype=np.float64)
    balances = np.maximum(balances, 0)
    
    n = len(balances)
    if n == 0:
        return 0.0
    
    total = np.sum(balances)
    if total == 0:
        return 0.0
    
    sorted_balances = np.sort(balances)
    weighted_sum = np.sum((np.arange(1, n + 1) * sorted_balances))
    gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
    
    return max(0.0, min(1.0, gini))


def shannon_entropy(prices: np.ndarray) -> float:
    """
    Calculate Shannon entropy on price changes to detect manipulation patterns.
    Low entropy = predictable/manipulated movements.
    """
    if len(prices) < 2:
        return 0.0
    
    prices = np.asarray(prices, dtype=np.float64)
    changes = np.diff(prices)
    
    if len(changes) == 0 or np.all(changes == 0):
        return 0.0
    
    std = np.std(changes)
    if std == 0:
        return 0.0
    
    bins = np.array([-np.inf, -std, -std/3, std/3, std, np.inf])
    hist, _ = np.histogram(changes, bins=bins)
    
    probs = hist / len(changes)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calculate_velocity_acceleration_jerk(
    prices: List[float], 
    timestamps: List[float]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Calculate price velocity (dP/dt), acceleration (dV/dt), and jerk (dA/dt).
    
    Jerk = rate of change of acceleration (3rd derivative).
    High jerk indicates sudden changes in price momentum - potential manipulation.
    """
    if len(prices) < 2 or len(timestamps) < 2:
        return None, None, None
    
    prices = np.asarray(prices, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    
    dt = np.diff(timestamps)
    dt = np.where(dt == 0, 1e-6, dt)
    
    # Velocity (1st derivative)
    dp = np.diff(prices)
    velocities = dp / dt
    velocity = velocities[-1]
    
    # Acceleration (2nd derivative)
    acceleration = None
    jerk = None
    
    if len(velocities) >= 2:
        dv = np.diff(velocities)
        dt_vel = dt[1:]
        accelerations = dv / dt_vel
        acceleration = accelerations[-1]
        
        # Jerk (3rd derivative)
        if len(accelerations) >= 2:
            da = np.diff(accelerations)
            dt_acc = dt_vel[1:]
            jerks = da / dt_acc
            jerk = jerks[-1]
    
    return velocity, acceleration, jerk


# Legacy wrapper for backwards compatibility
def calculate_velocity_acceleration(
    prices: List[float], 
    timestamps: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate price velocity (dP/dt) and acceleration (dV/dt)."""
    vel, acc, _ = calculate_velocity_acceleration_jerk(prices, timestamps)
    return vel, acc


def calculate_holder_dump_percentage(current: float, previous: float) -> Optional[float]:
    """Calculate % change in holder balance."""
    if previous <= 0:
        return None
    return ((current - previous) / previous) * 100


# =============================================================================
# V2: CHAOS & MEMORY THEORY
# =============================================================================

def hurst_exponent(prices: np.ndarray) -> Optional[float]:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) analysis.
    
    H > 0.5: Trending (persistent) - momentum continues
    H = 0.5: Random walk
    H < 0.5: Mean-reverting - tends to reverse
    
    Pump & dump tokens often show H >> 0.5 during pump phase.
    
    Returns 0.5 for constant arrays (random walk assumption).
    """
    if len(prices) < 20:
        return None
    
    try:
        prices = np.asarray(prices, dtype=np.float64)
        
        # Check for constant prices (zero variance)
        if np.std(prices) < 1e-10:
            return 0.5  # Random walk (no trend info)
        
        returns = np.diff(np.log(prices + 1e-10))  # Log returns
        
        if len(returns) < 10:
            return None
        
        # Check for constant returns
        if np.std(returns) < 1e-10:
            return 0.5
        
        n = len(returns)
        
        # Calculate R/S for different time lags
        lags = []
        rs_values = []
        
        for lag in [10, 15, 20] if n >= 20 else [n // 2]:
            if lag < 3 or lag > n:
                continue
                
            # Split into chunks
            num_chunks = n // lag
            if num_chunks < 1:
                continue
                
            rs_chunk = []
            for i in range(num_chunks):
                chunk = returns[i * lag:(i + 1) * lag]
                if len(chunk) < 2:
                    continue
                    
                mean = np.mean(chunk)
                deviations = chunk - mean
                cumsum = np.cumsum(deviations)
                
                R = np.max(cumsum) - np.min(cumsum)  # Range
                S = np.std(chunk, ddof=1)  # Std dev
                
                if S > 1e-10:  # Stricter check
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                lags.append(lag)
                rs_values.append(np.mean(rs_chunk))
        
        if len(lags) < 2:
            # Single estimate
            if rs_values:
                result = np.log(rs_values[0] + 1e-10) / np.log(lags[0])
                if np.isfinite(result):
                    return max(0.0, min(1.0, float(result)))
            return None
        
        # Linear regression: log(R/S) = H * log(n) + c
        log_lags = np.log(lags)
        log_rs = np.log(np.array(rs_values) + 1e-10)
        
        slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
        
        if not np.isfinite(slope):
            return None
        
        return max(0.0, min(1.0, float(slope)))
        
    except (ZeroDivisionError, RuntimeWarning, ValueError, FloatingPointError):
        return None
    except Exception:
        return None


def lyapunov_exponent(prices: np.ndarray, m: int = 3, tau: int = 1) -> Optional[float]:
    """
    Estimate largest Lyapunov exponent using Rosenstein method.
    
    Positive λ: Chaotic system (sensitive to initial conditions)
    Zero λ: Stable/periodic
    Negative λ: Converging (predictable)
    
    High positive values may indicate unstable/manipulated markets.
    
    Returns 0.0 for constant arrays (stable system assumption).
    """
    if len(prices) < 30:
        return None
    
    try:
        prices = np.asarray(prices, dtype=np.float64)
        
        # Check for constant prices (zero variance)
        if np.std(prices) < 1e-10:
            return 0.0  # Stable (no divergence)
        
        n = len(prices)
        
        # Create delay vectors
        vectors = []
        for i in range(n - (m - 1) * tau):
            vec = [prices[i + j * tau] for j in range(m)]
            vectors.append(vec)
        
        vectors = np.array(vectors)
        if len(vectors) < 10:
            return None
        
        # Find nearest neighbors (excluding temporal neighbors)
        divergences = []
        min_temporal_sep = 5
        
        for i in range(len(vectors) - 10):
            # Calculate distances to all other points
            dists = np.linalg.norm(vectors - vectors[i], axis=1)
            dists[:max(0, i - min_temporal_sep)] = np.inf
            dists[i:min(len(dists), i + min_temporal_sep + 1)] = np.inf
            
            j = np.argmin(dists)
            if dists[j] == np.inf:
                continue
            
            # Track divergence over time
            initial_dist = dists[j]
            if initial_dist < 1e-10:
                continue
                
            # Look ahead
            steps = min(5, len(vectors) - max(i, j) - 1)
            if steps < 1:
                continue
                
            final_dist = np.linalg.norm(vectors[i + steps] - vectors[j + steps])
            if final_dist < 1e-10:
                continue
                
            divergences.append(np.log(final_dist / initial_dist) / steps)
        
        if not divergences:
            return None
        
        result = np.mean(divergences)
        
        # Check for NaN or infinite results
        if not np.isfinite(result):
            return None
        
        return float(result)
        
    except (ZeroDivisionError, RuntimeWarning, ValueError, FloatingPointError):
        return None
    except Exception:
        return None


def fractal_dimension(prices: np.ndarray) -> Optional[float]:
    """
    Estimate fractal dimension using box-counting approximation.
    
    D ≈ 1: Smooth line
    D ≈ 1.5: Random walk
    D ≈ 2: Space-filling curve
    
    Artificial pump patterns often have D close to 1 (smooth, unnatural).
    """
    if len(prices) < 10:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    n = len(prices)
    
    # Normalize to [0, 1]
    p_min, p_max = np.min(prices), np.max(prices)
    if p_max - p_min < 1e-10:
        return 1.0  # Flat line
    
    normalized = (prices - p_min) / (p_max - p_min)
    
    # Box counting at different scales
    scales = [2, 3, 4, 5, 8, 10]
    counts = []
    
    for scale in scales:
        if scale >= n:
            continue
        
        # Count boxes needed to cover the curve
        box_size = 1.0 / scale
        boxes = set()
        
        for i in range(n):
            x_box = int(i / n * scale)
            y_box = int(normalized[i] / box_size)
            boxes.add((x_box, min(y_box, scale - 1)))
        
        counts.append((scale, len(boxes)))
    
    if len(counts) < 2:
        return None
    
    # log(N) = D * log(1/r) => D = -log(N) / log(r)
    log_scales = np.log([c[0] for c in counts])
    log_counts = np.log([c[1] for c in counts])
    
    try:
        slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
        return max(1.0, min(2.0, slope))
    except ValueError:
        return None


# =============================================================================
# V2: STATISTICAL ANOMALY DETECTION
# =============================================================================

def benford_deviation(values: np.ndarray) -> Optional[float]:
    """
    Calculate Chi-squared deviation from Benford's Law.
    
    Benford's Law: P(d) = log10(1 + 1/d) for first digit d
    
    High deviation suggests artificial/manipulated numbers.
    Useful for detecting fake volume or suspicious price patterns.
    """
    if len(values) < 10:
        return None
    
    values = np.abs(np.asarray(values, dtype=np.float64))
    values = values[values > 0]
    
    if len(values) < 10:
        return None
    
    # Extract first digits
    first_digits = []
    for v in values:
        # Get first non-zero digit
        s = f"{v:.10e}"
        for c in s:
            if c.isdigit() and c != '0':
                first_digits.append(int(c))
                break
    
    if len(first_digits) < 10:
        return None
    
    # Expected Benford distribution
    expected = {d: np.log10(1 + 1/d) for d in range(1, 10)}
    
    # Observed distribution
    observed = {d: 0 for d in range(1, 10)}
    for d in first_digits:
        if 1 <= d <= 9:
            observed[d] += 1
    
    n = len(first_digits)
    
    # Chi-squared statistic
    chi_sq = 0
    for d in range(1, 10):
        exp_count = expected[d] * n
        obs_count = observed[d]
        if exp_count > 0:
            chi_sq += (obs_count - exp_count) ** 2 / exp_count
    
    return chi_sq


def jarque_bera_test(returns: np.ndarray) -> Optional[float]:
    """
    Jarque-Bera test for normality of returns.
    
    High JB statistic: Returns are NOT normally distributed
    This can indicate fat tails, manipulation, or unusual market behavior.
    """
    if len(returns) < 10:
        return None
    
    returns = np.asarray(returns, dtype=np.float64)
    
    n = len(returns)
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    if std < 1e-10:
        return 0.0
    
    # Standardize
    z = (returns - mean) / std
    
    # Skewness and kurtosis
    s = np.mean(z ** 3)
    k = np.mean(z ** 4) - 3  # Excess kurtosis
    
    # JB statistic
    jb = (n / 6) * (s ** 2 + (k ** 2) / 4)
    
    return jb


def calculate_autocorrelation(prices: np.ndarray, lag: int = 1) -> Optional[float]:
    """
    Calculate autocorrelation at given lag.
    
    High autocorrelation: Price movements are predictable/patterned
    This can indicate algorithmic manipulation or artificial markets.
    """
    if len(prices) < lag + 5:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    returns = np.diff(prices)
    
    if len(returns) < lag + 2:
        return None
    
    n = len(returns)
    mean = np.mean(returns)
    var = np.var(returns)
    
    if var < 1e-10:
        return 0.0
    
    cov = np.mean((returns[:-lag] - mean) * (returns[lag:] - mean))
    
    return cov / var


def calculate_kurtosis(prices: np.ndarray) -> Optional[float]:
    """
    Calculate excess kurtosis of returns.
    
    High kurtosis: Fat tails = more extreme events than normal
    Crypto often has kurtosis > 3, but extremely high may indicate manipulation.
    
    Returns 0.0 for constant arrays (kurtosis of normal distribution).
    """
    if len(prices) < 10:
        return None
    
    try:
        returns = np.diff(np.asarray(prices, dtype=np.float64))
        
        if len(returns) < 5:
            return None
        
        # Check for constant returns (zero variance)
        if np.std(returns) < 1e-10:
            return 0.0  # Normal distribution kurtosis
        
        result = stats.kurtosis(returns, fisher=True)
        
        # Check for NaN or infinite results
        if not np.isfinite(result):
            return None
        
        return float(result)
        
    except (ZeroDivisionError, RuntimeWarning, ValueError, FloatingPointError):
        return None


def calculate_skewness(prices: np.ndarray) -> Optional[float]:
    """
    Calculate skewness of returns.
    
    Negative skew: More crash risk than pump potential
    Positive skew: More pump potential than crash risk
    
    Returns 0.0 for constant arrays (symmetric distribution).
    """
    if len(prices) < 10:
        return None
    
    try:
        returns = np.diff(np.asarray(prices, dtype=np.float64))
        
        if len(returns) < 5:
            return None
        
        # Check for constant returns (zero variance)
        if np.std(returns) < 1e-10:
            return 0.0  # Symmetric distribution
        
        result = stats.skew(returns)
        
        # Check for NaN or infinite results
        if not np.isfinite(result):
            return None
        
        return float(result)
        
    except (ZeroDivisionError, RuntimeWarning, ValueError, FloatingPointError):
        return None


# =============================================================================
# V2: SPECTRAL ANALYSIS (FFT)
# =============================================================================

def spectral_analysis(prices: np.ndarray, sample_rate: float = 1.0) -> Dict[str, Optional[float]]:
    """
    Perform FFT spectral analysis to detect periodic patterns.
    
    Returns:
        dominant_frequency: Main frequency component (Hz) - bot trading detection
        spectral_entropy: Randomness of spectrum (low = patterned)
        periodicity_score: Strength of periodic components (high = algorithmic)
    """
    if len(prices) < 20:
        return {"dominant_frequency": None, "spectral_entropy": None, "periodicity_score": None}
    
    prices = np.asarray(prices, dtype=np.float64)
    
    # Detrend
    detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
    
    # FFT
    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, 1 / sample_rate)
    
    # Only positive frequencies
    positive_mask = xf > 0
    xf_pos = xf[positive_mask]
    power = np.abs(yf[positive_mask]) ** 2
    
    if len(power) == 0 or np.sum(power) < 1e-10:
        return {"dominant_frequency": None, "spectral_entropy": None, "periodicity_score": None}
    
    # Dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = xf_pos[dominant_idx]
    
    # Spectral entropy
    power_norm = power / np.sum(power)
    power_norm = power_norm[power_norm > 0]
    spec_entropy = -np.sum(power_norm * np.log2(power_norm))
    
    # Periodicity score: ratio of top 3 frequencies to total
    top_k = min(3, len(power))
    top_power = np.sum(np.sort(power)[-top_k:])
    periodicity = top_power / np.sum(power)
    
    return {
        "dominant_frequency": dominant_freq,
        "spectral_entropy": spec_entropy,
        "periodicity_score": periodicity
    }


# =============================================================================
# V2: MARKET MICROSTRUCTURE
# =============================================================================

def kyle_lambda(price_changes: np.ndarray, volumes: np.ndarray) -> Optional[float]:
    """
    Calculate Kyle's Lambda - price impact per unit volume.
    
    λ = ΔP / ΔV
    
    High lambda: Low liquidity, easy to manipulate
    Based on Kyle (1985) Nobel-worthy market microstructure work.
    """
    if len(price_changes) < 5 or len(volumes) < 5:
        return None
    
    if len(price_changes) != len(volumes):
        min_len = min(len(price_changes), len(volumes))
        price_changes = price_changes[:min_len]
        volumes = volumes[:min_len]
    
    volumes = np.asarray(volumes, dtype=np.float64)
    price_changes = np.asarray(price_changes, dtype=np.float64)
    
    # Filter out zero volumes
    mask = volumes > 0
    if np.sum(mask) < 3:
        return None
    
    volumes = volumes[mask]
    price_changes = price_changes[mask]
    
    # OLS: ΔP = λ * V + ε
    try:
        slope, _, _, _, _ = stats.linregress(volumes, price_changes)
        return abs(slope)
    except ValueError:
        return None


def amihud_illiquidity(returns: np.ndarray, volumes: np.ndarray) -> Optional[float]:
    """
    Calculate Amihud illiquidity ratio.
    
    ILLIQ = mean(|return| / volume)
    
    High ILLIQ: Market is illiquid, price moves easily
    """
    if len(returns) < 5 or len(volumes) < 5:
        return None
    
    min_len = min(len(returns), len(volumes))
    returns = np.abs(np.asarray(returns[:min_len], dtype=np.float64))
    volumes = np.asarray(volumes[:min_len], dtype=np.float64)
    
    # Filter out zero volumes
    mask = volumes > 0
    if np.sum(mask) < 3:
        return None
    
    ratios = returns[mask] / volumes[mask]
    
    return np.mean(ratios)


def roll_spread(prices: np.ndarray) -> Optional[float]:
    """
    Calculate Roll's implied spread.
    
    Based on negative autocovariance of price changes.
    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    
    High spread indicates hidden transaction costs.
    """
    if len(prices) < 10:
        return None
    
    changes = np.diff(np.asarray(prices, dtype=np.float64))
    
    if len(changes) < 5:
        return None
    
    # Autocovariance at lag 1
    cov = np.cov(changes[:-1], changes[1:])[0, 1]
    
    if cov >= 0:
        return 0.0  # No spread implied
    
    return 2 * np.sqrt(-cov)


# =============================================================================
# V2: HOLDER NETWORK
# =============================================================================

def herfindahl_index(balances: np.ndarray) -> Optional[float]:
    """
    Calculate Herfindahl-Hirschman Index (HHI) for holder concentration.
    
    HHI = Σ(share_i²) where share_i is the market share of holder i
    
    0: Perfect competition
    1: Monopoly (one holder has all)
    """
    if len(balances) == 0:
        return None
    
    balances = np.asarray(balances, dtype=np.float64)
    balances = balances[balances > 0]
    
    if len(balances) == 0:
        return None
    
    total = np.sum(balances)
    if total <= 0:
        return None
    
    shares = balances / total
    hhi = np.sum(shares ** 2)
    
    return hhi


def top3_vs_rest(balances: np.ndarray) -> Optional[float]:
    """
    Calculate ratio of top 3 holders vs the rest.
    
    High ratio: Whales dominate the market.
    """
    if len(balances) < 4:
        return None
    
    balances = np.asarray(balances, dtype=np.float64)
    sorted_bal = np.sort(balances)[::-1]  # Descending
    
    top3 = np.sum(sorted_bal[:3])
    rest = np.sum(sorted_bal[3:])
    
    if rest <= 0:
        return float('inf') if top3 > 0 else 0.0
    
    return top3 / rest


# =============================================================================
# V2: TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss
    
    >70: Overbought (potential reversal down)
    <30: Oversold (potential reversal up)
    """
    if len(prices) < period + 1:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    changes = np.diff(prices)
    
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)
    
    # Simple moving average for first calculation
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss < 1e-10:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: np.ndarray) -> Optional[float]:
    """
    Calculate MACD signal.
    
    Returns:
        1: Bullish crossover
        -1: Bearish crossover
        0: Neutral
    """
    if len(prices) < 26:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    
    # EMA calculation helper
    def ema(data, span):
        alpha = 2 / (span + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        return ema_val
    
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    macd_line = ema12 - ema26
    
    # Signal line (9-period EMA of MACD - simplified)
    # For simplicity, compare to previous MACD
    if len(prices) > 27:
        prev_ema12 = ema(prices[:-1], 12)
        prev_ema26 = ema(prices[:-1], 26)
        prev_macd = prev_ema12 - prev_ema26
        
        if macd_line > 0 and prev_macd <= 0:
            return 1.0  # Bullish crossover
        elif macd_line < 0 and prev_macd >= 0:
            return -1.0  # Bearish crossover
    
    return 0.0


def calculate_bollinger_position(prices: np.ndarray, period: int = 20) -> Optional[float]:
    """
    Calculate position within Bollinger Bands (0-1).
    
    0: At or below lower band
    0.5: At middle (SMA)
    1: At or above upper band
    """
    if len(prices) < period:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    if std < 1e-10:
        return 0.5
    
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    current = prices[-1]
    
    if upper == lower:
        return 0.5
    
    position = (current - lower) / (upper - lower)
    
    return max(0.0, min(1.0, position))


def calculate_vpt(prices: np.ndarray, volumes: np.ndarray) -> Optional[float]:
    """
    Calculate Volume Price Trend (VPT).
    
    VPT = VPT_prev + Volume * (Close - Close_prev) / Close_prev
    
    Measures the relationship between price and volume.
    Divergence from price can signal reversal.
    """
    if len(prices) < 5 or len(volumes) < 5:
        return None
    
    min_len = min(len(prices), len(volumes))
    prices = np.asarray(prices[:min_len], dtype=np.float64)
    volumes = np.asarray(volumes[:min_len], dtype=np.float64)
    
    vpt = 0.0
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            vpt += volumes[i] * (prices[i] - prices[i-1]) / prices[i-1]
    
    return vpt


# =============================================================================
# V2: RISK METRICS
# =============================================================================

def calculate_max_drawdown(prices: np.ndarray) -> Optional[float]:
    """
    Calculate maximum drawdown from peak.
    
    MDD = max((Peak - Trough) / Peak)
    
    High MDD indicates high volatility and risk.
    """
    if len(prices) < 2:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    
    # Running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Drawdowns
    drawdowns = (running_max - prices) / (running_max + 1e-10)
    
    return np.max(drawdowns)


def calculate_var_95(returns: np.ndarray) -> Optional[float]:
    """
    Calculate Value at Risk at 95% confidence.
    
    VaR = percentile 5 of returns
    
    Represents the maximum expected loss in 95% of cases.
    """
    if len(returns) < 10:
        return None
    
    returns = np.asarray(returns, dtype=np.float64)
    
    return np.percentile(returns, 5)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> Optional[float]:
    """
    Calculate Sharpe ratio.
    
    Sharpe = (mean_return - risk_free) / std_return
    
    Higher is better. Negative = worse than risk-free.
    """
    if len(returns) < 5:
        return None
    
    returns = np.asarray(returns, dtype=np.float64)
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    if std_ret < 1e-10:
        return 0.0
    
    return (mean_ret - risk_free_rate) / std_ret


def calculate_calmar_ratio(returns: np.ndarray, prices: np.ndarray) -> Optional[float]:
    """
    Calculate Calmar ratio.
    
    Calmar = Annualized Return / Max Drawdown
    
    Higher is better. Measures return per unit of drawdown risk.
    """
    if len(returns) < 10 or len(prices) < 10:
        return None
    
    mdd = calculate_max_drawdown(prices)
    if mdd is None or mdd < 1e-10:
        return None
    
    # Sort of annualized return? Or just total return over period?
    # Standard Calmar is Annualized Return / Max Drawdown
    # simplified: Total Return / Max Drawdown
    
    # Calculate total return from prices if possible, or sum returns
    if len(prices) >= 2:
        total_return = (prices[-1] - prices[0]) / prices[0]
    else:
        total_return = np.sum(returns)
        
    if mdd == 0:
        return 0.0 # No drawdown = infinite ratio? Or just 0 to be safe.
        
    return total_return / mdd


# =============================================================================
# V2: TIME PATTERNS
# =============================================================================

def calculate_price_age_correlation(
    prices: np.ndarray, 
    timestamps: np.ndarray
) -> Optional[float]:
    """
    Calculate correlation between price and token age.
    
    Negative correlation: Price decreasing with age (dump pattern)
    Positive correlation: Price increasing with age (growth)
    """
    if len(prices) < 10 or len(timestamps) < 10:
        return None
    
    min_len = min(len(prices), len(timestamps))
    prices = np.asarray(prices[:min_len], dtype=np.float64)
    timestamps = np.asarray(timestamps[:min_len], dtype=np.float64)
    
    # Check for constant input (would cause ConstantInputWarning)
    if np.std(prices) < 1e-10 or np.std(timestamps) < 1e-10:
        return None
    
    try:
        corr, _ = stats.pearsonr(timestamps, prices)
        return corr
    except Exception:
        return None


def detect_pump_duration(prices: np.ndarray, threshold: float = 0.1) -> Optional[int]:
    """
    Detect the duration of initial pump phase.
    
    Pump = consecutive increases > threshold
    
    Short pump duration may indicate pump & dump scheme.
    """
    if len(prices) < 5:
        return None
    
    prices = np.asarray(prices, dtype=np.float64)
    
    # Find first significant peak
    max_idx = 0
    max_price = prices[0]
    
    for i, p in enumerate(prices):
        if p > max_price:
            max_price = p
            max_idx = i
        elif max_price > 0 and (max_price - p) / max_price > threshold:
            # Significant drop - pump ended
            break
    
    return max_idx  # Duration in ticks
