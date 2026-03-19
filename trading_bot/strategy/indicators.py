from dataclasses import dataclass, field
from trading_bot.types import Candle


@dataclass
class Indicators:
    valid: bool = False
    n_candles: int = 0

    # Moving averages
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0

    # RSI
    rsi_14: float = 50.0

    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0

    # ATR
    atr_14: float = 0.0

    # VWAP
    vwap: float = 0.0

    # ADX
    adx_14: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0

    # Keltner
    kc_upper: float = 0.0
    kc_middle: float = 0.0
    kc_lower: float = 0.0

    # Donchian
    dc_upper: float = 0.0
    dc_lower: float = 0.0
    dc_middle: float = 0.0

    # Stochastic RSI
    stoch_rsi_k: float = 50.0
    stoch_rsi_d: float = 50.0

    # CCI
    cci_20: float = 0.0

    # Williams %R
    williams_r: float = -50.0

    # OBV
    obv: float = 0.0
    obv_sma: float = 0.0

    # Ichimoku
    ichi_tenkan: float = 0.0
    ichi_kijun: float = 0.0
    ichi_senkou_a: float = 0.0
    ichi_senkou_b: float = 0.0
    ichi_chikou: float = 0.0

    # CMF
    cmf_20: float = 0.0

    # MFI
    mfi_14: float = 50.0

    # Squeeze
    squeeze_mom: float = 0.0
    squeeze_on: bool = False

    # ROC
    roc_12: float = 0.0

    # Z-Score
    zscore_20: float = 0.0

    # FVG
    fvg_bull: bool = False
    fvg_bear: bool = False
    fvg_size: float = 0.0

    # Supertrend
    supertrend: float = 0.0
    supertrend_up: bool = True

    # Parabolic SAR
    psar: float = 0.0
    psar_up: bool = True

    # Funding
    funding_rate: float = 0.0
    funding_extreme_long: bool = False
    funding_extreme_short: bool = False

    # Derived signals
    above_sma_200: bool = False
    golden_cross: bool = False
    rsi_oversold: bool = False
    rsi_overbought: bool = False
    bb_squeeze: bool = False
    macd_bullish_cross: bool = False
    adx_trending: bool = False
    kc_squeeze: bool = False
    ichi_bullish: bool = False

    atr_pct_rank: float = 0.0
    range_pct_rank: float = 0.0
    ema12_dist_pct: float = 0.0
    sma20_dist_pct: float = 0.0

    vol_ratio: float = 0.0
    atr_pct: float = 0.0

    consec_green: int = 0
    consec_red: int = 0

    bullish_engulf: bool = False
    bearish_engulf: bool = False
    shooting_star: bool = False
    hammer: bool = False
    doji: bool = False

    macd_hist_incr: bool = False
    macd_hist_decr: bool = False

    di_bull: bool = False
    di_bear: bool = False

    rsi_bull_div: bool = False
    rsi_bear_div: bool = False

    # Aliases
    @property
    def sma(self) -> float:
        return self.sma_20

    @property
    def ema(self) -> float:
        return self.ema_12

    @property
    def ema_fast(self) -> float:
        return self.ema_12

    @property
    def ema_slow(self) -> float:
        return self.ema_26

    @property
    def bb_mid(self) -> float:
        return self.bb_middle


# --- Computation helpers ---

def _sma(values: list[float], period: int) -> float:
    if len(values) < period:
        return 0.0
    return sum(values[-period:]) / period


def _ema_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [sum(values[:period]) / period]
    for i in range(period, len(values)):
        ema_vals.append(values[i] * k + ema_vals[-1] * (1 - k))
    return ema_vals


def _wilder_smooth(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    result = [sum(values[:period]) / period]
    for i in range(period, len(values)):
        result.append((result[-1] * (period - 1) + values[i]) / period)
    return result


def _stddev(values: list[float], period: int) -> float:
    if len(values) < period:
        return 0.0
    subset = values[-period:]
    mean = sum(subset) / period
    variance = sum((x - mean) ** 2 for x in subset) / period
    return variance ** 0.5


def _highest(values: list[float], period: int) -> float:
    if len(values) < period:
        return max(values) if values else 0.0
    return max(values[-period:])


def _lowest(values: list[float], period: int) -> float:
    if len(values) < period:
        return min(values) if values else 0.0
    return min(values[-period:])


def _percentile_rank(values: list[float], current: float, lookback: int = 100) -> float:
    subset = values[-lookback:] if len(values) >= lookback else values
    if not subset:
        return 0.5
    count_below = sum(1 for v in subset if v <= current)
    return count_below / len(subset)


def compute_indicators(candles: list[Candle], funding_rate: float = 0.0) -> Indicators:
    ind = Indicators()
    n = len(candles)
    ind.n_candles = n

    if n < 2:
        return ind

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    opens = [c.open for c in candles]

    price = closes[-1]

    ind.valid = n >= 200

    # --- Moving averages ---
    ind.sma_20 = _sma(closes, 20)
    ind.sma_50 = _sma(closes, 50)
    ind.sma_200 = _sma(closes, 200)

    ema12_series = _ema_series(closes, 12)
    ema26_series = _ema_series(closes, 26)
    ind.ema_12 = ema12_series[-1] if ema12_series else 0.0
    ind.ema_26 = ema26_series[-1] if ema26_series else 0.0

    # --- RSI(14) ---
    ind.rsi_14 = _compute_rsi(closes, 14)

    # --- MACD(12,26,9) ---
    if ema12_series and ema26_series:
        offset = len(ema12_series) - len(ema26_series)
        macd_line_series = [
            ema12_series[offset + i] - ema26_series[i]
            for i in range(len(ema26_series))
        ]
        if macd_line_series:
            ind.macd_line = macd_line_series[-1]
            signal_series = _ema_series(macd_line_series, 9)
            ind.macd_signal = signal_series[-1] if signal_series else 0.0
            ind.macd_histogram = ind.macd_line - ind.macd_signal

            if len(signal_series) >= 2:
                prev_macd = macd_line_series[-2] if len(macd_line_series) >= 2 else 0
                prev_sig = signal_series[-2] if len(signal_series) >= 2 else 0
                ind.macd_bullish_cross = prev_macd <= prev_sig and ind.macd_line > ind.macd_signal

            if len(macd_line_series) >= 2:
                prev_hist = macd_line_series[-2] - (signal_series[-2] if len(signal_series) >= 2 else 0)
                ind.macd_hist_incr = ind.macd_histogram > prev_hist
                ind.macd_hist_decr = ind.macd_histogram < prev_hist

    # --- Bollinger Bands(20, 2.0) ---
    if n >= 20:
        ind.bb_middle = ind.sma_20
        sd = _stddev(closes, 20)
        ind.bb_upper = ind.bb_middle + 2.0 * sd
        ind.bb_lower = ind.bb_middle - 2.0 * sd
        ind.bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_middle if ind.bb_middle else 0

    # --- ATR(14) ---
    ind.atr_14 = _compute_atr(highs, lows, closes, 14)

    # --- VWAP ---
    cum_tpv = 0.0
    cum_vol = 0.0
    for i in range(n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_tpv += tp * volumes[i]
        cum_vol += volumes[i]
    ind.vwap = cum_tpv / cum_vol if cum_vol > 0 else price

    # --- ADX(14) ---
    _compute_adx(ind, highs, lows, closes, 14)

    # --- Keltner Channels(20, 14, 1.5) ---
    ema20_series = _ema_series(closes, 20)
    if ema20_series:
        ind.kc_middle = ema20_series[-1]
        ind.kc_upper = ind.kc_middle + 1.5 * ind.atr_14
        ind.kc_lower = ind.kc_middle - 1.5 * ind.atr_14

    # --- Donchian Channels(20) ---
    if n >= 20:
        ind.dc_upper = _highest(highs, 20)
        ind.dc_lower = _lowest(lows, 20)
        ind.dc_middle = (ind.dc_upper + ind.dc_lower) / 2.0

    # --- Stochastic RSI(14,14,3,3) ---
    _compute_stoch_rsi(ind, closes, 14, 14, 3, 3)

    # --- CCI(20) ---
    if n >= 20:
        typical = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
        tp_sma = _sma(typical, 20)
        mean_dev = sum(abs(typical[-20 + i] - tp_sma) for i in range(20)) / 20
        ind.cci_20 = (typical[-1] - tp_sma) / (0.015 * mean_dev) if mean_dev > 0 else 0

    # --- Williams %R(14) ---
    if n >= 14:
        hh = _highest(highs, 14)
        ll = _lowest(lows, 14)
        ind.williams_r = ((hh - price) / (hh - ll) * -100) if (hh - ll) > 0 else -50

    # --- OBV ---
    obv_vals = _compute_obv(closes, volumes)
    ind.obv = obv_vals[-1] if obv_vals else 0
    ind.obv_sma = _sma(obv_vals, 20) if len(obv_vals) >= 20 else ind.obv

    # --- Ichimoku ---
    _compute_ichimoku(ind, highs, lows, closes)

    # --- CMF(20) ---
    if n >= 20:
        mf_sum = 0.0
        vol_sum = 0.0
        for i in range(n - 20, n):
            hl = highs[i] - lows[i]
            mf_mult = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl if hl > 0 else 0
            mf_sum += mf_mult * volumes[i]
            vol_sum += volumes[i]
        ind.cmf_20 = mf_sum / vol_sum if vol_sum > 0 else 0

    # --- MFI(14) ---
    ind.mfi_14 = _compute_mfi(highs, lows, closes, volumes, 14)

    # --- Squeeze Momentum ---
    ind.squeeze_on = ind.bb_upper < ind.kc_upper and ind.bb_lower > ind.kc_lower
    if ind.kc_middle > 0 and ind.dc_middle > 0:
        avg_mid = (ind.kc_middle + ind.dc_middle) / 2.0
        ind.squeeze_mom = price - avg_mid

    # --- ROC(12) ---
    if n > 12 and closes[-13] != 0:
        ind.roc_12 = (price - closes[-13]) / closes[-13] * 100

    # --- Z-Score(20) ---
    if n >= 20:
        sd20 = _stddev(closes, 20)
        ind.zscore_20 = (price - ind.sma_20) / sd20 if sd20 > 0 else 0

    # --- FVG ---
    if n >= 3:
        if candles[-3].high < candles[-1].low:
            ind.fvg_bull = True
            ind.fvg_size = (candles[-1].low - candles[-3].high) / price * 100
        if candles[-3].low > candles[-1].high:
            ind.fvg_bear = True
            ind.fvg_size = (candles[-3].low - candles[-1].high) / price * 100

    # --- Supertrend(10, 3.0) ---
    _compute_supertrend(ind, highs, lows, closes, 10, 3.0)

    # --- Parabolic SAR(0.02, 0.20, 0.02) ---
    _compute_psar(ind, highs, lows, closes, 0.02, 0.20, 0.02)

    # --- Funding ---
    ind.funding_rate = funding_rate
    ind.funding_extreme_long = funding_rate > 0.0001
    ind.funding_extreme_short = funding_rate < -0.0001

    # --- Derived signals ---
    ind.above_sma_200 = price > ind.sma_200 if ind.sma_200 > 0 else False
    ind.golden_cross = ind.sma_50 > ind.sma_200 if ind.sma_200 > 0 else False
    ind.rsi_oversold = ind.rsi_14 < 30
    ind.rsi_overbought = ind.rsi_14 > 70
    ind.bb_squeeze = ind.bb_width < 0.03 if ind.bb_width > 0 else False
    ind.adx_trending = ind.adx_14 > 25
    ind.kc_squeeze = ind.squeeze_on

    if price > 0:
        ind.ema12_dist_pct = (price - ind.ema_12) / price * 100 if ind.ema_12 > 0 else 0
        ind.sma20_dist_pct = (price - ind.sma_20) / price * 100 if ind.sma_20 > 0 else 0
        ind.atr_pct = ind.atr_14 / price

    vol_sma20 = _sma(volumes, 20) if n >= 20 else (sum(volumes) / n if n > 0 else 1)
    ind.vol_ratio = volumes[-1] / vol_sma20 if vol_sma20 > 0 else 0

    # ATR percentile rank
    if n >= 14:
        atr_vals = _compute_atr_series(highs, lows, closes, 14)
        if atr_vals:
            ind.atr_pct_rank = _percentile_rank(atr_vals, ind.atr_14, 100)

    # Range percentile rank
    ranges = [highs[i] - lows[i] for i in range(n)]
    if ranges:
        ind.range_pct_rank = _percentile_rank(ranges, ranges[-1], 100)

    # Ichimoku bullish
    if ind.ichi_tenkan > 0 and ind.ichi_kijun > 0:
        cloud_top = max(ind.ichi_senkou_a, ind.ichi_senkou_b)
        ind.ichi_bullish = price > cloud_top and ind.ichi_tenkan > ind.ichi_kijun

    # Consecutive candles
    ind.consec_green = 0
    ind.consec_red = 0
    for i in range(n - 1, -1, -1):
        if closes[i] > opens[i]:
            if ind.consec_red > 0:
                break
            ind.consec_green += 1
        elif closes[i] < opens[i]:
            if ind.consec_green > 0:
                break
            ind.consec_red += 1
        else:
            break

    # Candle patterns
    if n >= 2:
        _compute_candle_patterns(ind, candles)

    # DI signals
    ind.di_bull = ind.plus_di > ind.minus_di
    ind.di_bear = ind.minus_di > ind.plus_di

    # RSI divergence (simplified: price new low but RSI higher low, or reverse)
    if n >= 30:
        _compute_rsi_divergence(ind, closes, highs, lows, 14)

    return ind


def _compute_rsi(closes: list[float], period: int) -> float:
    if len(closes) < period + 1:
        return 50.0

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain_series = _wilder_smooth(gains, period)
    avg_loss_series = _wilder_smooth(losses, period)

    if not avg_gain_series or not avg_loss_series:
        return 50.0

    avg_gain = avg_gain_series[-1]
    avg_loss = avg_loss_series[-1]

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_rsi_series(closes: list[float], period: int) -> list[float]:
    """Compute RSI at every point in O(n) using a single Wilder smooth pass."""
    if len(closes) < period + 1:
        return []

    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain_series = _wilder_smooth(gains, period)
    avg_loss_series = _wilder_smooth(losses, period)

    if not avg_gain_series or not avg_loss_series:
        return []

    rsi_series = []
    for i in range(len(avg_gain_series)):
        ag = avg_gain_series[i]
        al = avg_loss_series[i]
        if al == 0:
            rsi_series.append(100.0 if ag > 0 else 50.0)
        else:
            rs = ag / al
            rsi_series.append(100.0 - 100.0 / (1.0 + rs))
    return rsi_series


def _compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int
) -> float:
    series = _compute_atr_series(highs, lows, closes, period)
    return series[-1] if series else 0.0


def _compute_atr_series(
    highs: list[float], lows: list[float], closes: list[float], period: int
) -> list[float]:
    n = len(closes)
    if n < 2:
        return []
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return _wilder_smooth(tr, period)


def _compute_adx(
    ind: Indicators, highs: list[float], lows: list[float],
    closes: list[float], period: int
) -> None:
    n = len(closes)
    if n < period + 1:
        return

    plus_dm = []
    minus_dm = []
    tr_list = []

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(max(up_move, 0) if up_move > down_move else 0)
        minus_dm.append(max(down_move, 0) if down_move > up_move else 0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    smoothed_plus = _wilder_smooth(plus_dm, period)
    smoothed_minus = _wilder_smooth(minus_dm, period)
    smoothed_tr = _wilder_smooth(tr_list, period)

    if not smoothed_plus or not smoothed_minus or not smoothed_tr:
        return

    min_len = min(len(smoothed_plus), len(smoothed_minus), len(smoothed_tr))
    dx_list = []
    for i in range(min_len):
        tr_val = smoothed_tr[i]
        if tr_val == 0:
            dx_list.append(0)
            continue
        pdi = 100 * smoothed_plus[i] / tr_val
        mdi = 100 * smoothed_minus[i] / tr_val
        di_sum = pdi + mdi
        dx_list.append(abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0)

    adx_series = _wilder_smooth(dx_list, period)
    if adx_series:
        ind.adx_14 = adx_series[-1]

    if smoothed_tr[-1] > 0:
        ind.plus_di = 100 * smoothed_plus[-1] / smoothed_tr[-1]
        ind.minus_di = 100 * smoothed_minus[-1] / smoothed_tr[-1]


def _compute_stoch_rsi(
    ind: Indicators, closes: list[float],
    rsi_period: int, stoch_period: int, k_smooth: int, d_smooth: int
) -> None:
    n = len(closes)
    if n < rsi_period + stoch_period:
        return

    # O(n) RSI series instead of O(n^2) loop
    rsi_vals = _compute_rsi_series(closes, rsi_period)

    if len(rsi_vals) < stoch_period:
        return

    stoch_k_raw = []
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1: i + 1]
        hi = max(window)
        lo = min(window)
        if hi - lo > 0:
            stoch_k_raw.append((rsi_vals[i] - lo) / (hi - lo) * 100)
        else:
            stoch_k_raw.append(50.0)

    if len(stoch_k_raw) >= k_smooth:
        k_sma = _sma(stoch_k_raw, k_smooth)
        ind.stoch_rsi_k = k_sma

        if len(stoch_k_raw) >= k_smooth + d_smooth - 1:
            k_smoothed = []
            for i in range(k_smooth - 1, len(stoch_k_raw)):
                k_smoothed.append(_sma(stoch_k_raw[i - k_smooth + 1: i + 1], k_smooth))
            if len(k_smoothed) >= d_smooth:
                ind.stoch_rsi_d = _sma(k_smoothed, d_smooth)


def _compute_obv(closes: list[float], volumes: list[float]) -> list[float]:
    if not closes:
        return []
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv


def _compute_ichimoku(
    ind: Indicators, highs: list[float], lows: list[float], closes: list[float]
) -> None:
    n = len(closes)
    if n < 52:
        return

    ind.ichi_tenkan = (_highest(highs[-9:], 9) + _lowest(lows[-9:], 9)) / 2
    ind.ichi_kijun = (_highest(highs[-26:], 26) + _lowest(lows[-26:], 26)) / 2
    ind.ichi_senkou_a = (ind.ichi_tenkan + ind.ichi_kijun) / 2
    ind.ichi_senkou_b = (_highest(highs[-52:], 52) + _lowest(lows[-52:], 52)) / 2
    ind.ichi_chikou = closes[-1] if n >= 26 else 0


def _compute_mfi(
    highs: list[float], lows: list[float], closes: list[float],
    volumes: list[float], period: int
) -> float:
    n = len(closes)
    if n < period + 1:
        return 50.0

    typical = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    pos_flow = 0.0
    neg_flow = 0.0

    for i in range(n - period, n):
        raw_mf = typical[i] * volumes[i]
        if typical[i] > typical[i - 1]:
            pos_flow += raw_mf
        elif typical[i] < typical[i - 1]:
            neg_flow += raw_mf

    if neg_flow == 0:
        return 100.0 if pos_flow > 0 else 50.0

    mfi_ratio = pos_flow / neg_flow
    return 100.0 - 100.0 / (1.0 + mfi_ratio)


def _compute_supertrend(
    ind: Indicators, highs: list[float], lows: list[float],
    closes: list[float], period: int, multiplier: float
) -> None:
    n = len(closes)
    if n < period:
        return

    atr_series = _compute_atr_series(highs, lows, closes, period)
    if not atr_series:
        return

    offset = n - len(atr_series)
    upper_band = [0.0] * len(atr_series)
    lower_band = [0.0] * len(atr_series)
    supertrend = [0.0] * len(atr_series)
    direction = [1] * len(atr_series)

    for i in range(len(atr_series)):
        idx = offset + i
        hl2 = (highs[idx] + lows[idx]) / 2.0
        upper_band[i] = hl2 + multiplier * atr_series[i]
        lower_band[i] = hl2 - multiplier * atr_series[i]

        if i > 0:
            if lower_band[i] < lower_band[i - 1] and closes[idx - 1] > lower_band[i - 1]:
                lower_band[i] = lower_band[i - 1]
            if upper_band[i] > upper_band[i - 1] and closes[idx - 1] < upper_band[i - 1]:
                upper_band[i] = upper_band[i - 1]

            if direction[i - 1] == 1:
                direction[i] = -1 if closes[idx] < lower_band[i] else 1
            else:
                direction[i] = 1 if closes[idx] > upper_band[i] else -1

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    ind.supertrend = supertrend[-1]
    ind.supertrend_up = direction[-1] == 1


def _compute_psar(
    ind: Indicators, highs: list[float], lows: list[float],
    closes: list[float], af_start: float, af_max: float, af_step: float
) -> None:
    n = len(closes)
    if n < 3:
        return

    is_long = closes[1] > closes[0]
    sar = lows[0] if is_long else highs[0]
    ep = highs[1] if is_long else lows[1]
    af = af_start

    for i in range(2, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)

        if is_long:
            sar = min(sar, lows[i - 1], lows[i - 2])
            if lows[i] < sar:
                is_long = False
                sar = ep
                ep = lows[i]
                af = af_start
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            sar = max(sar, highs[i - 1], highs[i - 2])
            if highs[i] > sar:
                is_long = True
                sar = ep
                ep = highs[i]
                af = af_start
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)

    ind.psar = sar
    ind.psar_up = is_long


def _compute_candle_patterns(ind: Indicators, candles: list[Candle]) -> None:
    c = candles[-1]
    p = candles[-2]

    body = abs(c.close - c.open)
    total_range = c.high - c.low
    prev_body = abs(p.close - p.open)

    if total_range > 0:
        body_pct = body / total_range
        upper_wick = c.high - max(c.open, c.close)
        lower_wick = min(c.open, c.close) - c.low

        ind.doji = body_pct < 0.1
        ind.hammer = lower_wick > 2 * body and upper_wick < body * 0.3 and c.close > c.open
        ind.shooting_star = upper_wick > 2 * body and lower_wick < body * 0.3 and c.close < c.open

    ind.bullish_engulf = (
        p.close < p.open and c.close > c.open and
        c.close > p.open and c.open < p.close
    )
    ind.bearish_engulf = (
        p.close > p.open and c.close < c.open and
        c.close < p.open and c.open > p.close
    )


def _compute_rsi_divergence(
    ind: Indicators, closes: list[float], highs: list[float],
    lows: list[float], period: int
) -> None:
    n = len(closes)
    lookback = min(30, n)

    # Use pre-computed RSI series instead of O(n*lookback) loop
    all_rsi = _compute_rsi_series(closes, period)
    if not all_rsi:
        return
    # Extract the last `lookback+1` RSI values
    rsi_vals = all_rsi[-(lookback + 1):] if len(all_rsi) > lookback else all_rsi

    if len(rsi_vals) < 10:
        return

    recent_lows = lows[-lookback:]
    recent_rsi = rsi_vals[-lookback:] if len(rsi_vals) >= lookback else rsi_vals

    price_low_idx = recent_lows.index(min(recent_lows))
    price_high_idx = [i for i in range(len(highs[-lookback:])) if highs[-lookback:][i] == max(highs[-lookback:])]

    if price_low_idx > 0 and len(recent_rsi) > price_low_idx:
        prior_lows = recent_lows[:price_low_idx]
        if prior_lows:
            prior_low_idx = prior_lows.index(min(prior_lows))
            if (recent_lows[price_low_idx] < prior_lows[prior_low_idx] and
                    len(recent_rsi) > price_low_idx and len(recent_rsi) > prior_low_idx and
                    recent_rsi[price_low_idx] > recent_rsi[prior_low_idx]):
                ind.rsi_bull_div = True

    if price_high_idx:
        hi_idx = price_high_idx[0]
        recent_highs = highs[-lookback:]
        if hi_idx > 0:
            prior_highs = recent_highs[:hi_idx]
            if prior_highs:
                prior_hi_idx = prior_highs.index(max(prior_highs))
                if (recent_highs[hi_idx] > prior_highs[prior_hi_idx] and
                        len(recent_rsi) > hi_idx and len(recent_rsi) > prior_hi_idx and
                        recent_rsi[hi_idx] < recent_rsi[prior_hi_idx]):
                    ind.rsi_bear_div = True
