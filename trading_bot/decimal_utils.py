from trading_bot.types import Decimal


def float_to_decimal(value: float, scale: int = 8) -> Decimal:
    return Decimal.from_float(value, scale)


def str_to_decimal(value: str, scale: int = 8) -> Decimal:
    return Decimal.from_str(value, scale)


def decimal_to_float(d: Decimal) -> float:
    return d.to_float()


def decimal_to_str(d: Decimal) -> str:
    return d.to_str()


def decimal_add(a: Decimal, b: Decimal) -> Decimal:
    scale = max(a.scale, b.scale)
    va = a.mantissa * (10 ** (scale - a.scale))
    vb = b.mantissa * (10 ** (scale - b.scale))
    return Decimal(va + vb, scale)


def decimal_sub(a: Decimal, b: Decimal) -> Decimal:
    scale = max(a.scale, b.scale)
    va = a.mantissa * (10 ** (scale - a.scale))
    vb = b.mantissa * (10 ** (scale - b.scale))
    return Decimal(va - vb, scale)


def decimal_mul(a: Decimal, b: Decimal) -> Decimal:
    scale = a.scale + b.scale
    mantissa = a.mantissa * b.mantissa
    target_scale = max(a.scale, b.scale)
    if scale > target_scale:
        divisor = 10 ** (scale - target_scale)
        mantissa = round(mantissa / divisor)
        scale = target_scale
    return Decimal(mantissa, scale)


def decimal_div(a: Decimal, b: Decimal, result_scale: int = 8) -> Decimal:
    if b.mantissa == 0:
        raise ZeroDivisionError("decimal division by zero")
    numerator = a.mantissa * (10 ** (result_scale + b.scale - a.scale))
    mantissa = round(numerator / b.mantissa)
    return Decimal(mantissa, result_scale)


def decimal_abs(d: Decimal) -> Decimal:
    return Decimal(abs(d.mantissa), d.scale)


def decimal_neg(d: Decimal) -> Decimal:
    return Decimal(-d.mantissa, d.scale)


def decimal_is_zero(d: Decimal) -> bool:
    return d.mantissa == 0


def decimal_sign(d: Decimal) -> int:
    if d.mantissa > 0:
        return 1
    if d.mantissa < 0:
        return -1
    return 0


def format_price(value: float, sz_decimals: int) -> str:
    return f"{value:.{sz_decimals}f}"


def round_size(size: float, sz_decimals: int) -> float:
    factor = 10 ** sz_decimals
    return int(size * factor) / factor
