from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class TIF(Enum):
    GTC = "gtc"
    IOC = "ioc"
    ALO = "alo"


class OrderType(Enum):
    LIMIT = "limit"
    TRIGGER = "trigger"


class TPSL(Enum):
    TP = "tp"
    SL = "sl"


class Grouping(Enum):
    NA = 0
    NORMAL_TPSL = 1
    POS_TPSL = 2


@dataclass
class Decimal:
    mantissa: int
    scale: int

    def to_float(self) -> float:
        return self.mantissa / (10 ** self.scale)

    def to_str(self) -> str:
        if self.scale == 0:
            return str(self.mantissa)
        sign = "-" if self.mantissa < 0 else ""
        abs_m = abs(self.mantissa)
        divisor = 10 ** self.scale
        integer_part = abs_m // divisor
        frac_part = abs_m % divisor
        frac_str = str(frac_part).zfill(self.scale)
        return f"{sign}{integer_part}.{frac_str}"

    @staticmethod
    def from_float(value: float, scale: int = 8) -> "Decimal":
        mantissa = round(value * (10 ** scale))
        return Decimal(mantissa=mantissa, scale=scale)

    @staticmethod
    def from_str(value: str, scale: int = 8) -> "Decimal":
        if "." in value:
            parts = value.split(".")
            actual_scale = len(parts[1])
            mantissa = int(parts[0]) * (10 ** actual_scale) + (
                int(parts[1]) if int(parts[0]) >= 0 and not value.startswith("-")
                else -int(parts[1])
            )
            if actual_scale != scale:
                mantissa = round(mantissa * (10 ** (scale - actual_scale)))
        else:
            mantissa = int(value) * (10 ** scale)
        return Decimal(mantissa=mantissa, scale=scale)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Decimal):
            return self.to_float() == other.to_float()
        if isinstance(other, (int, float)):
            return self.to_float() == float(other)
        return NotImplemented

    def __lt__(self, other: "Decimal | int | float") -> bool:
        if isinstance(other, Decimal):
            return self.to_float() < other.to_float()
        return self.to_float() < float(other)

    def __le__(self, other: "Decimal | int | float") -> bool:
        if isinstance(other, Decimal):
            return self.to_float() <= other.to_float()
        return self.to_float() <= float(other)

    def __gt__(self, other: "Decimal | int | float") -> bool:
        if isinstance(other, Decimal):
            return self.to_float() > other.to_float()
        return self.to_float() > float(other)

    def __ge__(self, other: "Decimal | int | float") -> bool:
        if isinstance(other, Decimal):
            return self.to_float() >= other.to_float()
        return self.to_float() >= float(other)

    def __repr__(self) -> str:
        return f"Decimal({self.to_str()})"


def _zero_dec() -> Decimal:
    return Decimal(0, 8)


@dataclass
class OrderRequest:
    asset: int
    coin: str
    side: Side
    price: Decimal
    size: Decimal
    reduce_only: bool = False
    order_type: OrderType = OrderType.LIMIT
    tif: TIF = TIF.ALO
    is_market: bool = False
    trigger_px: Decimal | None = None
    tpsl: TPSL | None = None
    cloid: str = ""
    grouping: Grouping = Grouping.NA


@dataclass
class Order:
    oid: int
    asset: int
    coin: str
    side: Side
    limit_px: Decimal
    sz: Decimal
    orig_sz: Decimal
    timestamp_ms: int
    reduce_only: bool = False
    tif: TIF = TIF.GTC
    cloid: str = ""


@dataclass
class Fill:
    coin: str
    px: Decimal
    sz: Decimal
    side: Side
    time_ms: int
    closed_pnl: Decimal
    fee: Decimal
    oid: int
    tid: int
    crossed: bool = False
    hash: str = ""


@dataclass
class Position:
    coin: str
    asset: int = 0
    size: Decimal = field(default_factory=_zero_dec)
    entry_px: Decimal = field(default_factory=_zero_dec)
    unrealized_pnl: Decimal = field(default_factory=_zero_dec)
    realized_pnl: Decimal = field(default_factory=_zero_dec)
    liquidation_px: Decimal = field(default_factory=_zero_dec)
    margin_used: Decimal = field(default_factory=_zero_dec)
    leverage: int = 1
    is_cross: bool = False


@dataclass
class Candle:
    time_open: int
    time_close: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    n_trades: int = 0


@dataclass
class BookLevel:
    px: Decimal
    sz: Decimal
    n_orders: int = 0


@dataclass
class Book:
    coin: str
    bids: list[BookLevel]
    asks: list[BookLevel]
    timestamp_ms: int = 0


@dataclass
class Mid:
    coin: str
    mid: Decimal


@dataclass
class Account:
    account_value: Decimal
    total_margin_used: Decimal
    total_unrealized_pnl: Decimal
    withdrawable: Decimal
    positions: list[Position]


@dataclass
class AssetMeta:
    name: str
    asset_id: int
    sz_decimals: int


@dataclass
class AssetCtx:
    coin: str
    funding_rate: float
    premium: float
    open_interest: float
    mark_px: float
    day_ntl_vlm: float = 0.0
    valid: bool = False


@dataclass
class AssetSentiment:
    score: float
    confidence: float
    reason: str


@dataclass
class SentimentResult:
    scores: dict[str, AssetSentiment]
    market_regime: str
    key_events: list[str]
    fetched_at: datetime
