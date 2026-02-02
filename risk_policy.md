# Risk Policy

**Version:** 1.0
**Status:** DRAFT - Pending Review
**Purpose:** Define position sizing, exposure limits, and risk controls for all strategies

---

## 1. Core Principles

1. **Capital Preservation First:** Never risk more than acceptable drawdown limits
2. **Size Positions by Risk, Not Conviction:** Volatile assets get smaller positions
3. **Defense in Depth:** Multiple independent risk controls
4. **No Live Trading by Default:** All code runs in backtest/paper mode unless explicitly enabled

---

## 2. Position Sizing Methods

### 2.1 Volatility Targeting (Default)

The primary position sizing method. Target a specific portfolio volatility.

```python
def volatility_target_position(
    signal: float,           # -1 to 1 signal strength
    price: float,            # Current asset price
    volatility: float,       # Annualized asset volatility (e.g., 0.25 = 25%)
    target_vol: float,       # Target portfolio volatility (default: 0.10 = 10%)
    capital: float,          # Total portfolio capital
    max_position_pct: float = 0.20  # Max single position as % of capital
) -> float:
    """
    Calculate position size targeting specific portfolio volatility.

    Returns: Number of shares (positive=long, negative=short)
    """
    # Base position size from volatility targeting
    # If asset vol = 25% and target = 10%, we want 40% of capital in the asset
    vol_scalar = target_vol / volatility if volatility > 0 else 0

    # Apply signal (reduces position for weak signals)
    scaled_size = vol_scalar * abs(signal)

    # Cap at maximum position size
    position_pct = min(scaled_size, max_position_pct)

    # Convert to dollar amount
    position_value = capital * position_pct

    # Convert to shares
    shares = position_value / price

    # Apply signal direction
    return shares * np.sign(signal) if signal != 0 else 0
```

### 2.2 Fixed Fractional (Alternative)

Simpler method for single-asset strategies.

```python
def fixed_fractional_position(
    signal: float,
    price: float,
    capital: float,
    fraction: float = 0.02  # Risk 2% of capital per trade
) -> float:
    """
    Fixed percentage of capital at risk per trade.
    """
    position_value = capital * fraction * abs(signal)
    return (position_value / price) * np.sign(signal)
```

### 2.3 Kelly Criterion (Research Only)

Use half-Kelly for practical application due to parameter uncertainty.

```python
def half_kelly_position(
    win_rate: float,         # Historical win rate
    avg_win: float,          # Average winning return
    avg_loss: float,         # Average losing return (positive number)
    capital: float,
    price: float,
    max_position_pct: float = 0.25
) -> float:
    """
    Half-Kelly position sizing.
    """
    # Full Kelly fraction
    kelly = (win_rate / avg_loss) - ((1 - win_rate) / avg_win)

    # Half Kelly for safety
    half_kelly = kelly / 2

    # Ensure non-negative and capped
    position_pct = max(0, min(half_kelly, max_position_pct))

    return (capital * position_pct) / price
```

---

## 3. Exposure Limits

### 3.1 Single Position Limits

| Limit Type | Default | Hard Cap | Description |
|------------|---------|----------|-------------|
| Max Single Long | 20% | 30% | Max capital in one long position |
| Max Single Short | 10% | 15% | Max capital in one short position |
| Concentration Warning | 15% | - | Log warning if exceeded |

### 3.2 Portfolio-Level Limits

| Limit Type | Default | Hard Cap | Description |
|------------|---------|----------|-------------|
| Max Gross Exposure | 100% | 150% | Sum of abs(positions) |
| Max Net Exposure | 100% | 100% | Long - Short |
| Max Long Exposure | 100% | 120% | Sum of long positions |
| Max Short Exposure | 0% | 50% | Sum of abs(short positions) |
| Target Gross Exposure | 80% | - | Typical operating level |

### 3.3 Sector Limits (If Applicable)

| Limit Type | Default | Description |
|------------|---------|-------------|
| Max Sector Exposure | 30% | Max in any GICS sector |
| Max Industry Exposure | 15% | Max in any industry group |

---

## 4. Drawdown Controls

### 4.1 Drawdown Limits

| Threshold | Action | Recovery |
|-----------|--------|----------|
| -5% | Reduce position sizes by 25% | Resume at -3% |
| -10% | Reduce position sizes by 50% | Resume at -7% |
| -15% | Reduce to 25% target exposure | Resume at -10% |
| -20% | **KILL SWITCH** - Flatten all | Manual restart required |

### 4.2 Drawdown Control Implementation

```python
class DrawdownController:
    def __init__(
        self,
        warning_threshold: float = 0.05,    # -5%
        reduce_threshold: float = 0.10,     # -10%
        severe_threshold: float = 0.15,     # -15%
        kill_threshold: float = 0.20        # -20%
    ):
        self.thresholds = {
            'warning': warning_threshold,
            'reduce': reduce_threshold,
            'severe': severe_threshold,
            'kill': kill_threshold
        }
        self.high_water_mark = 0
        self.killed = False

    def update(self, portfolio_value: float) -> float:
        """
        Update HWM and return position size multiplier.

        Returns: Multiplier between 0 and 1 to apply to positions
        """
        if self.killed:
            return 0.0

        self.high_water_mark = max(self.high_water_mark, portfolio_value)
        drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark

        if drawdown >= self.thresholds['kill']:
            self.killed = True
            logger.critical(f"KILL SWITCH ACTIVATED at {drawdown:.1%} drawdown")
            return 0.0
        elif drawdown >= self.thresholds['severe']:
            logger.error(f"SEVERE drawdown {drawdown:.1%} - reducing to 25% exposure")
            return 0.25
        elif drawdown >= self.thresholds['reduce']:
            logger.warning(f"Drawdown {drawdown:.1%} - reducing to 50% exposure")
            return 0.50
        elif drawdown >= self.thresholds['warning']:
            logger.warning(f"Drawdown warning at {drawdown:.1%}")
            return 0.75
        else:
            return 1.0

    def reset_kill_switch(self):
        """Manual reset after kill switch - requires explicit action."""
        self.killed = False
        logger.info("Kill switch manually reset")
```

---

## 5. Stop-Loss Rules

### 5.1 Position-Level Stops

| Stop Type | Default | Description |
|-----------|---------|-------------|
| Hard Stop | -8% | Exit if position down 8% from entry |
| Trailing Stop | -5% | Exit if position down 5% from peak |
| Time Stop | 20 days | Exit if position not profitable after 20 days |

### 5.2 Stop Implementation

```python
def check_stop_loss(
    entry_price: float,
    current_price: float,
    peak_price: float,
    days_held: int,
    hard_stop: float = 0.08,
    trailing_stop: float = 0.05,
    time_stop: int = 20
) -> tuple[bool, str]:
    """
    Check if position should be stopped out.

    Returns: (should_exit, reason)
    """
    # Hard stop from entry
    loss_from_entry = (entry_price - current_price) / entry_price
    if loss_from_entry >= hard_stop:
        return True, f"Hard stop: {loss_from_entry:.1%} loss from entry"

    # Trailing stop from peak
    loss_from_peak = (peak_price - current_price) / peak_price
    if loss_from_peak >= trailing_stop:
        return True, f"Trailing stop: {loss_from_peak:.1%} from peak"

    # Time stop
    if days_held >= time_stop and current_price <= entry_price:
        return True, f"Time stop: {days_held} days held, not profitable"

    return False, ""
```

---

## 6. Live Trading Safeguards

### 6.1 Mode Configuration

```python
class TradingMode(Enum):
    BACKTEST = "backtest"      # Historical simulation only
    PAPER = "paper"            # Live data, simulated execution
    LIVE = "live"              # Real money (REQUIRES EXPLICIT ENABLE)

# Default is ALWAYS backtest
DEFAULT_MODE = TradingMode.BACKTEST
```

### 6.2 Live Trading Gate

```python
def check_live_trading_enabled() -> bool:
    """
    Multi-factor check before any live trade can execute.

    ALL conditions must be True:
    1. LIVE_TRADING env var is "true"
    2. Config file has live_enabled: true
    3. Interactive confirmation received
    """
    # Check 1: Environment variable
    env_enabled = os.environ.get("LIVE_TRADING", "").lower() == "true"
    if not env_enabled:
        return False

    # Check 2: Config file
    config_path = Path("config/trading.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            config_enabled = config.get("live_enabled", False)
    else:
        config_enabled = False

    if not config_enabled:
        return False

    # Check 3: Interactive confirmation
    if sys.stdin.isatty():
        response = input(
            "\n⚠️  LIVE TRADING CONFIRMATION ⚠️\n"
            "You are about to execute REAL trades with REAL money.\n"
            "Type 'I UNDERSTAND THE RISKS' to proceed: "
        )
        if response != "I UNDERSTAND THE RISKS":
            print("Live trading NOT confirmed. Exiting.")
            return False
    else:
        # Non-interactive - require explicit bypass flag
        if os.environ.get("LIVE_TRADING_NONINTERACTIVE") != "CONFIRMED":
            return False

    return True
```

### 6.3 Paper Trading Mode

```python
class PaperBroker:
    """
    Simulated broker for paper trading with realistic behavior.
    """
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.positions = {}
        self.order_history = []

    def submit_order(self, symbol: str, quantity: int, order_type: str = "MARKET"):
        """
        Submit a paper order.
        Uses delayed execution to simulate real market behavior.
        """
        # Simulate order delay (1-3 seconds in paper mode)
        time.sleep(random.uniform(1, 3))

        # Get current price (from live feed in paper mode)
        price = self.get_current_price(symbol)

        # Apply slippage
        slippage = random.uniform(0.001, 0.003)  # 0.1-0.3%
        if quantity > 0:
            fill_price = price * (1 + slippage)
        else:
            fill_price = price * (1 - slippage)

        # Execute
        self._execute_fill(symbol, quantity, fill_price)

        return {"status": "FILLED", "fill_price": fill_price}
```

---

## 7. Risk Monitoring and Alerts

### 7.1 Monitoring Metrics

| Metric | Check Frequency | Alert Threshold |
|--------|-----------------|-----------------|
| Portfolio Drawdown | Real-time | > 5% |
| Position Concentration | After each trade | > 20% single |
| Gross Exposure | After each trade | > 100% |
| Daily P&L | End of day | < -2% |
| Sharpe (rolling 20d) | Daily | < 0 |

### 7.2 Alert Actions

```python
class RiskMonitor:
    def check_all_limits(self, portfolio: Portfolio) -> list[Alert]:
        alerts = []

        # Check drawdown
        dd = portfolio.current_drawdown
        if dd > 0.10:
            alerts.append(Alert("CRITICAL", f"Drawdown at {dd:.1%}"))
        elif dd > 0.05:
            alerts.append(Alert("WARNING", f"Drawdown at {dd:.1%}"))

        # Check concentration
        for position in portfolio.positions:
            weight = position.value / portfolio.total_value
            if weight > 0.20:
                alerts.append(Alert("WARNING", f"{position.symbol} at {weight:.1%}"))

        # Check exposure
        gross = portfolio.gross_exposure
        if gross > 1.0:
            alerts.append(Alert("WARNING", f"Gross exposure at {gross:.0%}"))

        return alerts
```

---

## 8. Emergency Procedures

### 8.1 Kill Switch Activation

When triggered (drawdown > 20% or manual):
1. **Immediately** cancel all open orders
2. **Immediately** flatten all positions at market
3. Log all actions with timestamps
4. Send alert notifications
5. Disable automated trading
6. Require manual review before restart

### 8.2 System Failure Response

| Failure Type | Response |
|--------------|----------|
| Data feed lost | Hold positions, no new trades |
| Broker connection lost | Alert and monitor, manual intervention if needed |
| Strategy error | Stop strategy, hold existing positions |
| Infrastructure failure | Fail-safe to cash (if possible) |

---

## 9. Compliance Notes

### 9.1 Regulatory Considerations
- Pattern day trader rules (>4 day trades in 5 days requires $25K minimum)
- Wash sale rules for tax purposes
- Short selling restrictions (uptick rule, hard-to-borrow)

### 9.2 Record Keeping
All trades must be logged with:
- Timestamp (entry and exit)
- Symbol
- Quantity
- Price
- Commission/fees
- Reason for trade
- Strategy that generated signal

---

## Appendix: Default Configuration

```yaml
# config/risk_policy.yaml
risk_policy:
  # Position sizing
  position_sizing:
    method: "volatility_target"
    target_volatility: 0.10
    max_single_position: 0.20

  # Exposure limits
  exposure:
    max_gross: 1.00
    max_net: 1.00
    max_single_long: 0.20
    max_single_short: 0.10

  # Drawdown controls
  drawdown:
    warning: 0.05
    reduce: 0.10
    severe: 0.15
    kill: 0.20

  # Stop losses
  stops:
    hard_stop: 0.08
    trailing_stop: 0.05
    time_stop_days: 20

  # Trading mode (NEVER change to live without explicit action)
  mode: "backtest"
```

---

*This policy must be enforced for all trading strategies. Violations should be logged and reviewed.*
