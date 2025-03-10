import logging
from decimal import Decimal
from math import ceil, floor
from typing import Dict, List

import numpy as np
import pandas as pd

from hummingbot.connector.derivative.position import Position
from hummingbot.connector.derivative_base import DerivativeBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from .data_types import PriceSize, Proposal
from .perpetual_market_making_order_tracker import PerpetualMarketMakingOrderTracker
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.strategy.utils import order_age
from hummingbot.strategy.order_book_asset_price_delegate import OrderBookAssetPriceDelegate


NaN = float("nan")
s_decimal_zero = Decimal(0)
s_decimal_neg_one = Decimal(-1)


class PerpetualMarketMakingStrategy(StrategyPyBase):
    OPTION_LOG_CREATE_ORDER = 1 << 3
    OPTION_LOG_MAKER_ORDER_FILLED = 1 << 4
    OPTION_LOG_STATUS_REPORT = 1 << 5
    OPTION_LOG_ALL = 0x7fffffffffffffff
    _logger = None

    @classmethod
    def logger(cls):
        if cls._logger is None:
            cls._logger = logging.getLogger(__name__)
        return cls._logger

    def init_params(
        self,
        market_info: MarketTradingPairTuple,
        leverage: int,
        position_mode: str,
        bid_spread: Decimal,
        ask_spread: Decimal,
        order_qty: Decimal = Decimal("1"),
        grid_num: int = 10,
        # The following parameters are kept for compatibility but are not used in the core grid logic.
        long_profit_taking_spread: Decimal = s_decimal_zero,
        short_profit_taking_spread: Decimal = s_decimal_zero,
        stop_loss_spread: Decimal = s_decimal_zero,
        time_between_stop_loss_orders: float = 60.0,
        stop_loss_slippage_buffer: Decimal = s_decimal_zero,
        order_level_spread: Decimal = s_decimal_zero,
        order_level_amount: Decimal = s_decimal_zero,
        order_refresh_time: float = 30.0,
        order_refresh_tolerance_pct: Decimal = s_decimal_neg_one,
        filled_order_delay: float = 60.0,
        order_optimization_enabled: bool = False,
        ask_order_optimization_depth: Decimal = s_decimal_zero,
        bid_order_optimization_depth: Decimal = s_decimal_zero,
        asset_price_delegate=None,
        price_type: str = "mid_price",
        price_ceiling: Decimal = s_decimal_neg_one,
        price_floor: Decimal = s_decimal_neg_one,
        logging_options: int = OPTION_LOG_ALL,
        status_report_interval: float = 900,
        minimum_spread: Decimal = Decimal(0),
        hb_app_notification: bool = False,
        order_override: Dict[str, List[str]] = {},
    ):
        if price_ceiling != s_decimal_neg_one and price_ceiling < price_floor:
            raise ValueError("Parameter price_ceiling cannot be lower than price_floor.")

        self._sb_order_tracker = PerpetualMarketMakingOrderTracker()
        self._market_info = market_info
        self._leverage = leverage
        self._position_mode = (
            PositionMode.HEDGE if position_mode == "Hedge" else PositionMode.ONEWAY
        )
        self._bid_spread = bid_spread
        self._ask_spread = ask_spread
        self._order_qty = order_qty  # Same as simulation order_qty (default = 1)
        self._grid_num = grid_num    # Same as simulation grid_num (default = 10)
        self._minimum_spread = minimum_spread
        # Extra parameters (not used in core logic) for compatibility.
        self._long_profit_taking_spread = long_profit_taking_spread
        self._short_profit_taking_spread = short_profit_taking_spread
        self._stop_loss_spread = stop_loss_spread
        self._order_level_spread = order_level_spread
        self._order_level_amount = order_level_amount
        self._order_refresh_time = order_refresh_time
        self._order_refresh_tolerance_pct = order_refresh_tolerance_pct
        self._filled_order_delay = filled_order_delay
        self._order_optimization_enabled = order_optimization_enabled
        self._ask_order_optimization_depth = ask_order_optimization_depth
        self._bid_order_optimization_depth = bid_order_optimization_depth
        self._asset_price_delegate = asset_price_delegate
        self._price_type = self.get_price_type(price_type)
        self._price_ceiling = price_ceiling
        self._price_floor = price_floor
        self._hb_app_notification = hb_app_notification
        self._order_override = order_override

        self._cancel_timestamp = 0
        self._create_timestamp = 0
        self._all_markets_ready = False
        self._logging_options = logging_options
        self._last_timestamp = 0
        self._status_report_interval = status_report_interval
        self._last_own_trade_price = Decimal("nan")
        self._ts_peak_bid_price = Decimal("0")
        self._ts_peak_ask_price = Decimal("0")
        self._exit_orders = dict()
        self._next_buy_exit_order_timestamp = 0
        self._next_sell_exit_order_timestamp = 0

        self.add_markets([market_info.market])

        self._close_order_type = OrderType.LIMIT
        self._time_between_stop_loss_orders = time_between_stop_loss_orders
        self._stop_loss_slippage_buffer = stop_loss_slippage_buffer

        self._position_mode_ready = False
        self._position_mode_not_ready_counter = 0

    def all_markets_ready(self):
        return all([market.ready for market in self.active_markets])

    @property
    def order_qty(self) -> Decimal:
        return self._order_qty

    @order_qty.setter
    def order_qty(self, value: Decimal):
        self._order_qty = value

    @property
    def grid_num(self) -> int:
        return self._grid_num

    @grid_num.setter
    def grid_num(self, value: int):
        self._grid_num = value

    @property
    def base_asset(self):
        return self._market_info.base_asset

    @property
    def quote_asset(self):
        return self._market_info.quote_asset

    @property
    def trading_pair(self):
        return self._market_info.trading_pair

    def get_price(self) -> float:
        if self._asset_price_delegate is not None:
            price_provider = self._asset_price_delegate
        else:
            price_provider = self._market_info
        if self._price_type is PriceType.LastOwnTrade:
            price = self._last_own_trade_price
        else:
            price = price_provider.get_price_by_type(self._price_type)
        if price.is_nan():
            price = price_provider.get_price_by_type(PriceType.MidPrice)
        return price

    @property
    def active_orders(self) -> List[LimitOrder]:
        if self._market_info not in self._sb_order_tracker.market_pair_to_active_orders:
            return []
        return self._sb_order_tracker.market_pair_to_active_orders[self._market_info]

    @property
    def active_positions(self) -> Dict[str, Position]:
        return self._market_info.market.account_positions

    def perpetual_mm_assets_df(self) -> pd.DataFrame:
        market, trading_pair, base_asset, quote_asset = self._market_info
        quote_balance = float(market.get_balance(quote_asset))
        available_quote_balance = float(market.get_available_balance(quote_asset))
        data = [
            ["", quote_asset],
            ["Total Balance", round(quote_balance, 4)],
            ["Available Balance", round(available_quote_balance, 4)]
        ]
        return pd.DataFrame(data=data)

    def active_orders_df(self) -> pd.DataFrame:
        price = self.get_price()
        active_orders = self.active_orders
        active_orders.sort(key=lambda x: x.price, reverse=True)
        columns = ["Level", "Type", "Price", "Spread", "Amount", "Age"]
        data = []
        lvl = 1
        for order in active_orders:
            spread = 0 if price == 0 else abs(order.price - price) / price
            age = pd.Timestamp(order_age(order, self.current_timestamp), unit='s').strftime('%H:%M:%S')
            data.append([
                lvl,
                "buy" if order.is_buy else "sell",
                float(order.price),
                f"{spread:.2%}",
                self._order_qty,
                age
            ])
            lvl += 1
        return pd.DataFrame(data=data, columns=columns)

    def market_status_data_frame(self) -> pd.DataFrame:
        markets_data = []
        markets_columns = ["Exchange", "Market", "Best Bid", "Best Ask", f"Ref Price ({self._price_type.name})"]
        market_books = [(self._market_info.market, self._market_info.trading_pair)]
        if isinstance(self._asset_price_delegate, OrderBookAssetPriceDelegate):
            market_books.append((self._asset_price_delegate.market, self._asset_price_delegate.trading_pair))
        for market, trading_pair in market_books:
            bid_price = market.get_price(trading_pair, False)
            ask_price = market.get_price(trading_pair, True)
            ref_price = float("nan")
            if market == self._market_info.market and self._asset_price_delegate is None:
                ref_price = self.get_price()
            elif market == self._asset_price_delegate.market and self._price_type is not PriceType.LastOwnTrade:
                ref_price = self._asset_price_delegate.get_price_by_type(self._price_type)
            markets_data.append([
                market.display_name,
                trading_pair,
                float(bid_price),
                float(ask_price),
                float(ref_price)
            ])
        return pd.DataFrame(data=markets_data, columns=markets_columns).replace(np.nan, '', regex=True)

    def format_status(self) -> str:
        if not self._all_markets_ready:
            return "Market connectors are not ready."
        lines = []
        markets_df = self.market_status_data_frame()
        lines.extend(["", "  Markets:"] + ["    " + line for line in markets_df.to_string(index=False).split("\n")])
        assets_df = self.perpetual_mm_assets_df().to_string(index=False).split("\n")
        lines.extend(["", "  Assets:"] + ["    " + line for line in assets_df])
        if len(self.active_orders) > 0:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  No active maker orders."])
        return "\n".join(lines)

    def start(self, clock: Clock, timestamp: float):
        self._market_info.market.set_leverage(self.trading_pair, self._leverage)

    def tick(self, timestamp: float):
        # Ensure position mode is ready.
        if not self._position_mode_ready:
            self._position_mode_not_ready_counter += 1
            if self._position_mode_not_ready_counter == 10:
                market: DerivativeBase = self._market_info.market
                if market.ready:
                    market.set_leverage(self.trading_pair, self._leverage)
                    market.set_position_mode(self._position_mode)
                self._position_mode_not_ready_counter = 0
            return
        self._position_mode_not_ready_counter = 0
        market: DerivativeBase = self._market_info.market
        session_positions = [s for s in self.active_positions.values() if s.trading_pair == self.trading_pair]
        # Operate only when no open position exists (mimicking simulation logic).
        if len(session_positions) == 0:
            self._exit_orders = dict()  # reset exit orders
            proposal = None
            if self._create_timestamp <= self.current_timestamp:
                proposal = self.create_base_proposal()
                self.logger().debug(f"Base proposal: {proposal}")
            self.cancel_active_orders(proposal)
            self.cancel_orders_below_min_spread()
            if self.to_create_orders(proposal):
                self.execute_orders_proposal(proposal, PositionAction.OPEN)
            # Update peak bid/ask prices.
            self._ts_peak_ask_price = market.get_price(self.trading_pair, False)
            self._ts_peak_bid_price = market.get_price(self.trading_pair, True)
        else:
            # When a position is open, do nothing.
            pass
        self._last_timestamp = timestamp

    def create_base_proposal(self) -> Proposal:
        """
        Creates a proposal for base orders using a grid strategy based on market microstructure
        signals (book pressure) and inventory skew. This exactly mimics the simulation logic.
        """
        market = self._market_info.market
        trading_pair = self.trading_pair

        # Get current price and use the order price quantum as tick size.
        current_price = self.get_price()
        tick_size = market.get_order_price_quantum(trading_pair, current_price)
        tick_size_f = float(tick_size)

        # Retrieve best bid/ask and quantities.
        if self._asset_price_delegate is not None and hasattr(self._asset_price_delegate, "order_book"):
            best_bid, best_bid_qty = self._asset_price_delegate.order_book.get_best_bid()
            best_ask, best_ask_qty = self._asset_price_delegate.order_book.get_best_ask()
            best_bid = float(best_bid)
            best_ask = float(best_ask)
            best_bid_qty = float(best_bid_qty)
            best_ask_qty = float(best_ask_qty)
        else:
            best_bid = float(market.get_price(trading_pair, False))
            best_ask = float(market.get_price(trading_pair, True))
            best_bid_qty = 1.0
            best_ask_qty = 1.0

        # Compute book pressure.
        book_pressure = (best_bid * best_bid_qty + best_ask * best_ask_qty) / (best_bid_qty + best_ask_qty)

        # Grid parameters (as in simulation).
        grid_num = self._grid_num
        order_qty_f = float(self._order_qty)
        max_position = grid_num * order_qty_f
        half_spread = tick_size_f * 0.49
        grid_interval = tick_size_f
        skew_adj = 1.0

        # Compute net position.
        net_position = sum(
            float(p.amount) for p in self.active_positions.values() if p.trading_pair == trading_pair
        )
        normalized_position = net_position / order_qty_f

        # Compute reservation price with inventory skew.
        skew = half_spread / grid_num * skew_adj
        reservation_price = book_pressure - skew * normalized_position

        # Set bid and ask prices (limited by best bid/ask).
        bid_price = min(reservation_price - half_spread, best_bid)
        ask_price = max(reservation_price + half_spread, best_ask)

        # Align prices to the grid.
        bid_price = floor(bid_price / grid_interval) * grid_interval
        ask_price = ceil(ask_price / grid_interval) * grid_interval

        # Build the grid proposals.
        buys = []
        sells = []

        if net_position < max_position and bid_price > 0:
            current_bid = bid_price
            for _ in range(grid_num):
                price = market.quantize_order_price(trading_pair, Decimal(str(current_bid)))
                size = self._order_qty
                buys.append(PriceSize(price, size))
                current_bid -= grid_interval

        if net_position > -max_position and ask_price > 0:
            current_ask = ask_price
            for _ in range(grid_num):
                price = market.quantize_order_price(trading_pair, Decimal(str(current_ask)))
                size = self._order_qty
                sells.append(PriceSize(price, size))
                current_ask += grid_interval

        return Proposal(buys, sells)

    def cancel_active_orders(self, proposal: Proposal):
        if self._cancel_timestamp > self.current_timestamp:
            return
        for order in self.active_orders:
            self.cancel_order(self._market_info, order.client_order_id)
            self.logger().info(f"Canceling active order {order.client_order_id}.")

    def cancel_orders_below_min_spread(self):
        price = self.get_price()
        for order in self.active_orders:
            negation = -1 if order.is_buy else 1
            if (negation * (order.price - price) / price) < self._minimum_spread:
                self.logger().info(
                    f"Order below minimum spread ({self._minimum_spread}). Canceling order {order.client_order_id}."
                )
                self.cancel_order(self._market_info, order.client_order_id)

    def to_create_orders(self, proposal: Proposal) -> bool:
        return (self._create_timestamp < self.current_timestamp and
                proposal is not None and
                len(self.active_orders) == 0)

    def execute_orders_proposal(self, proposal: Proposal, position_action: PositionAction):
        orders_created = False
        if len(proposal.buys) > 0:
            if position_action == PositionAction.CLOSE:
                if self.current_timestamp < self._next_buy_exit_order_timestamp:
                    return
                else:
                    self._next_buy_exit_order_timestamp = self.current_timestamp + self._filled_order_delay
            self.logger().info(
                f"({self.trading_pair}) Creating {len(proposal.buys)} bid orders for {position_action.name} position."
            )
            for buy in proposal.buys:
                bid_order_id = self.buy_with_specific_market(
                    self._market_info,
                    buy.size,
                    order_type=self._close_order_type,
                    price=buy.price,
                    position_action=position_action
                )
                if position_action == PositionAction.CLOSE:
                    self._exit_orders[bid_order_id] = self.current_timestamp
                orders_created = True
        if len(proposal.sells) > 0:
            if position_action == PositionAction.CLOSE:
                if self.current_timestamp < self._next_sell_exit_order_timestamp:
                    return
                else:
                    self._next_sell_exit_order_timestamp = self.current_timestamp + self._filled_order_delay
            self.logger().info(
                f"({self.trading_pair}) Creating {len(proposal.sells)} ask orders for {position_action.name} position."
            )
            for sell in proposal.sells:
                ask_order_id = self.sell_with_specific_market(
                    self._market_info,
                    sell.size,
                    order_type=self._close_order_type,
                    price=sell.price,
                    position_action=position_action
                )
                if position_action == PositionAction.CLOSE:
                    self._exit_orders[ask_order_id] = self.current_timestamp
                orders_created = True
        if orders_created:
            self.set_timers()

    def set_timers(self):
        next_cycle = self.current_timestamp + self._order_refresh_time
        if self._create_timestamp <= self.current_timestamp:
            self._create_timestamp = next_cycle
        if self._cancel_timestamp <= self.current_timestamp:
            self._cancel_timestamp = min(self._create_timestamp, next_cycle)

    def notify_hb_app(self, msg: str):
        if self._hb_app_notification:
            super().notify_hb_app(msg)

    def get_price_type(self, price_type_str: str) -> PriceType:
        if price_type_str == "mid_price":
            return PriceType.MidPrice
        elif price_type_str == "best_bid":
            return PriceType.BestBid
        elif price_type_str == "best_ask":
            return PriceType.BestAsk
        elif price_type_str == "last_price":
            return PriceType.LastTrade
        elif price_type_str == 'last_own_trade_price':
            return PriceType.LastOwnTrade
        elif price_type_str == "custom":
            return PriceType.Custom
        else:
            raise ValueError(f"Unrecognized price type string {price_type_str}.")
