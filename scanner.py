#!/usr/bin/env python3
"""
Dome Trial — Sports Arbitrage Scanner
Finds >1% arb opportunities on Dome API matched sports events (Polymarket ↔ Kalshi).

Usage:
    python scanner.py                     # Scan all sports for today
    python scanner.py --sport nfl         # Scan specific sport
    python scanner.py --date 2026-03-01   # Scan specific date
    python scanner.py --threshold 2.5     # Custom arb threshold (%)
    python scanner.py --verbose           # Show all matched events, not just arbs
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from dome_api_sdk import DomeClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

load_dotenv()

console = Console()

SPORTS = ["nfl", "mlb", "nba", "nhl", "cfb", "cbb"]
SPORT_NAMES = {
    "nfl": "🏈 NFL",
    "mlb": "⚾ MLB",
    "nba": "🏀 NBA",
    "nhl": "🏒 NHL",
    "cfb": "🏈 CFB",
    "cbb": "🏀 CBB",
}

# Rate limiting: free tier = 1 QPS
RATE_LIMIT_DELAY = 1.1  # seconds between API calls

# ---- Fee constants ----
# Polymarket: most sports markets are FREE. Only NCAAB (cbb) and Serie A have fees.
# fee = C × feeRate × p × (1 - p)   where C = shares, p = price
POLY_FEE_RATE_SPORTS = 0.0175   # NCAAB / Serie A only
POLY_FEE_RATE_DEFAULT = 0.0     # most markets: no fee
POLY_FEE_SPORTS = {"cbb"}       # sports with Polymarket fees enabled

# Kalshi: fee = ceil(0.07 × C × p × (1 - p)) in cents, i.e. per-contract
# In dollar terms: fee_per_contract = ceil(7 × p × (1-p)) / 100
KALSHI_FEE_RATE = 0.07  # 7% of expected earnings


@dataclass
class PlatformPrices:
    """Actual Yes and No prices from a platform (not assumed)."""
    yes: float
    no: float
    platform: str


@dataclass
class OrderbookDepth:
    """Available volume at the best price level."""
    price: float
    size: float  # shares/contracts available at this price


@dataclass
class ArbSizing:
    """How much you can actually execute on an arb, including fees."""
    max_shares: float          # shares/contracts you can do (bottleneck of both legs)
    poly_shares: float         # shares available on Poly leg
    kalshi_contracts: float    # contracts available on Kalshi leg
    cost_per_share: float      # total cost per share across both legs (< 1.0 = arb)
    max_outlay: float          # total $ to place both legs (before fees)
    poly_fees: float           # Polymarket fees $
    kalshi_fees: float         # Kalshi fees $
    total_fees: float          # combined fees $
    gross_profit: float        # profit before fees $
    net_profit: float          # profit after fees $
    net_arb_pct: float         # net arb % after fees


def get_client() -> DomeClient:
    """Initialise Dome API client."""
    api_key = os.getenv("DOME_API_KEY")
    if not api_key:
        console.print("[red bold]ERROR:[/] DOME_API_KEY not set. Copy .env.example → .env and add your key.")
        console.print("Get a free key at: https://dashboard.domeapi.io/")
        sys.exit(1)
    return DomeClient({"api_key": api_key})


def rate_limit():
    """Respect free tier rate limit."""
    time.sleep(RATE_LIMIT_DELAY)


def fetch_matched_events(dome: DomeClient, sport: str, date: str) -> Dict:
    """Fetch matched events for a sport + date from Dome API."""
    try:
        result = dome.matching_markets.get_matching_markets_by_sport({
            "sport": sport,
            "date": date,
        })
        return result.markets
    except Exception as e:
        console.print(f"  [yellow]⚠ Error fetching {sport}: {e}[/]")
        return {}


def get_polymarket_prices(dome: DomeClient, token_ids: List[str]) -> Optional[PlatformPrices]:
    """
    Get actual Yes and No prices from Polymarket.
    token_ids[0] = Yes token, token_ids[1] = No token (if available).
    Each token is priced independently — do NOT assume No = 1 - Yes.
    """
    try:
        yes_result = dome.polymarket.markets.get_market_price({"token_id": token_ids[0]})
        yes_price = yes_result.price
        rate_limit()

        if len(token_ids) >= 2:
            no_result = dome.polymarket.markets.get_market_price({"token_id": token_ids[1]})
            no_price = no_result.price
        else:
            # Fallback only if no second token — flag it
            no_price = None

        return PlatformPrices(yes=yes_price, no=no_price, platform="POLYMARKET")
    except Exception as e:
        console.print(f"    [dim]⚠ Poly price error: {e}[/]")
        return None


def get_kalshi_prices(dome: DomeClient, market_ticker: str) -> Optional[PlatformPrices]:
    """
    Get actual Yes and No prices from Kalshi.
    Kalshi API returns both sides independently.
    """
    try:
        result = dome.kalshi.markets.get_market_price({"market_ticker": market_ticker})
        yes_price = result.yes.price
        no_price = result.no.price
        # Normalise cent-based (0-100) to decimal (0-1) if needed
        if yes_price > 1:
            yes_price = yes_price / 100.0
        if no_price > 1:
            no_price = no_price / 100.0
        return PlatformPrices(yes=yes_price, no=no_price, platform="KALSHI")
    except Exception as e:
        console.print(f"    [dim]⚠ Kalshi price error: {e}[/]")
        return None


def get_polymarket_best_offer(token_id: str, side: str) -> Optional[OrderbookDepth]:
    """
    Hit the Polymarket CLOB for the LIVE orderbook.
    side: 'buy' → best ask (cheapest you can buy at) + total size at that level
          'sell' → best bid (highest you can sell at) + total size at that level
    Returns the actual executable price and shares available there.
    """
    try:
        resp = httpx.get(
            f"https://clob.polymarket.com/book",
            params={"token_id": token_id},
            timeout=10,
        )
        resp.raise_for_status()
        book = resp.json()

        if side == "buy":
            levels = book.get("asks", [])
            if not levels:
                return None
            # Best ask = lowest price
            best_price = min(float(lvl["price"]) for lvl in levels)
            total = sum(
                float(lvl["size"]) for lvl in levels
                if abs(float(lvl["price"]) - best_price) < 0.0001
            )
        else:
            levels = book.get("bids", [])
            if not levels:
                return None
            # Best bid = highest price
            best_price = max(float(lvl["price"]) for lvl in levels)
            total = sum(
                float(lvl["size"]) for lvl in levels
                if abs(float(lvl["price"]) - best_price) < 0.0001
            )

        if total > 0:
            return OrderbookDepth(price=best_price, size=total)
        return None
    except Exception as e:
        console.print(f"    [dim]⚠ Poly live book error: {e}[/]")
        return None


def get_kalshi_best_offer(ticker: str, side: str) -> Optional[OrderbookDepth]:
    """
    Hit the Kalshi API for the LIVE orderbook.
    side: 'yes' or 'no' — which side we want to BUY.

    Kalshi book structure:
      yes = resting BUY orders for YES [price_cents, qty]
      no  = resting BUY orders for NO  [price_cents, qty]

    To BUY YES: take the highest NO bid → you pay (100 - no_bid) cents
    To BUY NO:  take the highest YES bid → you pay (100 - yes_bid) cents

    Volume at that price = qty at the highest opposing bid.
    """
    try:
        resp = httpx.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
            timeout=10,
        )
        resp.raise_for_status()
        book = resp.json().get("orderbook", {})

        if side == "yes":
            # To buy YES, take the best (highest) NO bid
            opposite_bids = book.get("no", [])
        else:
            # To buy NO, take the best (highest) YES bid
            opposite_bids = book.get("yes", [])

        if not opposite_bids:
            return None

        # Highest opposing bid = cheapest price to buy our side
        best_opposing_cents = max(lvl[0] for lvl in opposite_bids)
        total = sum(lvl[1] for lvl in opposite_bids if lvl[0] == best_opposing_cents)

        # Price we pay = 100 - opposing bid (in cents), convert to decimal
        buy_price = (100 - best_opposing_cents) / 100.0

        if total > 0:
            return OrderbookDepth(price=buy_price, size=total)
        return None
    except Exception as e:
        console.print(f"    [dim]⚠ Kalshi live book error: {e}[/]")
        return None


def calculate_arb(
    poly: PlatformPrices, kalshi: PlatformPrices
) -> Tuple[float, str, str]:
    """
    Calculate arbitrage using ACTUAL prices from both platforms.
    No assumptions — uses real Yes and No prices from each.

    Arb strategies:
    1. Buy Yes on Poly + Buy No on Kalshi → cost = poly.yes + kalshi.no, payout = 1.0
    2. Buy No on Poly + Buy Yes on Kalshi → cost = poly.no + kalshi.yes, payout = 1.0

    Returns: (arb_pct, strategy_description, strategy_key)
    """
    results = []

    # Strategy 1: Yes@Poly + No@Kalshi
    if poly.yes is not None and kalshi.no is not None:
        cost = poly.yes + kalshi.no
        arb = ((1.0 - cost) / cost * 100) if cost > 0 else -999
        results.append((
            arb,
            f"YES @Poly {poly.yes:.3f} + NO @Kalshi {kalshi.no:.3f} = cost {cost:.4f}",
            "poly_yes_kalshi_no",
        ))

    # Strategy 2: No@Poly + Yes@Kalshi
    if poly.no is not None and kalshi.yes is not None:
        cost = poly.no + kalshi.yes
        arb = ((1.0 - cost) / cost * 100) if cost > 0 else -999
        results.append((
            arb,
            f"NO @Poly {poly.no:.3f} + YES @Kalshi {kalshi.yes:.3f} = cost {cost:.4f}",
            "poly_no_kalshi_yes",
        ))

    if not results:
        return (-999, "Insufficient price data", "none")

    # Return best arb
    best = max(results, key=lambda x: x[0])
    return best


def calc_poly_fee(shares: float, price: float, sport: str) -> float:
    """
    Polymarket taker fee.
    Most sports = FREE. NCAAB (cbb) and Serie A have fees.
    fee = shares × feeRate × price × (1 - price)
    """
    if sport in POLY_FEE_SPORTS:
        rate = POLY_FEE_RATE_SPORTS
    else:
        rate = POLY_FEE_RATE_DEFAULT
    return shares * rate * price * (1.0 - price)


def calc_kalshi_fee(contracts: float, price: float) -> float:
    """
    Kalshi taker fee per contract, then × contracts.
    fee_per_contract = ceil(7 × p × (1-p)) cents → dollars
    Total = contracts × fee_per_contract
    """
    import math
    fee_cents_per = math.ceil(7.0 * price * (1.0 - price))
    return contracts * fee_cents_per / 100.0


def get_arb_sizing(
    strategy_key: str,
    poly_token_ids: List[str],
    kalshi_ticker: str,
    sport: str,
) -> Optional[ArbSizing]:
    """
    Get full arb sizing from LIVE orderbooks — actual executable prices, not last-trade.
    Hits Polymarket CLOB + Kalshi API for the real best ask on each leg.

    Both platforms pay $1 per share/contract on win, so:
    - cost_per_share = best_ask_leg1 + best_ask_leg2
    - max_shares = min(poly_available, kalshi_available)
    - max_outlay = max_shares × cost_per_share
    - max_profit = max_shares × (1.0 - cost_per_share)
    """
    poly_depth = None
    kalshi_depth = None

    if strategy_key == "poly_yes_kalshi_no":
        # Leg 1: Buy Yes on Poly — get best ask
        poly_depth = get_polymarket_best_offer(poly_token_ids[0], "buy")
        # Leg 2: Buy No on Kalshi — get best ask
        kalshi_depth = get_kalshi_best_offer(kalshi_ticker, "no")

    elif strategy_key == "poly_no_kalshi_yes":
        # Leg 1: Buy No on Poly — get best ask for No token
        if len(poly_token_ids) >= 2:
            poly_depth = get_polymarket_best_offer(poly_token_ids[1], "buy")
        # Leg 2: Buy Yes on Kalshi — get best ask
        kalshi_depth = get_kalshi_best_offer(kalshi_ticker, "yes")

    if not poly_depth or not kalshi_depth:
        return None

    cost_per_share = poly_depth.price + kalshi_depth.price
    if cost_per_share >= 1.0:
        return None  # no arb at actual executable prices

    # Bottleneck = min of both legs
    max_shares = min(poly_depth.size, kalshi_depth.size)
    max_outlay = max_shares * cost_per_share
    gross_profit = max_shares * (1.0 - cost_per_share)

    # Fees
    poly_fees = calc_poly_fee(max_shares, poly_depth.price, sport)
    kalshi_fees = calc_kalshi_fee(max_shares, kalshi_depth.price)
    total_fees = poly_fees + kalshi_fees

    net_profit = gross_profit - total_fees
    net_arb_pct = (net_profit / max_outlay * 100) if max_outlay > 0 else 0

    return ArbSizing(
        max_shares=max_shares,
        poly_shares=poly_depth.size,
        kalshi_contracts=kalshi_depth.size,
        cost_per_share=cost_per_share,
        max_outlay=max_outlay,
        poly_fees=poly_fees,
        kalshi_fees=kalshi_fees,
        total_fees=total_fees,
        gross_profit=gross_profit,
        net_profit=net_profit,
        net_arb_pct=net_arb_pct,
    )


def scan_sport(dome: DomeClient, sport: str, date: str, threshold: float, verbose: bool) -> List[dict]:
    """Scan a single sport for arb opportunities."""
    console.print(f"\n{SPORT_NAMES.get(sport, sport)} — {date}")

    markets = fetch_matched_events(dome, sport, date)
    rate_limit()

    if not markets:
        console.print("  [dim]No matched events found[/]")
        return []

    console.print(f"  Found [cyan]{len(markets)}[/] matched event(s)")

    arbs = []

    for event_key, platforms in markets.items():
        poly_data = None
        kalshi_data = None

        for platform in platforms:
            if platform.platform == "POLYMARKET":
                poly_data = platform
            elif platform.platform == "KALSHI":
                kalshi_data = platform

        if not poly_data or not kalshi_data:
            if verbose:
                console.print(f"  [dim]  {event_key}: Missing platform data, skipping[/]")
            continue

        if not poly_data.token_ids or len(poly_data.token_ids) < 1:
            if verbose:
                console.print(f"  [dim]  {event_key}: No Poly token IDs[/]")
            continue

        # Get ACTUAL prices from both platforms — no assumptions
        poly_prices = get_polymarket_prices(dome, poly_data.token_ids)
        rate_limit()

        if poly_prices is None:
            continue

        for kalshi_ticker in kalshi_data.market_tickers:
            kalshi_prices = get_kalshi_prices(dome, kalshi_ticker)
            rate_limit()

            if kalshi_prices is None:
                continue

            arb_pct, strategy, strategy_key = calculate_arb(poly_prices, kalshi_prices)

            result = {
                "event": event_key,
                "sport": sport,
                "poly_slug": poly_data.market_slug,
                "kalshi_ticker": kalshi_ticker,
                "poly_yes": poly_prices.yes,
                "poly_no": poly_prices.no,
                "kalshi_yes": kalshi_prices.yes,
                "kalshi_no": kalshi_prices.no,
                "arb_pct": arb_pct,
                "strategy": strategy,
                "strategy_key": strategy_key,
                "sizing": None,
            }

            if arb_pct > threshold:
                console.print(f"  [green bold]✅ ARB {arb_pct:+.2f}%[/] — {event_key}")
                console.print(f"     {strategy}")
                poly_no_display = f"{poly_prices.no:.3f}" if poly_prices.no is not None else "n/a"
                console.print(f"     Poly  YES={poly_prices.yes:.3f}  NO={poly_no_display}")
                console.print(f"     Kalshi YES={kalshi_prices.yes:.3f}  NO={kalshi_prices.no:.3f}")

                sizing = get_arb_sizing(
                    strategy_key, poly_data.token_ids, kalshi_ticker, sport,
                )
                result["sizing"] = sizing

                if sizing:
                    console.print(f"     [cyan]Book depth — Poly: {sizing.poly_shares:,.1f} shares | Kalshi: {sizing.kalshi_contracts:,.0f} contracts[/]")
                    console.print(f"     [cyan]Max executable: {sizing.max_shares:,.1f} shares @ ${sizing.cost_per_share:.4f}/share[/]")
                    console.print(f"     Fees — Poly: ${sizing.poly_fees:,.2f} | Kalshi: ${sizing.kalshi_fees:,.2f} | Total: ${sizing.total_fees:,.2f}")
                    if sizing.net_profit > 0:
                        console.print(f"     [green bold]💰 Outlay: ${sizing.max_outlay:,.2f} → Net profit: ${sizing.net_profit:,.2f} ({sizing.net_arb_pct:+.2f}% after fees)[/]")
                    else:
                        console.print(f"     [red]❌ Fees eat the arb: ${sizing.net_profit:,.2f} net ({sizing.net_arb_pct:+.2f}% after fees)[/]")
                else:
                    console.print(f"     [dim]No liquidity on book at these prices[/]")

                arbs.append(result)
            elif verbose:
                colour = "yellow" if arb_pct > 0 else "dim"
                console.print(
                    f"  [{colour}]  {arb_pct:+.2f}% — {event_key} | "
                    f"P: Y={poly_prices.yes:.3f} N={poly_prices.no} | "
                    f"K: Y={kalshi_prices.yes:.3f} N={kalshi_prices.no:.3f}"
                    f"[/{colour}]"
                )

    return arbs


def main():
    parser = argparse.ArgumentParser(description="Dome Trial — Sports Arb Scanner")
    parser.add_argument("--sport", choices=SPORTS, help="Scan a single sport (default: all)")
    parser.add_argument("--date", help="Date to scan (YYYY-MM-DD, default: today UTC)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Min arb %% to flag (default: 1.0)")
    parser.add_argument("--verbose", action="store_true", help="Show all matched events, not just arbs")
    args = parser.parse_args()

    scan_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sports_to_scan = [args.sport] if args.sport else SPORTS

    console.print(Panel.fit(
        f"[bold]Dome Trial — Sports Arb Scanner[/]\n"
        f"Date: {scan_date} | Threshold: >{args.threshold}% | Sports: {', '.join(sports_to_scan)}",
        border_style="cyan",
    ))

    dome = get_client()
    all_arbs = []

    for sport in sports_to_scan:
        arbs = scan_sport(dome, sport, scan_date, args.threshold, args.verbose)
        all_arbs.extend(arbs)

    # Summary
    console.print("\n")
    if all_arbs:
        table = Table(title=f"🚨 Arb Opportunities > {args.threshold}%", box=box.ROUNDED)
        table.add_column("Sport", style="cyan")
        table.add_column("Event", style="white", max_width=25)
        table.add_column("Gross %", style="yellow", justify="right")
        table.add_column("P.Yes", justify="right")
        table.add_column("P.No", justify="right")
        table.add_column("K.Yes", justify="right")
        table.add_column("K.No", justify="right")
        table.add_column("Shares", justify="right", style="cyan")
        table.add_column("Outlay", justify="right")
        table.add_column("Fees", justify="right", style="red")
        table.add_column("Net $", justify="right", style="green bold")
        table.add_column("Net %", justify="right", style="green bold")

        all_arbs.sort(key=lambda x: (x["sizing"].net_profit if x["sizing"] else -9999), reverse=True)

        for arb in all_arbs:
            poly_no_str = f"{arb['poly_no']:.3f}" if arb['poly_no'] is not None else "n/a"
            s = arb["sizing"]
            if s:
                shares_str = f"{s.max_shares:,.1f}"
                outlay_str = f"${s.max_outlay:,.2f}"
                fees_str = f"${s.total_fees:,.2f}"
                net_str = f"${s.net_profit:,.2f}"
                net_pct_str = f"{s.net_arb_pct:+.2f}%"
                net_style = "green bold" if s.net_profit > 0 else "red"
            else:
                shares_str = "—"
                outlay_str = "—"
                fees_str = "—"
                net_str = "—"
                net_pct_str = "—"

            table.add_row(
                SPORT_NAMES.get(arb["sport"], arb["sport"]),
                arb["event"],
                f"{arb['arb_pct']:+.2f}%",
                f"{arb['poly_yes']:.3f}",
                poly_no_str,
                f"{arb['kalshi_yes']:.3f}",
                f"{arb['kalshi_no']:.3f}",
                shares_str,
                outlay_str,
                fees_str,
                net_str,
                net_pct_str,
            )

        console.print(table)
        console.print(f"\n[green bold]Found {len(all_arbs)} arb(s) above {args.threshold}%[/]")
    else:
        console.print(Panel(
            f"[yellow]No arb opportunities found above {args.threshold}% threshold.[/]\n"
            "Try lowering --threshold or scanning a different date.",
            title="Results",
            border_style="yellow",
        ))


if __name__ == "__main__":
    main()
