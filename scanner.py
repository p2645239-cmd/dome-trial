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
    size: float  # shares available


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


def get_polymarket_orderbook_depth(dome: DomeClient, token_id: str, side: str) -> Optional[OrderbookDepth]:
    """
    Get the best price + available volume from the Polymarket orderbook.
    side: 'buy' (look at asks — cheapest offer) or 'sell' (look at bids — best bid).
    
    Returns the top-of-book level with total size available at that price.
    """
    try:
        now_ms = int(time.time() * 1000)
        result = dome.polymarket.markets.get_orderbooks({
            "token_id": token_id,
            "start_time": now_ms - 60_000,  # last 60 seconds
            "end_time": now_ms,
            "limit": 1,  # latest snapshot only
        })
        if not result.snapshots:
            return None

        snapshot = result.snapshots[-1]  # most recent
        # asks/bids are lists of dicts: {"price": "0.65", "size": "1500"}
        book = snapshot.asks if side == "buy" else snapshot.bids

        if not book:
            return None

        # Best ask = lowest price; Best bid = highest price
        if side == "buy":
            best = min(book, key=lambda x: float(x["price"]))
        else:
            best = max(book, key=lambda x: float(x["price"]))

        # Sum all volume at the best price level
        best_price = float(best["price"])
        total_size = sum(
            float(level["size"]) for level in book
            if float(level["price"]) == best_price
        )

        return OrderbookDepth(price=best_price, size=total_size)
    except Exception as e:
        console.print(f"    [dim]⚠ Poly orderbook error: {e}[/]")
        return None


def get_kalshi_orderbook_depth(dome: DomeClient, ticker: str, side: str) -> Optional[OrderbookDepth]:
    """
    Get the best price + available volume from the Kalshi orderbook.
    side: 'yes' or 'no' — which side we want to buy.
    """
    try:
        now_ms = int(time.time() * 1000)
        result = dome.kalshi.orderbooks.get_orderbooks({
            "ticker": ticker,
            "start_time": now_ms - 60_000,
            "end_time": now_ms,
            "limit": 1,
        })
        if not result.snapshots:
            return None

        snapshot = result.snapshots[-1]
        # Kalshi orderbook: yes/no arrays of [price_cents, count]
        book = snapshot.orderbook.yes if side == "yes" else snapshot.orderbook.no

        if not book:
            return None

        # Best offer = lowest price available to buy
        best = min(book, key=lambda x: x[0])
        best_price = best[0]
        # Normalise cents to decimal
        if best_price > 1:
            best_price = best_price / 100.0

        total_contracts = sum(level[1] for level in book if level[0] == best[0])

        return OrderbookDepth(price=best_price, size=total_contracts)
    except Exception as e:
        console.print(f"    [dim]⚠ Kalshi orderbook error: {e}[/]")
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


def get_max_arb_volume(
    dome: DomeClient,
    strategy_key: str,
    poly_token_ids: List[str],
    kalshi_ticker: str,
) -> Optional[float]:
    """
    Get the max volume (in shares/contracts) available at the arb price.
    The bottleneck is the smaller side of the two legs.
    """
    try:
        if strategy_key == "poly_yes_kalshi_no":
            # Leg 1: Buy Yes on Poly (check asks for Yes token)
            poly_depth = get_polymarket_orderbook_depth(dome, poly_token_ids[0], "buy")
            rate_limit()
            # Leg 2: Buy No on Kalshi
            kalshi_depth = get_kalshi_orderbook_depth(dome, kalshi_ticker, "no")
            rate_limit()
        elif strategy_key == "poly_no_kalshi_yes":
            # Leg 1: Buy No on Poly (check asks for No token)
            if len(poly_token_ids) >= 2:
                poly_depth = get_polymarket_orderbook_depth(dome, poly_token_ids[1], "buy")
            else:
                poly_depth = None
            rate_limit()
            # Leg 2: Buy Yes on Kalshi
            kalshi_depth = get_kalshi_orderbook_depth(dome, kalshi_ticker, "yes")
            rate_limit()
        else:
            return None

        sizes = []
        if poly_depth and poly_depth.size > 0:
            sizes.append(poly_depth.size)
        if kalshi_depth and kalshi_depth.size > 0:
            sizes.append(kalshi_depth.size)

        if sizes:
            return min(sizes)  # bottleneck = smaller side
        return None
    except Exception as e:
        console.print(f"    [dim]⚠ Volume check error: {e}[/]")
        return None


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
                "max_volume": None,
            }

            if arb_pct > threshold:
                # Fetch orderbook depth for the winning strategy
                console.print(f"  [green bold]✅ ARB {arb_pct:+.2f}%[/] — {event_key}")
                console.print(f"     {strategy}")
                console.print(f"     Poly YES={poly_prices.yes:.3f}  NO={poly_prices.no}")
                console.print(f"     Kalshi YES={kalshi_prices.yes:.3f}  NO={kalshi_prices.no:.3f}")

                max_vol = get_max_arb_volume(dome, strategy_key, poly_data.token_ids, kalshi_ticker)
                result["max_volume"] = max_vol

                if max_vol is not None:
                    console.print(f"     [cyan]Max volume at arb price: {max_vol:,.0f} shares/contracts[/]")
                else:
                    console.print(f"     [dim]Max volume: unavailable (no orderbook data)[/]")

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
        table.add_column("Event", style="white", max_width=30)
        table.add_column("Arb %", style="green bold", justify="right")
        table.add_column("Poly YES", justify="right")
        table.add_column("Poly NO", justify="right")
        table.add_column("Kalshi YES", justify="right")
        table.add_column("Kalshi NO", justify="right")
        table.add_column("Max Vol", justify="right", style="cyan")
        table.add_column("Strategy", max_width=45)

        all_arbs.sort(key=lambda x: x["arb_pct"], reverse=True)

        for arb in all_arbs:
            poly_no_str = f"{arb['poly_no']:.3f}" if arb['poly_no'] is not None else "n/a"
            vol_str = f"{arb['max_volume']:,.0f}" if arb['max_volume'] is not None else "n/a"

            table.add_row(
                SPORT_NAMES.get(arb["sport"], arb["sport"]),
                arb["event"],
                f"{arb['arb_pct']:+.2f}%",
                f"{arb['poly_yes']:.3f}",
                poly_no_str,
                f"{arb['kalshi_yes']:.3f}",
                f"{arb['kalshi_no']:.3f}",
                vol_str,
                arb["strategy"],
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
