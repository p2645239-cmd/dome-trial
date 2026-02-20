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


def get_client() -> DomeClient:
    """Initialise Dome API client."""
    api_key = os.getenv("DOME_API_KEY")
    if not api_key:
        console.print("[red bold]ERROR:[/] DOME_API_KEY not set. Copy .env.example → .env and add your key.")
        console.print("Get a free key at: https://dashboard.domeapi.io/")
        sys.exit(1)
    return DomeClient({"api_key": api_key})


def fetch_matched_events(dome: DomeClient, sport: str, date: str) -> Dict:
    """Fetch matched events for a sport + date from Dome API."""
    try:
        result = dome.matching_markets.get_matching_markets_by_sport({
            "sport": sport,
            "date": date,
        })
        return result.markets  # Dict[str, List[MarketData]]
    except Exception as e:
        console.print(f"  [yellow]⚠ Error fetching {sport}: {e}[/]")
        return {}


def get_polymarket_price(dome: DomeClient, token_id: str) -> Optional[float]:
    """Get current Polymarket price for a token (Yes side, 0-1)."""
    try:
        result = dome.polymarket.markets.get_market_price({"token_id": token_id})
        return result.price
    except Exception as e:
        console.print(f"    [dim]⚠ Poly price error: {e}[/]")
        return None


def get_kalshi_price(dome: DomeClient, market_ticker: str) -> Optional[Tuple[float, float]]:
    """Get current Kalshi price (yes, no) normalised to 0-1."""
    try:
        result = dome.kalshi.markets.get_market_price({"market_ticker": market_ticker})
        # Kalshi prices are in cents (0-100), normalise to 0-1
        yes_price = result.yes.price
        no_price = result.no.price
        # Handle both cent-based (0-100) and decimal (0-1) formats
        if yes_price > 1:
            yes_price = yes_price / 100.0
            no_price = no_price / 100.0
        return (yes_price, no_price)
    except Exception as e:
        console.print(f"    [dim]⚠ Kalshi price error: {e}[/]")
        return None


def calculate_arb(poly_yes: float, kalshi_yes: float, kalshi_no: float) -> Tuple[float, str]:
    """
    Calculate arbitrage opportunity between Polymarket and Kalshi.
    
    Arb strategies:
    1. Buy Yes on Poly + Buy No on Kalshi → cost = poly_yes + kalshi_no, payout = 1.0
    2. Buy No on Poly (= 1 - poly_yes) + Buy Yes on Kalshi → cost = (1 - poly_yes) + kalshi_yes, payout = 1.0
    
    Arb % = (1 - cost) / cost * 100  (guaranteed profit as % of outlay)
    
    Returns: (arb_pct, strategy_description)
    """
    poly_no = 1.0 - poly_yes

    # Strategy 1: Yes on Poly + No on Kalshi
    cost_1 = poly_yes + kalshi_no
    arb_1 = ((1.0 - cost_1) / cost_1 * 100) if cost_1 > 0 else -999

    # Strategy 2: No on Poly + Yes on Kalshi
    cost_2 = poly_no + kalshi_yes
    arb_2 = ((1.0 - cost_2) / cost_2 * 100) if cost_2 > 0 else -999

    if arb_1 >= arb_2:
        return (arb_1, f"Buy YES @Poly {poly_yes:.2f} + NO @Kalshi {kalshi_no:.2f} = cost {cost_1:.4f}")
    else:
        return (arb_2, f"Buy NO @Poly {poly_no:.2f} + YES @Kalshi {kalshi_yes:.2f} = cost {cost_2:.4f}")


def scan_sport(dome: DomeClient, sport: str, date: str, threshold: float, verbose: bool) -> List[dict]:
    """Scan a single sport for arb opportunities."""
    console.print(f"\n{SPORT_NAMES.get(sport, sport)} — {date}")
    
    markets = fetch_matched_events(dome, sport, date)
    time.sleep(RATE_LIMIT_DELAY)

    if not markets:
        console.print("  [dim]No matched events found[/]")
        return []

    console.print(f"  Found [cyan]{len(markets)}[/] matched event(s)")
    
    arbs = []

    for event_key, platforms in markets.items():
        # Extract Polymarket and Kalshi data from the matched pair
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

        # For each Kalshi market ticker, try to find an arb against Polymarket
        # Polymarket has token_ids (usually 2: Yes and No tokens)
        # We need the Yes token price
        if not poly_data.token_ids or len(poly_data.token_ids) < 1:
            if verbose:
                console.print(f"  [dim]  {event_key}: No Poly token IDs[/]")
            continue

        # Get Polymarket Yes price (first token_id is typically the Yes side)
        poly_yes = get_polymarket_price(dome, poly_data.token_ids[0])
        time.sleep(RATE_LIMIT_DELAY)
        
        if poly_yes is None:
            continue

        # Check each Kalshi market ticker
        for kalshi_ticker in kalshi_data.market_tickers:
            kalshi_prices = get_kalshi_price(dome, kalshi_ticker)
            time.sleep(RATE_LIMIT_DELAY)
            
            if kalshi_prices is None:
                continue
            
            kalshi_yes, kalshi_no = kalshi_prices
            arb_pct, strategy = calculate_arb(poly_yes, kalshi_yes, kalshi_no)

            result = {
                "event": event_key,
                "sport": sport,
                "poly_slug": poly_data.market_slug,
                "kalshi_ticker": kalshi_ticker,
                "poly_yes": poly_yes,
                "kalshi_yes": kalshi_yes,
                "kalshi_no": kalshi_no,
                "arb_pct": arb_pct,
                "strategy": strategy,
            }

            if arb_pct > threshold:
                arbs.append(result)
                console.print(f"  [green bold]✅ ARB {arb_pct:+.2f}%[/] — {event_key}")
                console.print(f"     {strategy}")
                console.print(f"     Poly: {poly_data.market_slug} | Kalshi: {kalshi_ticker}")
            elif verbose:
                colour = "yellow" if arb_pct > 0 else "dim"
                console.print(f"  [{colour}]  {arb_pct:+.2f}% — {event_key}[/{colour}]")

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
        table.add_column("Event", style="white")
        table.add_column("Arb %", style="green bold", justify="right")
        table.add_column("Poly Yes", justify="right")
        table.add_column("Kalshi Yes", justify="right")
        table.add_column("Strategy")

        # Sort by arb % descending
        all_arbs.sort(key=lambda x: x["arb_pct"], reverse=True)

        for arb in all_arbs:
            table.add_row(
                SPORT_NAMES.get(arb["sport"], arb["sport"]),
                arb["event"],
                f"{arb['arb_pct']:+.2f}%",
                f"{arb['poly_yes']:.2f}",
                f"{arb['kalshi_yes']:.2f}",
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
