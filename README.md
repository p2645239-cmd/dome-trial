# Dome Trial — Sports Arb Scanner

Finds arbitrage opportunities >1% on **matched sports events** between Polymarket and Kalshi, powered by [Dome API](https://domeapi.io)'s cross-platform market matching.

## What It Does

1. Fetches matched Polymarket ↔ Kalshi sports events from Dome API
2. Gets live prices from both platforms
3. Calculates arbitrage % for each matched pair
4. Flags any opportunity above the threshold (default: 1%)

**Sports supported:** NFL, MLB, NBA, NHL, CFB, CBB

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Get a free API key from https://dashboard.domeapi.io/
cp .env.example .env
# Edit .env and add your DOME_API_KEY
```

## Usage

```bash
# Scan all sports for today
python scanner.py

# Scan specific sport
python scanner.py --sport nba

# Scan specific date
python scanner.py --date 2026-03-01

# Lower threshold to 0.5%
python scanner.py --threshold 0.5

# Verbose mode — show all matched events, not just arbs
python scanner.py --verbose

# Combine
python scanner.py --sport nfl --date 2026-09-07 --threshold 2.0 --verbose
```

## How Arb Calculation Works

For a binary Yes/No market that exists on both platforms:

| Strategy | Cost | Payout |
|----------|------|--------|
| Buy YES on Poly + Buy NO on Kalshi | `poly_yes + kalshi_no` | 1.00 |
| Buy NO on Poly + Buy YES on Kalshi | `(1 - poly_yes) + kalshi_yes` | 1.00 |

**Arb % = (1 - cost) / cost × 100**

If cost < 1.00, you have a guaranteed profit regardless of outcome.

## Rate Limits

Free tier: 1 query/second. The scanner respects this with built-in delays.
For faster scanning, upgrade at https://dashboard.domeapi.io/.

## Notes

- Dome matching is **sports only** (6 US sports). Non-sports arbs need a different approach.
- Token ID ordering: assumes first Polymarket token_id = Yes side (standard convention).
- Prices are point-in-time snapshots — execution slippage may eat small arbs.
