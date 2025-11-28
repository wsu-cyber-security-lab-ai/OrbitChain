import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from .config import DEVICE

import numpy as np

# Optional Skyfield / requests imports (graceful fallback)
try:
    from skyfield.api import load, EarthSatellite, wgs84
    _HAS_SKYFIELD = True
    _TS = load.timescale()
except Exception:
    _HAS_SKYFIELD = False
    _TS = None

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------
# Helper: build time grid
# ---------------------------------------------------------------------

def _build_times(start_dt, end_dt, step_s: int):
    total_seconds = int((end_dt - start_dt).total_seconds())
    tpoints = [start_dt + timedelta(seconds=s) for s in range(0, total_seconds + 1, step_s)]
    years  = [t.year for t in tpoints]
    months = [t.month for t in tpoints]
    days   = [t.day for t in tpoints]
    hours  = [t.hour for t in tpoints]
    mins   = [t.minute for t in tpoints]
    secs   = [t.second for t in tpoints]
    return _TS.utc(years, months, days, hours, mins, secs)


# ---------------------------------------------------------------------
# TLE fetching / parsing
# ---------------------------------------------------------------------

def _fetch_tles(group: str) -> str:
    if not (_HAS_REQUESTS and group):
        return ""
    urls = [
        f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle",
        f"https://celestrak.org/NORAD/elements/supplemental/{group}.txt",
        f"https://celestrak.com/NORAD/elements/supplemental/{group}.txt",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text.strip():
                return r.text
        except Exception:
            pass
    return ""


def _parse_tles(tle_text: str, max_sats: int) -> List["EarthSatellite"]:
    if not tle_text or not _HAS_SKYFIELD:
        return []
    lines = [ln.rstrip() for ln in tle_text.strip().splitlines() if ln.strip()]
    sats: List[EarthSatellite] = []
    i = 0
    while i < len(lines) - 2 and len(sats) < max_sats:
        name = lines[i]
        l1, l2 = lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                sats.append(EarthSatellite(l1, l2, name=name, ts=_TS))
            except Exception:
                pass
        i += 3
    return sats


# ---------------------------------------------------------------------
# Visibility computation
# ---------------------------------------------------------------------

def _compute_visibility(sat, observer, times, elev_mask_deg: float, step_s: int):
    """
    Returns a list of passes: [(start_dt, end_dt), ...]
    """
    diff = sat - observer
    topo = diff.at(times)
    alt, az, dist = topo.altaz()
    altitudes = np.array(alt.degrees)
    visible = altitudes >= elev_mask_deg
    passes = []
    if visible.any():
        idx = np.where(visible)[0]
        groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for g in groups:
            t0 = times[g[0]].utc_datetime().replace(tzinfo=timezone.utc)
            t1 = times[g[-1]].utc_datetime().replace(tzinfo=timezone.utc)
            # include last step
            t1 = t1 + timedelta(seconds=step_s)
            passes.append((t0, t1))
    return passes


# ---------------------------------------------------------------------
# Synthetic visibility (fallback)
# ---------------------------------------------------------------------

def _build_synthetic_visible_clients(
    vendor_map: Dict[str, List[int]],
    num_rounds: int,
    p_visible: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    all_clients = [cid for cids in vendor_map.values() for cid in cids]
    active_per_round: List[List[int]] = []
    for r in range(num_rounds):
        active = []
        for cid in all_clients:
            if rng.rand() < p_visible:
                active.append(cid)
        if not active:
            # Ensure at least one client is active
            active.append(rng.choice(all_clients))
        active_per_round.append(sorted(active))
    return active_per_round


# ---------------------------------------------------------------------
# Public API: build per-satellite visibility schedule
# ---------------------------------------------------------------------

def build_visible_clients_schedule(
    vendor_map: Dict[str, List[int]],
    num_rounds: int,
    cycle_minutes: int,
    elev_mask_deg: float = 10.0,
    hours_ahead: int = 72,
    step_s: int = 60,
    base_coords: Dict[str, Tuple[float, float]] = None,
    hap_offsets = None,
    celestrak_groups: Dict[str, str] = None,
    seed: int = 42,
) -> List[List[int]]:
    """
    Returns: visible_clients_per_round: List of length num_rounds,
             each element is a sorted list of active client_ids.

    If Skyfield/requests/TLEs are unavailable, falls back to synthetic
    random visibility (per-satellite, per-round).
    """

    if (not _HAS_SKYFIELD) or (not _HAS_REQUESTS):
        print("[visibility] Skyfield/requests not available → using synthetic schedule.")
        return _build_synthetic_visible_clients(vendor_map, num_rounds, p_visible=0.5, seed=seed)

    if base_coords is None:
        base_coords = {
            "Starlink": (37.7749, -122.4194),  # SF
            "OneWeb":   (51.5074,  -0.1278),   # London
            "Kuiper":   (40.7128, -74.0060),   # NYC
        }
    if hap_offsets is None:
        hap_offsets = [(0.0, 0.0), (20.0, -30.0), (-25.0, 40.0)]
    if celestrak_groups is None:
        celestrak_groups = {
            "Starlink": "starlink",
            "OneWeb":   "oneweb",
            "Kuiper":   "amazon",   # Amazon Kuiper
        }

    # --- Build cycles (round → time window) ---
    now = datetime.now(timezone.utc)
    cycles = []
    cur = now
    for _ in range(num_rounds):
        end = cur + timedelta(minutes=cycle_minutes)
        cycles.append((cur, end))
        cur = end

    # --- Build HAPs per vendor ---
    vendor_haps: Dict[str, Dict[str, object]] = {}
    for vendor, (lat0, lon0) in base_coords.items():
        haps = {}
        for i, (dlat, dlon) in enumerate(hap_offsets, start=1):
            lat, lon = lat0 + dlat, lon0 + dlon
            haps[f"{vendor}_HAP{i}"] = wgs84.latlon(lat, lon, elevation_m=20000)
        vendor_haps[vendor] = haps

    # Global time grid for hours_ahead
    end_dt = now + timedelta(hours=hours_ahead)
    times = _build_times(now, end_dt, step_s)

    # Map each client to (vendor, sat_index) – order is as in vendor_map
    client_to_vendor_sat = {}  # cid -> (vendor, sat_idx)
    for v, cids in vendor_map.items():
        for s_idx, cid in enumerate(cids):
            client_to_vendor_sat[cid] = (v, s_idx)

    # For each vendor, fetch TLEs and compute pass windows
    rng = np.random.RandomState(seed)
    client_passes: Dict[int, List[Tuple[datetime, datetime]]] = {
        cid: [] for cid in client_to_vendor_sat
    }

    for vendor, cids in vendor_map.items():
        group = celestrak_groups.get(vendor, "")
        tle_text = _fetch_tles(group)
        sats = _parse_tles(tle_text, max_sats=len(cids))
        if not sats:
            print(f"[visibility] {vendor}: TLEs unavailable → synthetic for this vendor.")
            # Synthetic vendor-local: choose some cycles randomly
            for cid in cids:
                for (c0, c1) in cycles:
                    if rng.rand() < 0.7:
                        client_passes[cid].append((c0, c1))
            continue

        # There may be more sats than clients; just map 1:1 up to len(cids)
        sats = sats[:len(cids)]
        for cid, sat in zip(cids, sats):
            # Each sat sees all HAPs of vendor
            for hap_id, hap in vendor_haps[vendor].items():
                passes = _compute_visibility(sat, hap, times, elev_mask_deg, step_s)
                client_passes[cid].extend(passes)

    # For each round, mark clients visible if they have any pass overlapping the cycle
    visible_clients_per_round: List[List[int]] = []
    for r, (c0, c1) in enumerate(cycles):
        active = []
        for cid, pass_list in client_passes.items():
            for (p0, p1) in pass_list:
                if p1 > c0 and p0 < c1:  # overlap
                    active.append(cid)
                    break
        if not active:
            # ensure at least some coverage
            all_cids = list(client_passes.keys())
            active.append(all_cids[r % len(all_cids)])
        visible_clients_per_round.append(sorted(active))

    return visible_clients_per_round
