#!/usr/bin/env python3
"""Find and optionally merge duplicate restaurant rows in a cluster cache.

Heuristic: same normalized name + geographic proximity (haversine distance).

Usage:
  python scripts/find_and_merge_duplicates.py --input data/cluster_cache.parquet --preview
  python scripts/find_and_merge_duplicates.py --input data/cluster_cache.parquet --output data/cluster_cache_dedup.parquet

This is conservative: it only drops rows that appear to be the same physical
location (same name and within `--threshold-m` meters). The keeper is chosen
by highest `review_count` then highest `avg_rating`.
"""
from __future__ import annotations

import argparse
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
import pandas as pd


def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = str(name).lower()
    # keep alphanum and spaces
    import re
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = " ".join(s.split())
    return s


def haversine_meters(lat1, lon1, lat2, lon2):
    # approximate haversine
    R = 6371000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def find_duplicate_groups(df: pd.DataFrame, threshold_m=150.0) -> list[list[int]]:
    """Return list of index groups (list of row indices) that are duplicates."""
    df = df.reset_index(drop=True)
    df['_norm_name'] = df['name'].fillna("").map(normalize_name)

    groups = []
    visited = set()

    for name, sub in df.groupby('_norm_name'):
        if not name:
            continue
        if len(sub) <= 1:
            continue
        idxs = list(sub.index)
        # pairwise clustering by proximity (naive O(n^2) within each name)
        parent = list(range(len(idxs)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                r1 = sub.loc[idxs[i]]
                r2 = sub.loc[idxs[j]]
                try:
                    lat1, lon1 = float(r1.get('lat') or r1.get('latitude') or r1.get('lat_norm') or 0), float(r1.get('lng') or r1.get('longitude') or r1.get('lng_norm') or 0)
                    lat2, lon2 = float(r2.get('lat') or r2.get('latitude') or r2.get('lat_norm') or 0), float(r2.get('lng') or r2.get('longitude') or r2.get('lng_norm') or 0)
                except Exception:
                    continue
                if lat1 == 0 and lat2 == 0 and lon1 == 0 and lon2 == 0:
                    continue
                dist = haversine_meters(lat1, lon1, lat2, lon2)
                if dist <= threshold_m:
                    union(i, j)

        clusters = {}
        for i in range(len(idxs)):
            root = find(i)
            clusters.setdefault(root, []).append(idxs[i])

        for c in clusters.values():
            if len(c) > 1:
                groups.append(c)

    return groups


def deduplicate(df: pd.DataFrame, groups: list[list[int]]) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    to_drop = set()
    for group in groups:
        # choose keeper by review_count then avg_rating
        cand = df.loc[group].copy()
        cand['review_count'] = pd.to_numeric(cand.get('review_count') or cand.get('g_reviews') or 0, errors='coerce').fillna(0).astype(int)
        cand['avg_rating'] = pd.to_numeric(cand.get('avg_rating') or cand.get('g_rating') or 0, errors='coerce').fillna(0.0)
        keeper_idx = cand.sort_values(['review_count', 'avg_rating'], ascending=[False, False]).index[0]
        for idx in group:
            if idx != keeper_idx:
                to_drop.add(idx)

    deduped = df.drop(index=list(to_drop)).reset_index(drop=True)
    return deduped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=False)
    parser.add_argument('--threshold-m', type=float, default=150.0, help='distance threshold in meters')
    parser.add_argument('--preview', action='store_true', help='only show duplicate groups, do not write output')
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input {path} not found")
        return

    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")
    groups = find_duplicate_groups(df, threshold_m=args.threshold_m)
    print(f"Found {len(groups)} duplicate groups (threshold={args.threshold_m} m)")
    for i, g in enumerate(groups, start=1):
        sub = df.reset_index(drop=True).loc[g]
        names = sub['name'].fillna('').tolist()
        locs = sub[['lat','lng']].fillna('').astype(str).apply(lambda x: ','.join(x), axis=1).tolist()
        if 'review_count' in sub.columns:
            rc_series = sub['review_count'].fillna('')
        elif 'g_reviews' in sub.columns:
            rc_series = sub['g_reviews'].fillna('')
        else:
            rc_series = pd.Series([''] * len(sub))

        print(f"Group {i}: indices={g} rows={len(g)}")
        for local_idx, row in sub.reset_index().iterrows():
            idx_display = row['index'] if 'index' in row else local_idx
            name_display = row.get('name') or row.get('dba')
            lat_val = row.get('lat') if 'lat' in row else row.get('latitude')
            lng_val = row.get('lng') if 'lng' in row else row.get('longitude')
            reviews = ''
            if 'review_count' in sub.columns:
                reviews = row.get('review_count')
            elif 'g_reviews' in sub.columns:
                reviews = row.get('g_reviews')
            print(f"  - idx={idx_display} name={name_display} loc=({lat_val},{lng_val}) reviews={reviews}")
        print()

    if args.preview or not args.output:
        print("Preview mode — no output written.")
        return

    deduped = deduplicate(df, groups)
    out = Path(args.output)
    deduped.to_parquet(out)
    print(f"Wrote deduplicated parquet to {out} ({len(deduped):,} rows)")


if __name__ == '__main__':
    main()
