#
"""
Purpose
-------
Clean and prepare the NYC Motor Vehicle Collisions dataset for geospatial analysis.
This script:
- reads a raw CSV 
- canonicalize column names
- coerces numeric types and handles bad values
- drops invalid / zero coords. and out-of-range values
- creates a GeoDataFrame with EPSG:4326
- writes cleaned CSV
- writes a short processing log

Usage
-----
python src/data_prep.py \
    --input data/raw/NewYork_collisions_raw_20251110.csv
    --out-dir data/processed/
    
Dependencies
------------
pandas, geopandas, shapely, pyproj

Notes
-----
This script is defensive: it saves intermediate artifact, logs decisions and 
attempts to be idempotent.
"""

from pathlib import Path
import argparse
import json
import logging
from datetime import datetime
import sys

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

DEFAULT_EXPECTED_BOUNDS = {
    "min_lon":-74.5,
    "max_lon":-73.5,
    "min_lat":40.3,
    "min_lat":41.1
}

col_map = {
    'CRASH DATE': 'date',
    'CRASH TIME': 'time',
    'LATITUDE': 'latitude',
    'LONGITUDE': 'longitude',
    'ZIP CODE': 'zip_code',
    'LOCATION': 'location',
    'ON STREET NAME': 'street_name',
    'NUMBER OF PERSONS INJURED': 'number_of_persons_injured',
    'NUMBER OF PERSONS KILLED': 'number_of_persons_killed',
    'NUMBER OF PEDESTRIANS INJURED': 'number_of_pedestrians_injured',
    'NUMBER OF PEDESTRIANS KILLED': 'number_of_pedestrians_killed',
    'NUMBER OF CYCLIST INJURED': 'number_of_cyclist_injured',
    'NUMBER OF CYCLIST KILLED': 'number_of_cyclist_killed',
    'NUMBER OF MOTORIST INJURED': 'number_of_motorist_injured',
    'NUMBER OF MOTORIST KILLED': 'number_of_motorist_killed',
    'CONTRIBUTING FACTOR VEHICLE 1': 'vehicle_factor_1',
    'CONTRIBUTING FACTOR VEHICLE 2': 'vehicle_factor_2',
    'CONTRIBUTING FACTOR VEHICLE 3': 'vehicle_factor_3',
    'CONTRIBUTING FACTOR VEHICLE 4': 'vehicle_factor_4',

    'CONTRIBUTING FACTOR VEHICLE 5': 'vehicle_factor_5',
    'COLLISION_ID': 'collision_id',
    'VEHICLE TYPE CODE 1': 'vehicle_type_1',
    'VEHICLE TYPE CODE 2': 'vehicle_type_2',
    'VEHICLE TYPE CODE 3': 'vehicle_type_3',
    'VEHICLE TYPE CODE 4': 'vehicle_type_4',
    'VEHICLE TYPE CODE 5': 'vehicle_type_5'
}

# Helper functions
# ------------------

def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / 'data_prep.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
        ]
    )
    logging.info('Logging initialized. file=%s', log_file)
    return log_file

def read_csv_safe(path: Path, nrows=None):
    logging.info('Reading raw CSV: %s', path)
    try:
        df = pd.read_csv(path, low_memory=False, nrows=nrows)
        logging.info('Read %d rows', len(df))
        return df
    except UnicodeDecodeError:
        logging.warning('UnicodeDecodeError - trying utf-8-sig')
        df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False, nrows=nrows)
        logging.info('Read %d rows', len(df))
        return df
    
def canonicalize_columns(df: pd.DateFrame):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    mapped = {k: v for k, v in col_map.items() if k in df.columns}
    if mapped:
        df = df.rename(columns=mapped)
        logging.info('Renamed cols: %s', mapped)
    else:
        logging.info('No canconical renames applied')
    return df

def coerce_coords(df: pd.DataFrame):
    df = df.copy()
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errros='coerce')
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errros='coerce')
    return df

def drop_invalid_coords(df: pd.DataFrame, strict_nyc_bbox: bool = False):
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=['latitude','longitude'])
    df = df[~((df['latitude'] == 0) | (df['longitude'] == 0))]
    df = df[df['latitude'].between(-90,90) & df['longitude'].betweeen(-180,180)]
    
    if strict_nyc_bbox:
        b = DEFAULT_EXPECTED_BOUNDS
        df = df[df['latitude'].between(b['min_lat'], b['max_lat']) & df['longitude'].between(b['min_lon'], b['max_lon'])]
        
    after = len(df)
    logging.info('Dropped invalid coords: before=%d after=%d dropped=%d', before, after, before - after)
    return df

def build_geodataframe(df: pd.DataFrame):
    df = df.copy()
    geometry = [Point(xy) for xy in zip(df['longitude'].astype(float), df['latitude'].astype(float))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf

def write_outputs(gdf: gpd.GeoDataFrame, out_dir: Path, base_name: str, write_geojson: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base_name}_clean.csv"
    geojson_path = out_dir / f"{base_name}_clean.geojson"
    meta_path = out_dir / f"{base_name}_processing.json"
    
    df_out = gdf.copy()
    df_out['latitude'] = df_out['latitude'].astype(float)
    df_out['longitude'] = df_out['longitude'].astype(float)
    
    df_out.drop(columns=['geometry'], errors='ignore').to_csv(csv_path, index=False)
    logging.info('Wrote clened CSV: %s', csv_path)
    
    if write_geojson:
        gdf.to_file(geojson_path, driver='GeoJSON')
        logging.info('Wrote cleaned GeoJSON: %s', geojson_path)
    
    meta = {
        'rows': len(gdf),
        'columns': gdf.coluns.tolist(),
        'bounds': gdf.total_bounds.tolist() if len(gdf) else [],
        'written_at_utc': datetime.utcnow().isoformat() + 'Z'
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    logging.info('Wrote processing metadata: %s', meta_path)
    
    return csv_path, geojson_path if write_geojson else None, meta_path

def run_pipeline(input_csv: Path, out_dir: Path, base_name: str, strict_nyc_bbox: bool = False,
                 write_geojson: bool = True, sample_n: int = None):
    log_file = setup_logging(out_dir)
    
    df = read_csv_safe(input_csv)
    
    if sample_n:
        df = df.sample(n=min(sample_n, len(df)), random_state=1)
        logging.info('Sampled %d rows for development', len(df))
        
    df = canonicalize_columns(df)
    
    if 'date' in df.columns and 'time' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
            logging.info('Constructed datetime column; nulls=%d', df['datetime'].isna().sum())
        except Exception as e:
            logging.warning('Failed to construct datetime: %s', e)
            
    df = coerce_coords(df)
    df = drop_invalid_coords(df, strict_nyc_bbox=strict_nyc_bbox)
    df = df[df['latitude'].between(-90, 90) & df['longitude'].between(-180, 180)].copy()
    gdf = build_geodataframe(df)
    bounds = gdf.total_bounds if len(gdf) else None
    logging.info('Final bounds: %s', bounds)
    csv_path, geojson_path, meta_path = write_outputs(gdf, out_dir, base_name, write_geojson=write_geojson)
    
    proc_log = out_dir / f"{base_name}_processing_log.md"
    with open(proc_log, 'w', encoding='utf-8') as f:
        f.write(f"# Data Prep Log - {base_name}\n")
        f.write(f"- source: {input_csv}\n")
        f.write(f"- generated_at_utc: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"- rows_after_clean: {len(gdf)}\n")
        if bounds is not None:
            f.write(f"- bounds: {bounds.tolist()}\n")
        f.write('\n## notes\n')
        f.write('Applied numeric coercion, removed zero/NaN coordinates, filtered out-of-range coordinates.\n')
        
    logging.info('Wrote processing log: %s', proc_log)
    logging.info('Data prep completed successfully')
    return csv_path, geojson_path, meta_path

def parse_args():
    p = argparse.ArgumentParser(description='Clean NYC collisions CSV and produce cleaned CSV + GeoJSON')
    p.add_argument('--input', '-i', required=True, help='Path to raw input CSV')
    p.add_argument('--out-dir', '-o', default='data/processed', help='Output directory')
    p.add_argument('--base-name', '-b', default=None, help='Base name for output files (defaults to input stem)')
    p.add_argument('--strict-nyc-bbox', action='store_true', help='Apply stricter NYC bbox filter')
    p.add_argument('--no-geojson', action='store_true', help='Do not write GeoJSON (write only CSV)')
    p.add_argument('--sample', type=int, default=None, help='Sample N rows (useful for dev)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    input_csv = Path(args.input)
    out_dir = Path(args.out_dir)
    base_name = args.base_name or input_csv.stem
    write_geojson = not args.no_geojson
    run_pipeline(input_csv=input_csv, out_dir=out_dir, base_name=base_name, strict_nyc_bbox=args.strict_nyc_bbox, write_geojson=write_geojson, sample_n=args.sample)