import requests
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import shape
import os
import sys
import json
import time

if 'workhorse' not in sys.executable.split('/'):
    origin = 'workspace/'
    sys.path.append('/media/')
else:
    origin = 'data/Aldhani/eoagritwin/'
    sys.path.append('/home/potzschf/repos/')

from helperToolz.helpsters import path_safe

outPath = path_safe('/data/Aldhani/eoagritwin/fields/01_IACS/1_Polygons/Hessen/')
hessen_url = "https://inspire-geo.ibykus.net/geoserver/lawi/wfs"

MAX_RETRIES = 5
BATCH_SIZE = 1000
RETRY_DELAY = 15

# ── Step 1: GetCapabilities ───────────────────────────────────────────────────
cap = requests.get(
    hessen_url,
    params={"service": "WFS", "version": "2.0.0", "request": "GetCapabilities"},
    timeout=60
)
cap.raise_for_status()
root = ET.fromstring(cap.content)
ns = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "ows": "http://www.opengis.net/ows/1.1",
}
layers = []
for ft in root.findall(".//wfs:FeatureType", ns):
    name  = ft.find("wfs:Name", ns)
    title = ft.find("wfs:Title", ns)
    crs   = ft.find("wfs:DefaultCRS", ns)
    layers.append({
        "name":  name.text  if name  is not None else None,
        "title": title.text if title is not None else None,
        "crs":   crs.text   if crs   is not None else "EPSG:25832",
    })

print(f"Found {len(layers)} layer(s):")
for l in layers:
    print(f"  {l['name']} — {l['title']} ({l['crs']})")

# ── Step 2: Download each layer ───────────────────────────────────────────────
for layer in layers:
    print(f"\n{'='*50}")
    print(f"Downloading: {layer['title']} ({layer['name']})")
    print(f"{'='*50}")

    safe_title  = layer["title"].replace(" ", "_").replace("/", "_")
    output_file = f"{outPath}{safe_title}.gpkg"
    checkpoint_file = f"{outPath}{safe_title}.checkpoint"

    if os.path.exists(output_file) and not os.path.exists(checkpoint_file):
        print(f"  Already complete, skipping: {output_file}")
        continue

    params = {
        "service":      "WFS",
        "version":      "2.0.0",
        "request":      "GetFeature",
        "typeNames":    layer["name"],
        "outputFormat": "application/json",
        "count":        BATCH_SIZE,
        "startIndex":   0,
    }

    # Checkpoint laden
    first_batch = True
    total = 0
    crs = layer["crs"]

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            cp = json.load(f)
        params["startIndex"] = cp["startIndex"]
        total = cp["total"]
        crs   = cp.get("crs", crs)
        first_batch = False
        print(f"  Checkpoint found: resuming at startIndex={params['startIndex']} ({total} already saved)")

    batch = params["startIndex"] // BATCH_SIZE
    layer_failed = False

    while True:
        print(f"  Batch {batch + 1}, startIndex={params['startIndex']} ...")

        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(hessen_url, params=params, timeout=120)
                r.raise_for_status()
                data = r.json()

                if data is None:
                    raise ValueError("Server returned null/empty JSON response")
                if not isinstance(data, dict):
                    raise ValueError(f"Unexpected response type: {type(data)}")

                break  # Erfolg

            except Exception as e:
                print(f"  ⚠ Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"    Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)

        if data is None:
            print(f"  ✗ Layer '{layer['name']}' failed after {MAX_RETRIES} attempts — skipping.")
            layer_failed = True
            break

        # CRS aus Response übernehmen (mit None-Guard)
        detected_crs = (data.get("crs") or {}).get("properties", {}).get("name")
        if detected_crs:
            crs = detected_crs

        features = data.get("features", [])
        if not features:
            print("  No more features.")
            break

        # Batch in GeoDataFrame und direkt auf Disk
        geometries = [shape(f["geometry"]) for f in features]
        properties = [f["properties"] for f in features]
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)

        if first_batch:
            gdf.to_file(output_file, driver="GPKG", engine="pyogrio")
            first_batch = False
        else:
            gdf.to_file(output_file, driver="GPKG", engine="pyogrio", mode="a")

        total += len(features)
        params["startIndex"] += len(features)
        batch += 1
        print(f"  → {len(features)} features saved, total: {total}")

        # Checkpoint schreiben
        with open(checkpoint_file, "w") as f:
            json.dump({"startIndex": params["startIndex"], "total": total, "crs": crs}, f)

    if not layer_failed:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print(f"  ✓ Saved: {output_file} ({total} features)")
    
print("\nAll done!")