import requests
import geopandas as gpd
from shapely.geometry import shape
import sys
import os
import json
import time

if 'workhorse' not in sys.executable.split('/'):
    origin = 'workspace/'
    sys.path.append('/media/')
else:
    origin = 'data/Aldhani/eoagritwin/'
    sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import path_safe

url = "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LW-BW_GISELA_landw_Parzellen"

storPath = path_safe('/data/Aldhani/eoagritwin/fields/01_IACS/1_Polygons/BW/')

years = [2022]

MAX_RETRIES = 5
BATCH_SIZE = 2000
RETRY_DELAY = 10  # seconds

for year in years:
    params = {
        "request": "GetFeature",
        "service": "WFS",
        "version": "2.0.0",
        "outputFormat": "json",
        "typeNames": f"lw:v_gisela_landw_parzellen_{year}",
        "count": BATCH_SIZE,
        "startIndex": 0
    }

    output_file = f"{storPath}BW_Parzellen_{year}.gpkg"
    layer_name = f"Parzellen{year}"

    checkpoint_file = f"{storPath}BW_Parzellen_{year}.checkpoint"

    first_batch = True
    total = 0

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        params["startIndex"] = checkpoint["startIndex"]
        total = checkpoint["total"]
        first_batch = False
        print(f"  Checkpoint gefunden: Weiter ab startIndex={params['startIndex']} ({total} bereits gespeichert)")

    batch = params["startIndex"] // BATCH_SIZE
    
    while True:
        print(f"Lade Batch {batch+1}, startIndex={params['startIndex']} ...")

        data = None
        current_batch_size = params["count"]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(url, params=params, timeout=120)
                r.raise_for_status()
                data = r.json()
                break  # Erfolg

            except requests.exceptions.JSONDecodeError:
                print(f"  ⚠ JSON-Fehler (Versuch {attempt}/{MAX_RETRIES})")
                print(f"    HTTP Status: {r.status_code}, Antwortlänge: {len(r.text)} Zeichen")

                # Batch-Größe halbieren als Ausweichstrategie
                if attempt == 2 and current_batch_size > 250:
                    current_batch_size = current_batch_size // 2
                    params["count"] = current_batch_size
                    print(f"    → Batch-Größe reduziert auf {current_batch_size}")

                if attempt < MAX_RETRIES:
                    print(f"    Warte {RETRY_DELAY}s vor erneutem Versuch...")
                    time.sleep(RETRY_DELAY)

            except requests.exceptions.RequestException as e:
                print(f"  ⚠ Request-Fehler (Versuch {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        if data is None:
            print(f"  ✗ Batch {batch+1} nach {MAX_RETRIES} Versuchen fehlgeschlagen. Abbruch.")
            print(f"    Checkpoint gespeichert – starte das Skript neu zum Fortfahren.")
            break

        features = data.get("features", [])
        if not features:
            print("Keine weiteren Features – fertig!")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            break

        geometries = [shape(f["geometry"]) for f in features]
        properties = [f["properties"] for f in features]
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:25832")

        if first_batch:
            gdf.to_file(output_file, layer=layer_name, driver="GPKG", engine="pyogrio")
            first_batch = False
        else:
            gdf.to_file(output_file, layer=layer_name, driver="GPKG", engine="pyogrio", mode="a")

        total += len(features)
        params["startIndex"] += len(features)
        batch += 1
        # Batch-Größe nach Erfolg wieder erhöhen
        params["count"] = BATCH_SIZE
        print(f"  → {len(features)} Features gespeichert, gesamt: {total}")

        # Checkpoint schreiben
        with open(checkpoint_file, "w") as f:
            json.dump({"startIndex": params["startIndex"], "total": total}, f)

    print(f"\nFertig! {total} Features gespeichert in '{output_file}'\n")