import geopandas as gpd

# Load the GeoParquet file
gdf = gpd.read_parquet("your_file.parquet")

# Ensure the geometries are polygons
if not all(gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
    raise ValueError("The file must contain Polygon or MultiPolygon geometries.")

# Convert polygons to lines
gdf["geometry"] = gdf.geometry.boundary
#export
gdf.to_file("converted_lines.geojson", driver="GeoJSON")  # Optional: Save as GeoJSON

print("Conversion complete: Polygons converted to lines.")