import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
from helperToolz.dicts_and_lists import *
from helperToolz.guzinski import * 
import geopandas as gpd
from collections import defaultdict


from joblib import Parallel, delayed
from datetime import datetime, timedelta
import time


ext = '07'
bound = '02'
aoi = 'GERMANY'
# field path
origin = "/data/Aldhani/eoagritwin/fields/"
# get uncertainties of fields
if aoi != 'GERMANY':
    path_to_vrt = f"{origin}04_Predictions/{aoi}/FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47/2023/vrt/ThuenenMask_256_20.vrt"
    path_fields = f"{origin}06_Segmentation/{aoi}/FromScratch_dilate_T/2023/ThuenenMasked/relabelled/{aoi}_FromScratch_dilate_T_2023_ThuenenMasked_ext_{ext}_bound_{bound}.tif"
    path_fields_pol = f"{origin}07_Polygonized/{aoi}/FromScratch_dilate_T/2023/ThuenenMasked/ThuenenMasked_ext_{ext}_bound_{bound}_10m.gpkg"
else:
    path_to_vrt = f"{origin}04_Predictions/{aoi}/FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47/2023/vrt/masked_chips_256_20.vrt"
    path_fields = f"{origin}06_Segmentation/{aoi}/FromScratch_dilate_T/2023/MaskedThuenen/relabelled/{aoi}_FromScratch_dilate_T_2023_MaskedThuenen_ext_{ext}_bound_{bound}.tif"
    path_fields_pol = f"{origin}07_Polygonized/{aoi}/FromScratch_dilate_T/2023/MaskedThuenen/MaskedThuenen_ext_{ext}_bound_{bound}_10m.gpkg"



ds_b1 = gdal.Open(path_to_vrt)
print(f'{ds_b1.RasterXSize}   {ds_b1.RasterYSize}')
print(ds_b1.GetGeoTransform())


ds_fields = gdal.Open(path_fields)
print(f'{ds_fields.RasterXSize}   {ds_fields.RasterYSize}')
print(ds_fields.GetGeoTransform())

arr_field = ds_fields.GetRasterBand(1).ReadAsArray()
arr_b1 = ds_b1.GetRasterBand(1).ReadAsArray()


# ############# cut off rows and cols manually for Brandenburg!!!!!!!!!!!!!!!!!!!!!!!!!!!
# arr_b1 = arr_b1[:-4,:-2]
# ############# cut off rows and cols manually for GERMANY!!!!!!!!!!!!!!!!!!!!!!!!!!!
arr_export = zeros = np.zeros_like(arr_field)
arr_field = arr_field[236:-2766,1416:]
arr_b1 = arr_b1[:,:-69]

##########################################################################

mask = (arr_field != 0) & (~np.isnan(arr_field)) & (~np.isnan(arr_b1))
flat_field = arr_field[mask].astype(int)
flat_b1 = arr_b1[mask]

# Create DataFrame
df = pd.DataFrame({'FieldID': flat_field, 'B1': flat_b1})

# Compute median per field
median_df = df.groupby('FieldID', as_index=False)['B1'].median()
median_df.rename(columns={'B1': 'B1_median'}, inplace=True)
median_df.to_csv(path_safe(f"{origin}output/uncertainty/{aoi}_mean_b1_score_per_field_ext{ext}_bound{bound}.csv"), index=False)

# Create a lookup map
median_map = dict(zip(median_df['FieldID'], median_df['B1_median']))

# Fill raster
arr_field_fill = np.vectorize(median_map.get)(arr_field)
arr_export[236:-2766,1416:] = arr_field_fill

npTOdisk(arr_export, path_fields,
         f"{origin}output/uncertainty/{aoi}_Fields_MEDIAN_B1_SCORE_ext{ext}_bound{bound}.tif",
         noData=0, d_type=gdal.GDT_Float32)




# ############################################################ field borders
# print("start with field borders")

# # Convert polygons to lines
# gdf = gpd.read_file(path_fields_pol)

# # Ensure the geometries are polygons
# if not all(gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
#     raise ValueError("The file must contain Polygon or MultiPolygon geometries.")


# gdf['geometry'] = gdf.geometry.boundary
# #export
# field_lines_path = f"{origin}output/uncertainty/temp/{aoi}_Field_Lines_ext{ext}_bound{bound}.gpkg"
# gdf.to_file(path_safe(field_lines_path), driver='GPKG') 


# ##### open field vector file
# field = ogr.Open(field_lines_path)
# field_lyr = field.GetLayer(0)

# output = f"{origin}output/uncertainty/temp/{aoi}_Field_Borders_ext{ext}_bound{bound}.tif"

# if not os.path.exists(output):
#     ds = gdal.Open(path_to_vrt)
#     target_ds = gdal.GetDriverByName('GTiff').Create(output, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_UInt32)
#     target_ds.SetGeoTransform(ds.GetGeoTransform())
#     target_ds.SetProjection(ds.GetProjection())

#     gdal.RasterizeLayer(target_ds, [1], field_lyr, options = ["ALL_TOUCHED=True", "ATTRIBUTE=FieldID"])
#     target_ds = None


# # extract b2 at rasterized borders
# ds_b2 = gdal.Open(path_to_vrt)
# arr_b2 = ds_b2.GetRasterBand(2).ReadAsArray()

# ds_border = gdal.Open(output)
# arr_border = ds_border.GetRasterBand(1).ReadAsArray()


# mask = (arr_border != 0) & (~np.isnan(arr_border)) & (~np.isnan(arr_b2))
# flat_field = arr_border[mask].astype(int)
# flat_b2 = arr_b2[mask]


# # Create DataFrame
# df = pd.DataFrame({'BorderID': flat_field, 'B2': flat_b2})

# # Compute median per field
# median_df = df.groupby('BorderID', as_index=False)['B2'].median()
# median_df.rename(columns={'B2': 'B2_median'}, inplace=True)
# median_df.to_csv(f"{origin}output/uncertainty/{aoi}_mean_b2_score_per_field_ext{ext}_bound{bound}.csv", index=False)


# # Create a lookup map
# median_map = dict(zip(median_df['BorderID'], median_df['B2_median']))

# # Fill raster
# arr_border_fill = np.vectorize(median_map.get)(arr_border)

# npTOdisk(arr_border_fill, output,
#          f"{origin}output/uncertainty/{aoi}_Borders_MEDIAN_B2_SCORE_ext{ext}_bound{bound}.tif",
#          noData=0, d_type=gdal.GDT_Float32)