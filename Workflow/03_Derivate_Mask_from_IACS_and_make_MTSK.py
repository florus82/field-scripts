import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.polygons_to_labels import *

workhorse = True

if workhorse:
    origin = 'Aldhani/eoagritwin/'
else:
    origin = ''

############# define attirubtes that are excluded (trees or other LC that might not be related to spatially detectable field)
exclude_list = ['afforestation_reforestation',
                'greenhouse_foil_film',
                'not_known_and_other', 
                'nurseries_nursery', 
                'tree_wood_forest', 
                'unmaintained']

year = 2023
spat_temp_id = f'BRB-{year}'

polygon_path = f'/data/{origin}fields/IACS/1_Polygons/GSA-DE_{spat_temp_id}.geoparquet'
vrt_path = f'/data/{origin}fields/Auxiliary/vrt/{year}/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt'
# convert lines to polyongs
polygons_to_lines(polygon_path,
                  f'/data/{origin}fields/IACS/2_Lines/GSA-DE_{spat_temp_id}.gpkg',
                  exclude_list)

# rasterize lines
rasterize_lines(f'/data/{origin}fields/IACS/2_Lines/GSA-DE_{spat_temp_id}.gpkg', 
                vrt_path, 
                f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_true.tif', all_touch=True)

rasterize_lines(f'/data/{origin}fields/IACS/2_Lines/GSA-DE_{spat_temp_id}.gpkg', 
                vrt_path, 
                f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_false.tif', all_touch=False)

# make a crop mask
make_crop_mask(polygon_path, 
               f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_true.tif', 
               vrt_path, 
               f'/data/{origin}fields/IACS/4_Crop_mask/{year}/GSA-DE_{spat_temp_id}_cropMask_lines_touch_true_crop_touch_true.tif', True)

make_crop_mask(polygon_path, 
               f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_false.tif', 
               vrt_path, 
               f'/data/{origin}fields/IACS/4_Crop_mask/{year}/GSA-DE_{spat_temp_id}_cropMask_lines_touch_false_crop_touch_true.tif', True)




make_crop_mask(polygon_path, 
               f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_true.tif', 
               vrt_path, 
               f'/data/{origin}fields/IACS/4_Crop_mask/{year}/GSA-DE_{spat_temp_id}_cropMask_lines_touch_true_crop_touch_false.tif', False)

make_crop_mask(polygon_path, 
               f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{spat_temp_id}_lines_touch_false.tif', 
               vrt_path, 
               f'/data/{origin}fields/IACS/4_Crop_mask/{year}/GSA-DE_{spat_temp_id}_cropMask_lines_touch_false_crop_touch_false.tif', False)



# # This should happen at tile level, when we use S2 to finetune model --> labels needed for training
# # rasterized lines to 3D label
# print('take the long way')
# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_true.tif', 
#                       f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_true_MTSK.tif')

# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_false.tif', 
#                        f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_false_MTSK.tif')