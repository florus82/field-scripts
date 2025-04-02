import sys
sys.path.append('/home/potzschf/repos/fields/from_Philippe')
from polygons_to_labels import *

exclude_list = ['afforestation_reforestation',
                'greenhouse_foil_film',
                'not_known_and_other', 
                'nurseries_nursery', 
                'tree_wood_forest', 
                'unmaintained']

polygon_path = '/data/fields/IACS/1_Polygons/GSA-DE_BRB-2019.geoparquet'
vrt_path = '/data/fields/Auxiliary/vrt/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt'
# convert lines to polyongs
polygons_to_lines(polygon_path,
                  '/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg',
                  exclude_list)

# rasterize lines
rasterize_lines('/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg', 
                vrt_path, 
                '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', all_touch=True)

rasterize_lines('/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg', 
                vrt_path, 
                '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', all_touch=False)

# make a crop mask
make_crop_mask(polygon_path, 
               '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', 
               vrt_path, 
               '/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_cropMask_lines_touch_true.tif')

make_crop_mask(polygon_path, 
               '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', 
               vrt_path, 
               '/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_cropMask_lines_touch_false.tif')



# This should happen at tile level, when we use S2 to finetune model --> labels needed for training
# # rasterized lines to 3D label
# make_multitask_labels('/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', 
#                       '/data/fields/IACS/5_Multitask_labels/GSA-DE_BRB-2019_lines_touch_true_MTSK.tif')

# make_multitask_labels('/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', 
#                        '/data/fields/IACS/5_Multitask_labels/GSA-DE_BRB-2019_lines_touch_false_MTSK.tif')