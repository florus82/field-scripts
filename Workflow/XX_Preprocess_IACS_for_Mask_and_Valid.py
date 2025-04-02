import sys
sys.path.append('/home/potzschf/repos/fields/from_Philippe')
from polygons_to_labels import *

exclude_list = ['afforestation_reforestation',
                'greenhouse_foil_film',
                'not_known_and_other', 
                'nurseries_nursery', 
                'tree_wood_forest', 
                'unmaintained']

# convert lines to polyongs
polygons_to_lines('/data/fields/IACS/1_Polygons/GSA-DE_BRB-2019.geoparquet',
                  '/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg',
                  exclude_list)

# rasterize lines
rasterize_lines('/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg', 
                '/data/fields/Auxiliary/vrt/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt', 
                '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', all_touch=True)

rasterize_lines('/data/fields/IACS/2_Lines/GSA-DE_BRB-2019.gpkg', 
                '/data/fields/Auxiliary/vrt/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt', 
                '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', all_touch=False)

# make a crop mask
make_crop_mask('/data/fields/IACS/1_Polygons/GSA-DE_BRB-2019.geoparquet', 
               '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', 
               '/data/fields/Auxiliary/vrt/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt', 
               '/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_lines_touch_true_cropMask.tif')

make_crop_mask('/data/fields/IACS/1_Polygons/GSA-DE_BRB-2019.geoparquet', 
               '/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', 
               '/data/fields/Auxiliary/vrt/Force_X_from_64_to_73_Y_from_39_to_47//Force_X_from_64_to_73_Y_from_39_to_47_0.vrt', 
               '/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_lines_touch_false_cropMask.tif')
# This should happen at tile level, when we use S2 to finetune model --> labels needed for training
# # rasterized lines to 3D label
# make_multitask_labels('/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_true.tif', 
#                       '/data/fields/IACS/5_Multitask_labels/GSA-DE_BRB-2019_lines_touch_true_MTSK.tif')

# make_multitask_labels('/data/fields/IACS/3_Rasterized_lines/GSA-DE_BRB-2019_lines_touch_false.tif', 
#                        '/data/fields/IACS/5_Multitask_labels/GSA-DE_BRB-2019_lines_touch_false_MTSK.tif')