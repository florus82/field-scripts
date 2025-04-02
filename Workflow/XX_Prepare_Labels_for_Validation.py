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