import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.polygons_to_labels import *
from helperToolz.helpsters import *

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

state = 'BRB'
state_year = f'{state}_{year}'

polygon_path = f'/data/{origin}fields/IACS/1_Polygons/GSA-DE_{spat_temp_id}.geoparquet'

vrt_out = path_safe(f'{origin}fields/Auxiliary/vrt/{state}/{year}/')
reduced_files = reduce_forceTSA_output_to_validmonths(f'{origin}force/output/{state}/{year}/', 3, 8)
ordered_files = force_order_Colors_for_VRT(reduced_files, ['BLU', 'GRN', 'RED', 'BNR'], [f'MONTH-{d:02d}' for d in range(3,9,1)])


if os.path.isdir(vrt_out):
    if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
        print('VRT seems to be already computated, probably to create masks based on IACS')
    else:
        force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])
else:
    os.makedirs(vrt_out)
    force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])


vrt_path = getFilelist(f"/data/{origin}/fields/Auxiliary/vrt/{state}/{year}/{dirfinder(f'/data/{origin}/fields/Auxiliary/vrt/{state}/{year}/')[0]}",
                       '.vrt', deep=True)[0]

# convert lines to polyongs
polygons_to_lines(polygon_path,
                  f'/data/{origin}fields/IACS/2_Lines/IACS_{state_year}.gpkg',
                  exclude_list)

# rasterize lines
rasterize_lines(f'/data/{origin}fields/IACS/2_Lines/GSA-DE_{state_year}.gpkg', 
                vrt_path, 
                f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_true.tif', all_touch=True)

rasterize_lines(f'/data/{origin}fields/IACS/2_Lines/GSA-DE_{state_year}.gpkg', 
                vrt_path, 
                f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_false.tif', all_touch=False)

# make a crop mask
make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_true.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=f'/data/{origin}fields/IACS/4_Crop_mask/{state}/{year}/GSA-DE_{state_year}_cropMask_lines_touch_true_crop_touch_true.tif',
               all_touch=True)

make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_false.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=f'/data/{origin}fields/IACS/4_Crop_mask/{state}/{year}/GSA-DE_{state_year}_cropMask_lines_touch_false_crop_touch_true.tif',
               all_touch=True)




make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_true.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=f'/data/{origin}fields/IACS/4_Crop_mask/{state}/{year}/GSA-DE_{state_year}_cropMask_lines_touch_true_crop_touch_false.tif',
               all_touch=False)

make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'/data/{origin}fields/IACS/3_Rasterized_lines/{year}/GSA-DE_{state_year}_lines_touch_false.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=f'/data/{origin}fields/IACS/4_Crop_mask/{state}/{year}/GSA-DE_{state_year}_cropMask_lines_touch_false_crop_touch_false.tif',
               all_touch=False)



# # This should happen at tile level, when we use S2 to finetune model --> labels needed for training
# # rasterized lines to 3D label
# print('take the long way')
# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_true.tif', 
#                       f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_true_MTSK.tif')

# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_false.tif', 
#                        f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_false_MTSK.tif')