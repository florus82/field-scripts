import sys

if 'workhorse' not in sys.executable.split('/'):
    origin = 'workspace/'
    sys.path.append('/media/')
else:
    origin = 'data/Aldhani/eoagritwin/'
    sys.path.append('/home/potzschf/repos/')

from helperToolz.polygons_to_labels import *
from helperToolz.helpsters import *
import xarray as xr
import rioxarray

IACS_path = f'/{origin}fields/IACS/1_Polygons/'
temp_folder = f'/{origin}fields/IACS/temp_trash/'
out_folder = f'/{origin}fields/Fine_tune/'
FORCE_folder = f'/{origin}force/output/'
aux_vrt_path = f'/{origin}fields/Auxiliary/vrt/'



# set state and year
states = ['Brandenburg', 'Niedersachsen', 'MV', 'NRW', 'Saarland']
state_folders = ['BRB', 'LSA', 'MV', 'NRW', 'SL']
state_types = ['.geoparquet','.geoparquet','.shp','.geoparquet','.shp']
state_exclude_columns = ['EC_hcat_n', 'EC_hcat_n', 'NU_BEZ', 'EC_hcat_n', 'BEZ']
state_burners = ['field_id', 'field_id', 'ID', 'field_id','LWREFSID']
years = [2022, 2024, 2023, 2020, 2021]

chip_size = 256


# get IACS file
for state, state_folder, state_type, state_exclude_column, state_burner, year in\
    zip(states, state_folders, state_types, state_exclude_columns, state_burners,years):

    print(state)
    path = [file for file in getFilelist(IACS_path + state_folder, state_type) if str(year) in file][0]

    # first, make raster vrts of FORCE output that can be used ass extent raster for IACS rasterization
    force_path = f'{FORCE_folder}{state_folder}/{year}/'
    getFilelist(force_path, '.tif', deep=True)

    reduced_files = reduce_forceTSA_output_to_validmonths(force_path, 3, 8)
    ordered_files = force_order_Colors_for_VRT(reduced_files, ['BLU', 'GRN', 'RED', 'BNR'], [f'MONTH-{d:02d}' for d in range(3,9,1)])

    vrt_out = path_safe(f'{aux_vrt_path}{state_folder}/{year}/')

    if os.path.isdir(vrt_out):
        if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
            print('VRT seems to be already computated, probably to create masks based on IACS')
        else:
            force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])
    else:
        os.makedirs(vrt_out)
        force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])

    vrt_cube_path = [file for file in getFilelist(vrt_out, '.vrt', deep=True) if 'Cube' in file][0]

    fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
    polyon_lines_path = f'{temp_folder}{state}/{state}_{year}_lines.gpkg'
    borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines_touch_true.tif'
    fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'

    # 1. rasterize fields (polygons) All_touch = False
    make_crop_mask(path_to_polygon=path,
                path_to_extent_raster=vrt_cube_path,
                path_to_mask_out=path_safe(fields_path),
                all_touch=False, 
                categories=EXCLUDE_LIST,
                category_col=state_exclude_column) 
    print('fields rasterized')
    # 2 .rasterize borders (after polygons to lines) All_touch = True
    polygons_to_lines(path_to_polygon=path,
                    path_to_lines_out=polyon_lines_path,
                    categories=EXCLUDE_LIST,
                    category_col=state_exclude_column)

    # rasterize lines
    rasterize_lines(path_to_lines=polyon_lines_path, 
                    path_to_extent_raster=vrt_cube_path, 
                    path_to_rasterlines_out=borders_path,
                    all_touch=True)
    print('field borders rasterized')
    # 3. get field IDs 
    make_crop_mask(path_to_polygon=path,
                path_to_extent_raster=vrt_cube_path,
                path_to_mask_out=fieldsID_path,
                all_touch=False, 
                categories=EXCLUDE_LIST,
                category_col=state_exclude_column,
                burn_col=state_burner) 
    print('unique fieldIDs rasterized')

    # 4. distance to border raster
    # as calculating the distance layer for the entire state is computational-wise too expensive, we cut image first in chips
    # first, load force stack to get number of rows and cols for chipping raster
    force_arr = loadVRTintoNumpyAI4(f'{aux_vrt_path}{state_folder}/{year}/{getFORCExyRangeName(get_forcetiles_range(reduced_files))}')
    row_col_ind =get_row_col_indices(chip_size, 0, force_arr.shape[2], force_arr.shape[3])
    row_start = row_col_ind[0]
    row_end   = row_col_ind[1]
    col_start = row_col_ind[2]
    col_end   = row_col_ind[3]

    gtiff_driver = gdal.GetDriverByName('GTiff')
    vrt_ds = gdal.Open(fieldsID_path)
    geoTF = vrt_ds.GetGeoTransform()
    prj = vrt_ds.GetProjection()

    arr_IDs = vrt_ds.GetRasterBand(1).ReadAsArray()

    if arr_IDs.shape != force_arr.shape[-2:]:
        raise ValueError(f"Shape mismatch of S2 iamgery and rasterized IACS products: S2:{force_arr.shape[-2:]} IACS:{arr_IDs.shape}")

    for i in range(len(row_end)):
        for j in range(len(col_end)):

            arr_dist = polygon_distance_normalized(arr_IDs[row_start[i]:row_end[i], col_start[j]:col_end[j]])
            distance_path = f'{temp_folder}{state}/{state}_{year}_rs{row_start[i]}_cs{col_start[j]}_distance_to_border.tif'
            # export labels as tif chips
            out_ds = gtiff_driver.Create(path_safe(distance_path), int(chip_size), int(chip_size), 1, gdal.GDT_Float32)
            # change the Geotransform for each chip
            geotf = list(geoTF)
            # get column and rows from filenames
            geotf[0] = geotf[0] + geotf[1] * col_start[j]
            geotf[3] = geotf[3] + geotf[5] * row_start[i]
            #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')

            out_ds.SetGeoTransform(tuple(geotf))
            out_ds.SetProjection(prj)

            out_ds.GetRasterBand(1).WriteArray(arr_dist)
            del out_ds

    # create a distance vrt
    dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
    vrt = gdal.BuildVRT(dist_path,
                        [file for file in getFilelist(f'{temp_folder}{state}/', '.tif') if all(substr in file for substr in ['rs', 'cs'])])
    vrt = None
    print('distance calculated rasterized')
