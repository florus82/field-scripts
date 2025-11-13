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
from helperToolz.feevos.rocksdbutils_copy import *
from math import ceil as mceil


################################################ create the images as nc and labels as tif files


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
Total_number_of_samples = 1500
gtiff_driver = gdal.GetDriverByName('GTiff')
nc_band_names = ["B2", "B3", "B4", "B8"]


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
                    all_touch=True,
                    dilate=True)
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

    vrt_ds = gdal.Open(fieldsID_path)
    geoTF = vrt_ds.GetGeoTransform()
    prj = vrt_ds.GetProjection()

    row_col_ind = get_row_col_indices(chip_size, 0, vrt_ds.RasterYSize, vrt_ds.RasterXSize)
    row_start = row_col_ind[0]
    row_end   = row_col_ind[1]
    col_start = row_col_ind[2]
    col_end   = row_col_ind[3]

    arr_IDs = vrt_ds.GetRasterBand(1).ReadAsArray()

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




# 5 get the number of samples to draw, stratified against state size, field and border fraction and number of fields
bands = 4
band_lkp = {0:'Field', 1:'Border',  2:'Distance', 3:'Fields'}
# initialize result lists
result = {
    'state': [],
    'band': [],
    'row_start': [],
    'row_end': [],
    'col_start': [],
    'col_end': [],
    'PixelCount': []
}

for state, year in zip(states, years):
    print(state)
    fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
    borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines_touch_true.tif'
    dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
    fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'
 
    arr_stack = stackReader(stack_tifs([fields_path, borders_path, dist_path, fieldsID_path], d_type=gdal.GDT_Float32))
    row_col_ind =get_row_col_indices(chip_size, 0, arr_stack.shape[0], arr_stack.shape[1])
    row_start = row_col_ind[0]
    row_end   = row_col_ind[1]
    col_start = row_col_ind[2]
    col_end   = row_col_ind[3]
    for i in range(len(row_end)):
        for j in range(len(col_end)):
            for band in range(bands):

                if band == 2:
                    continue

                sub = arr_stack[row_start[i]:row_end[i], col_start[j]:col_end[j],band]
                # take the sum and append
                result['state'].append(state)
                result['band'].append(band_lkp[band])
                result['row_start'].append(row_start[i])
                result['row_end'].append(row_end[i])
                result['col_start'].append(col_start[j])
                result['col_end'].append(col_end[j])
                # make a mask for all values that are not 0
                if band == 3:
                    result['PixelCount'].append(len(np.unique(sub))-1) # substract -10000
                else:
                    result['PixelCount'].append(np.count_nonzero(sub))

# Convert results dict to DataFrame
df = pd.DataFrame(result)

df_nonZero = df[df['PixelCount'] > 10]
state_sizes = df_nonZero['state'].value_counts()
samples_per_state = np.ceil(state_sizes / state_sizes.sum() * Total_number_of_samples).astype(int)

# Create percentiles at 10% intervals
percentiles = np.arange(0, 101, 10)  # 0, 10, 20, ..., 100

# To store sampled rows
samples = []
colkeys = ['row_start', 'row_end', 'col_start', 'col_end']

for stati in df_nonZero['state'].unique():

    samples_per_bin = np.ceil(samples_per_state[stati]/10).astype(int)
    
    for band in df_nonZero['band'].unique():
     
        group = df_nonZero[df_nonZero['band'] == band]
        # Compute percentile thresholds for this band's PixelCount
        thresholds = np.percentile(group['PixelCount'], percentiles)
        
        # Iterate over percentile *ranges* (0–25%, 25–50%, etc.)
        for k in range(len(thresholds) - 1):
            lower, upper = thresholds[k], thresholds[k + 1]
            subset = group[(group['PixelCount'] >= lower) & (group['PixelCount'] < upper)]

            if subset.empty:
                continue
            
            # Draw random samples from this range
            n_take = min(samples_per_bin, len(subset))
            sampled_rows = subset.sample(n_take, random_state=42)
            
            samples.append(sampled_rows) 
            # Define the keys that define a unique spatial block
            

            # Drop from df_nonZero all rows whose block exists in sampled_rows
            df_nonZero = (
            df_nonZero
            .merge(sampled_rows[colkeys].drop_duplicates(), on=colkeys, how='left', indicator=True)
            .query('_merge == "left_only"')
            .drop(columns=['_merge'])
            )

# 6. Combine all sampled data and add 5% of complete 0 raster chips
sample_df = pd.concat(samples).reset_index(drop=True)
samp0 = int(len(sample_df) / 20)
# filter the original df for a state and rol col combi, where field and border are both 0
colkeys_state = colkeys + ['state']
df_all0 = df.groupby(colkeys_state).filter(lambda g: (g[g['band'].isin(['Field', 'Border'])]['PixelCount'] == 0).all())
df_all0_unique = df_all0[df_all0['band'] == 'Field'] # drop duplicates for state row col combi


# get distribution of empty chips per state
state_sizes0 = df_all0_unique['state'].value_counts()
samples0_per_state = np.ceil(state_sizes0 / state_sizes0.sum() * samp0).astype(int)

# draw samples accordingly per state
samples0 = []
for statX in df_all0_unique['state'].unique():
    group = df_all0_unique[df_all0_unique['state'] == statX]
    sampled_rows0 = group.sample(samples0_per_state[statX], random_state=42)
    samples0.append(sampled_rows0)

# merge samples of empty chips with those that contain data
sample_block = pd.concat([sample_df, pd.concat(samples0)])
sample_block = sample_block[sample_block['band'] == 'Field']
sample_block_sorted = sample_block.sort_values(by='state')


# label and img chips creation

for state, state_folder, year in zip(states, state_folders, years):

    # get geo information
    print(state)
    fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
    borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines_touch_true.tif'
    dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
    fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'
 
    arr_stack = stackReader(stack_tifs([fields_path, borders_path, dist_path, fieldsID_path], d_type=gdal.GDT_Float32))
    row_col_ind =get_row_col_indices(chip_size, 0, arr_stack.shape[0], arr_stack.shape[1])
    row_start = row_col_ind[0]
    row_end   = row_col_ind[1]
    col_start = row_col_ind[2]
    col_end   = row_col_ind[3]

    arr_stack[:,:,3][arr_stack[:,:,3] == 0] = -10000 # makes the background of uniqueIDs same as in ai4bound
    vrt_ds = gdal.Open(f'{temp_folder}{state}/{state}_{year}_Fields.tif')
    vrt_arr = vrt_ds.GetRasterBand(1).ReadAsArray()
    geoTF = vrt_ds.GetGeoTransform()
    prj = vrt_ds.GetProjection()


    force_arr = loadVRTintoNumpyAI4(f"{aux_vrt_path}{state_folder}/{year}/{dirfinder(f'{aux_vrt_path}{state_folder}/{year}/')[0]}",
                                    applyNormalizer=False)

    for idx, row in sample_block_sorted.iterrows():#sample_df[sample_df['band'] == band_lkp[band]].iterrows():
        
        if row['state'] != state:
            continue
        # export labels as tif chips
        out_ds = gtiff_driver.Create(
            path_safe(
                f"{out_folder}label/{state_folder}/{year}/{state_folder}_{year}_rowstart_{row['row_start']:04d}_colstart_{row['col_start']:04d}.tif"),
                                        int(chip_size), int(chip_size), 4, gdal.GDT_Float32)
        # change the Geotransform for each chip
        geotf = list(geoTF)
        # get column and rows from filenames
        geotf[0] = geotf[0] + geotf[1] * row['col_start']
        geotf[3] = geotf[3] + geotf[5] * row['row_start']
        #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
        
        out_ds.SetGeoTransform(tuple(geotf))
        out_ds.SetProjection(prj)

        for bandx in range(arr_stack.shape[-1]):
            out_ds.GetRasterBand(bandx + 1).WriteArray(arr_stack[row['row_start']:row['row_end'], row['col_start']:row['col_end'],bandx])
        del out_ds

        # make Sentinel-2 chips as .nc
        force_sub = force_arr[:,:,row['row_start']:row['row_end'], row['col_start']:row['col_end']].copy()
        # set values below 1 to -9999
        force_sub[force_sub < 1] = -9999

        time = np.arange(6)
        x = geotf[0] + np.arange(chip_size) * geotf[1]
        y = geotf[3] + np.arange(chip_size) * geotf[5]
 
        data_vars = {
            nc_band_names[i]: (("time", "y", "x"), force_sub[i])
            for i in range(len(nc_band_names))
        }

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": time, "x": x, "y": y}
        )

        srs = osr.SpatialReference()
        srs.ImportFromWkt(vrt_ds.GetProjection())
        epsg_code = srs.GetAttrValue('AUTHORITY', 1)

        ds.rio.write_crs(f'EPSG:{epsg_code}', inplace=True)

        encoding = {band: {"zlib": True, "_FillValue": -9999} for band in nc_band_names}
        ds.to_netcdf(
            path_safe(f"{out_folder}img/{state_folder}/{year}/{state_folder}_{year}_rowstart_{row['row_start']:04d}_colstart_{row['col_start']:04d}.nc"),
            encoding=encoding)


################################################ create rocksdb from images (nc) and labels (tif)

img_size = 128
## create metadata file
metadata = {
    'inputs': {
        'inputs_shape': (4, 6, img_size, img_size),  # bands, time, rows, cols
        'inputs_dtype': np.float32     
    },
    'labels': {
        'labels_shape': (4, img_size, img_size),          
        'labels_dtype': np.float32
    }
}


## define function to load imgs and labs
def names2array_function(names):

    variables2use=['B2','B3','B4','B8']#,'NDVI']

    image_path, label_path = names
    # load image
    img = xr.open_dataset(image_path)
    image = np.concatenate([img[var].values[None] for var in variables2use],0)
    
    # load label
    ds = xr.open_dataset(label_path)
    label = np.asarray(ds['band_data'].values)
    label[np.isnan(label)] = 0
    return [image, label] 

## create list of images and labels

# database for AI4Boundaries
imgs = getFilelist(f'{out_folder}img/', '.nc', deep=True)
labs = getFilelist(f'{out_folder}label/', '.tif', deep=True)
imgs.sort()
labs.sort()

lab_dict = {lab.split('/')[-1].split('.')[0]: lab for lab in labs}

img_lab_paths = [(img, lab_dict[img.split('/')[-1].split('.')[0]]) 
                 for img in imgs 
                 if img.split('/')[-1].split('.')[0] in lab_dict
]

## create db

output_dir = f'/{origin}fields/output/rocks_db/Fine_tuner.db'
os.makedirs(output_dir, exist_ok=True)

rasters2rocks = Rasters2RocksDB(
    lstOfTuplesNames=img_lab_paths,            
    names2raster_function=names2array_function,  
    metadata=metadata,                       
    flname_prefix_save=output_dir,           
    batch_size=2,
    transformT=TrainingTransform_for_rocks_Train(),
    transformV=TrainingTransform_for_rocks_Valid(),
    stride_divisor=2,                    
    train_split=0.9,                         
    Filter=img_size,
    split_type='sequential'                  
)

rasters2rocks.create_dataset()


clear_directory(temp_folder)
 