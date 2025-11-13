import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
import shutil
from skimage import measure

workhorse = True

if workhorse:
    origin = 'Aldhani/eoagritwin/'
else:
    origin = ''

# transform unique-pairs array to dict
def unique_dict(unique_pairs_array):
    valid_dict = {}

    for key, value in unique_pairs_array:
        if key in valid_dict:
            valid_dict[key].append(value)
        else:
            valid_dict[key] = [value]

    return valid_dict

def make2000000(x):
    s = str(x)
    if len(s) == 7:
        return int('2' + s[1:] )
    else:
        return int('2' + s[2:] )
    
# load shapefiles and 
state = 'Brandenburg'

ger = gpd.read_file(f'/data/{origin}misc/gadm41_DEU_shp/gadm41_DEU_1.shp')
aoi = ger[ger['NAME_1'] == state]

paths = ['/data/Aldhani/eoagritwin/fields/segmented/Brandenburg/AI4_RGB_exclude_True_38/2023/masked_lines_touch_true_crop_touch_false_linecrop_text0.1_tbound0.7/',
         '/data/Aldhani/eoagritwin/fields/segmented/Brandenburg/AI4_RGB_exclude_True_38/2023/unmasked_text0.1_tbound0.65/']

combis = ['0.1_0.7', '0.1_0.65']

for path, combi in zip(paths, combis):

    if path.endswith('/'):
        pass
    else:
        path = f'{path}/'

    # get tile files for prediction
    files = getFilelist(path, '.tif', deep=True)
    files_sub = [file for file in files if combi in file and '_pred_' in file]
    export_path = path_safe(f'{path}prediction/{path.split('/')[-2]}.vrt')
    # store the prediction as .tif
    vrt = gdal.BuildVRT(path_safe(export_path), files_sub)
    vrt = None
    vrt_ds = gdal.Open(export_path)
    gt = vrt_ds.GetGeoTransform()
    proj = vrt_ds.GetProjection()
    xsize = vrt_ds.RasterXSize
    ysize = vrt_ds.RasterYSize
    
    vrt_arr = vrt_ds.GetRasterBand(1).ReadAsArray()
    # mask it before export

    # Reproject AOI 
    aoi = aoi.to_crs(proj)

    # Convert AOI GeoDataFrame to OGR layer in memory
    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("memAoi")
    mem_layer = mem_ds.CreateLayer("aoi", srs=ogr.osr.SpatialReference(wkt=proj))
    mem_layer_defn = mem_layer.GetLayerDefn()
    for geom in aoi.geometry:
        feat = ogr.Feature(mem_layer_defn)
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        mem_layer.CreateFeature(feat)
        feat = None

    # Create in-memory raster aligned with the VRT
    mem_raster = gdal.GetDriverByName("MEM").Create("", xsize, ysize, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(gt)
    mem_raster.SetProjection(proj)
    band = mem_raster.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    # Rasterize AOI (burn 1 where AOI exists)
    gdal.RasterizeLayer(mem_raster, [1], mem_layer, burn_values=[1])

    # Read mask to NumPy array (1 = inside AOI, 0 = outside)
    mask_arr = mem_raster.ReadAsArray().astype(np.uint8)

    npTOdisk(vrt_arr * mask_arr, export_path, f'{path}prediction/{path.split('/')[-2]}.tif', bands=1, noData=0)
  





    # make a physical copy of these files
    outpath = path_safe(f'{path}quick_n_dirty/')
    if len(getFilelist(outpath, '.tif')) != 0:
        for file in getFilelist(outpath, '.tif'):
            os.remove(file)
    for file in files_sub:
        shutil.copy2(file, outpath + file.split('intermediates/')[-1])

    files_sub = getFilelist(outpath, '.tif')

    # loop over files to get row and col cuts
    rows, cols = [], []
    for file in files_sub:
        rows.append(int(file.split('_instance_pred_')[-1].split('_')[0]))
        cols.append(int(file.split('_instance_pred_')[-1].split('_')[-1].split('.')[0]))
    rows.sort()
    cols.sort()
    rows = list(set(rows))
    cols = list(set(cols))
    rows.sort()
    cols.sort()


    ##### iterate over possible row/col connections and search for fields in neighbouring tiles
    # start at top-left-corner
    for row in rows:
        for col in cols:

            # row = rows[0]
            # col = cols[5]
            tile1 = [sub for sub in files_sub if f'_{row}_{col}.tif' in sub]
            if len(tile1) == 0:
                continue
            else: 
                ### check for neighbouring tiles and store result in list
                #print(f'_{row}_{col}.tif')
                # upper  (row -1)
                if rows.index(row)!=0:
                    upper = [sub for sub in files_sub if f'_{rows[rows.index(row)-1]}_{col}.tif' in sub]
                else:
                    upper = []

                # upper right (row -1 and col +1)
                if rows.index(row)!=0 and cols.index(col) < len(cols) - 1:
                    upper_right = [sub for sub in files_sub if f'_{rows[rows.index(row)-1]}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    upper_right = []

                # right (col +1)
                if cols.index(col) < len(cols) - 1:
                    right = [sub for sub in files_sub if f'_{row}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    right = []

                # lower right (row +1 and col +1)
                if rows.index(row) < len(rows) - 1 and cols.index(col) < len(cols) - 1:
                    lower_right = [sub for sub in files_sub if f'_{rows[rows.index(row)+1]}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    lower_right = []

                # bottom (row +1)
                if rows.index(row) < len(rows) - 1:
                    bottom = [sub for sub in files_sub if f'_{rows[rows.index(row)+1]}_{col}.tif' in sub]
                else:
                    bottom = []

                if any(len(lst) > 0 for lst in [upper, upper_right, right, lower_right, bottom]):
                    # load starting tile
                    ds = gdal.Open(tile1[0])
                    dat = ds.GetRasterBand(1).ReadAsArray()
                    
                    ### check the other tiles
                    # upper
                    if len(upper) == 1:
                        ds = gdal.Open(upper[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[0,:], neighbour[-1,:]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = 2000000 + v
                            for p in p_list:
                                neighbour[neighbour == p] = 2000000 + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, upper[0], upper[0], noData=0)

                    # upper right
                    if len(upper_right) == 1:
                        ds = gdal.Open(upper_right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[0,-1], neighbour[-1,0])), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all()]
                        if len(valid_pairs) > 1:
                            for v, p in valid_pairs:
                                dat[dat == v] = 2000000 + v
                                neighbour[neighbour == p] = 2000000 + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, upper_right[0], upper_right[0], noData=0)

                    # right
                    if len(right) == 1:
                        ds = gdal.Open(right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[:,-1], neighbour[:,0]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = 2000000 + v
                            for p in p_list:
                                neighbour[neighbour == p] = 2000000 + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, right[0], right[0], noData=0)

                    # lower right
                    if len(lower_right) == 1:
                        ds = gdal.Open(lower_right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[-1,-1], neighbour[0,0])), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all()]
                        if len(valid_pairs) > 1:
                            for v, p in valid_pairs:
                                dat[dat == v] = 2000000 + v
                                neighbour[neighbour == p] = 2000000 + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, lower_right[0], lower_right[0], noData=0)

                    # bottom
                    if len(bottom) == 1:
                        ds = gdal.Open(bottom[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()

                        unique_pairs = np.unique(np.stack((dat[-1,:], neighbour[0,:]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = 2000000 + v
                            for p in p_list:
                                neighbour[neighbour == p] = 2000000 + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, bottom[0], bottom[0], noData=0)

                    makeTif_np_to_matching_tif(dat, tile1[0], tile1[0], noData=0)
                else:
                    continue

    

    # loop over every tile again and clean up mess (e.g. 20000059, 40000059, 60000059)
    vec_func = np.vectorize(make2000000,otypes=[int])

    for file in files_sub:
        ds = gdal.Open(file)
        arr = ds.GetRasterBand(1).ReadAsArray()
        mask = arr > 3000000
        arr[mask] = vec_func(arr[mask])
        makeTif_np_to_matching_tif(arr, file, file, noData=0)

    # make a vrt and redo rasterIDs
    gdal.BuildVRT(f'{outpath}quick_n_dirty.vrt', files_sub)
    vrt = None

    ds = gdal.Open(f'{outpath}quick_n_dirty.vrt')
    block = ds.GetRasterBand(1).ReadAsArray()
    relabelled = measure.label(block, background=0, connectivity=1)

    #### polygonize
    # create an in-memory raster-band with geoinfo of the relabelled array
    rows, cols = relabelled.shape
    driver = gdal.GetDriverByName('MEM')  # In-memory raster
    raster_ds = driver.Create('', cols, rows, 1, gdal.GDT_Int32)
    raster_ds.GetRasterBand(1).WriteArray(relabelled)
    raster_ds.SetGeoTransform(ds.GetGeoTransform())
    raster_ds.SetProjection(ds.GetProjection())
    src_band = raster_ds.GetRasterBand(1)

    # create a mask for the background (otherwise it would get a polygon as well)
    mask_array = (relabelled != 0).astype(np.uint8)
    mask_ds = driver.Create('', cols, rows, 1, gdal.GDT_Int32)
    mask_ds.GetRasterBand(1).WriteArray(mask_array)
    mask_band = mask_ds.GetRasterBand(1)

    # create output for shp
    driver = ogr.GetDriverByName('ESRI Shapefile')  # or 'GeoJSON', 'GPKG', etc.
    out_ds = driver.CreateDataSource(f'{outpath}Fields_polygons.shp')  # Output vector file
    out_layer = out_ds.CreateLayer('polygons', getSpatRefRas(ds), geom_type=ogr.wkbPolygon)
    field_defn = ogr.FieldDefn('FieldID', ogr.OFTInteger)
    out_layer.CreateField(field_defn)

    # polygonize
    gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)
    del out_ds