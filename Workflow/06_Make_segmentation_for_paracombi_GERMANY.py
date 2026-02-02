from sklearn.linear_model import LinearRegression
from skimage import measure
from joblib import Parallel, delayed
import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *


#########################  parameter settings
# load the prediction and labels
year = 2019
model_name = 'AI4_RGB_exclude_True_38'
ncores = 14

t_ext = 0.8
t_bound = 0.2

storPath = path_safe(f"/data/Aldhani/eoagritwin/fields/segmented/Germany/{model_name}/{year}/ext_{''.join(str(t_ext).split('.'))}_bound_{''.join(str(t_bound).split('.'))}/")
folder_path = path_safe(f"{storPath}/intermediates/")

# load precition
pred_ds = gdal.Open('/data/Aldhani/eoagritwin/fields/output/predictions/FORCE/GERMANY/AI4_RGB_exclude_True_38/2019/vrt/unmasked_chips_256_20.vrt')
rows, cols = pred_ds.RasterYSize, pred_ds.RasterXSize

# set the number by which rows and cols will be divided --> determines the number of tiles // also set border limit (dont sample fields too close to tile borders) and sample size
slicer = 25

row_start = [i for i in range(0, rows, math.floor(rows/slicer))]
row_end = [i for i in range (math.floor(rows/slicer), rows, math.floor(rows/slicer))]
row_start = row_start[:len(row_end)] 

col_start = [i for i in range(0, cols, math.floor(cols/slicer))]
col_end = [i for i in range (math.floor(cols/slicer), cols, math.floor(cols/slicer))]
col_start = col_start[:len(col_end)] 

extent_pred_list = []
boundary_pred_list = []
result_dir_list = []
row_col_start = []

processed_chips = [file.split('segmented_')[-1].split('.')[0] for file in getFilelist(folder_path, '.tif')]

# read in vrt in tiles
for i in range(len(row_end)):
    for j in range(len(col_end)):
        if str(row_start[i]) + '_' + str(col_start[j]) not in processed_chips:
            ######### fill the lists with tiled data

            #subset the prediction of fields read-in
            extent_pred = pred_ds.GetRasterBand(1).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i]) # goes into InstSegm --> image of crop probability 
            extent_pred_list.append(extent_pred)
            
            # load predicted boundary prob subset // goes into InstSegm --> image of boundary probability
            boundary_pred_list.append(pred_ds.GetRasterBand(2).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) 
            
            # output folder
            result_dir_list.append(folder_path)
            row_col_start.append(str(row_start[i]) + '_' + str(col_start[j]))
 

jobs = [[t_ext, t_bound, extent_pred_list[i], boundary_pred_list[i], row_col_start[i], result_dir_list[i],\
            pred_ds.GetGeoTransform(), pred_ds.GetProjection()]  for i in range(len(result_dir_list))]
# tile_list[i], 
print(f'\n{len(jobs)} tiles will be processed\n')

del row_col_start, extent_pred_list, boundary_pred_list, result_dir_list

# overlords_jobs.append(jobs)


if __name__ == '__main__':
    starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("Starting process, time:" + starttime)
    print("")

    Parallel(n_jobs=ncores)(delayed(apply_segm_param)(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7]) for i in jobs)   # for jobs in overlords_jobs 

    print("")
    endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("start : " + starttime)
    print("end: " + endtime)
    print("")
