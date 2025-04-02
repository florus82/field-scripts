import numpy as np
import pandas as pd
from osgeo import gdal
import time
import math
from sklearn.linear_model import LinearRegression
import higra as hg
import scipy.ndimage as si
from skimage import measure
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
np.random.seed(42)

########### sub-functions

def TooCloseToBorder(numbered_array, border_limit):
    rows, cols = np.where(numbered_array==True)
    r,c = numbered_array.shape
    if any(value < border_limit for value in [np.min(rows), r - np.max(rows), np.min(cols), c - np.max(cols)]):
        return True
    
def InstSegm(extent, boundary, t_ext=0.4, t_bound=0.2):
    """
    INPUTS:
    extent : extent prediction
    boundary : boundary prediction
    t_ext : threshold for extent
    t_bound : threshold for boundary
    OUTPUT:
    instances
    """

    # Threshold extent mask
    ext_binary = np.uint8(extent >= t_ext)

    # Artificially create strong boundaries for
    # pixels with non-field labels
    input_hws = np.copy(boundary)
    input_hws[ext_binary == 0] = 1

    # Create the directed graph
    size = input_hws.shape[:2]
    graph = hg.get_8_adjacency_graph(size)
    edge_weights = hg.weight_graph(
        graph,
        input_hws,
        hg.WeightFunction.mean
    )

    tree, altitudes = hg.watershed_hierarchy_by_dynamics(
        graph,
        edge_weights
    )
    
    # Get individual fields
    # by cutting the graph using altitude
    instances = hg.labelisation_horizontal_cut_from_threshold(
        tree,
        altitudes,
        threshold=t_bound)
    
    instances[ext_binary == 0] = -1

    return instances

def get_IoUs(extent_true, extent_pred, boundary_pred, t_ext, 
             t_bound, plot=False, border_limit=10):

    # get predicted instance segmentation
    instances_pred = InstSegm(extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound)
    instances_pred = measure.label(instances_pred, background=-1) 
    
    # get instances from ground truth label
    # binary_true = extent_true > 0
    # instances_true = measure.label(binary_true, background=0, connectivity=1)
    instances_true = extent_true
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(instances_true)
        ax[1].imshow(instances_pred)
        plt.show()
    
    # loop through true fields
    field_values = np.unique(instances_true)
    
    best_IoUs = []
    field_IDs = []
    field_sizes = []
    centroid_rows = []
    centroid_cols = []
    centroid_IoUS = []

    for field_value in field_values:
        if field_value == 0:
            continue # move on to next value
    
        this_field = instances_true == field_value
        # check if field is close to border and throw away if too close
        if TooCloseToBorder(this_field, border_limit):
            continue

        # calculate centroid
        this_field_centroid = np.mean(np.column_stack(np.where(this_field)),axis=0).astype(int)
        
        # fill lists with info
        centroid_rows.append(this_field_centroid[0])
        centroid_cols.append(this_field_centroid[1])
        field_IDs.append(field_value)
        field_sizes.append(np.sum(this_field))
        
        # find predicted fields that intersect with true field
        intersecting_fields = this_field * instances_pred
        intersect_values = np.unique(intersecting_fields)
        intersect_fields = np.isin(instances_pred, intersect_values[1:])
        
        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].imshow(this_field)
            ax[1].imshow(intersect_fields)
            plt.show()
        
        # compute IoU for each intersecting field
        field_IoUs = []
        center_IoU = 0
        for intersect_value in intersect_values:
            if intersect_value == 0 or (len(np.unique(intersect_values)) == 1 and intersect_value == 0):
                continue # move on to next value
            
            pred_field = instances_pred == intersect_value
            union = this_field + pred_field > 0
            intersection = (this_field * pred_field) > 0
            IoU = np.sum(intersection) / np.sum(union)
            field_IoUs.append(IoU)
            # check for centroid condition
            if instances_pred[this_field_centroid[0], this_field_centroid[1]] == intersect_value:
                center_IoU = IoU
    
        # take maximum IoU - this is the IoU for this true field
        if len(field_IoUs) != 0:
            best_IoUs.append(np.max(field_IoUs))
            # fill centroid list
            centroid_IoUS.append(center_IoU)
        else:
            best_IoUs.append(0)
            # fill centroid list
            centroid_IoUS.append(0)
   
    return best_IoUs, centroid_IoUS, field_IDs, field_sizes

########## main-function

def get_IoUs_per_Tile(tile, extent_true, extent_pred, boundary_pred, result_dir, border_limit=10):
    print(f'Starting on tile {tile}')
    # make a dictionary for export
    k = ['tile','t_ext','t_bound', 'max_IoU', 'centroid_IoU', 'reference_field_IDs', 'reference_field_sizes'] #'medianIoU', 'meanIoU', 'IoU_50', 'IoU_80']
    v = [list() for i in range(len(k))]
    res = dict(zip(k, v))

    # set the parameter combinations and test combinations
    t_exts = [i/100 for i in range(10,55,5)] 
    t_bounds = [i/100 for i in range(10,55,5)]

    # loop over parameter combinations
    for t_ext in t_exts:
        for t_bound in t_bounds:
            #print('thresholds: ' + str(t_ext) + ', ' +str(t_bound))

            img_IoUs, centroid_IoUS, field_IDs, field_sizes = get_IoUs(extent_true, extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound,
                                border_limit=border_limit)
            
            for e, IoUs in enumerate(img_IoUs):
    
                res['tile'].append(tile)
                res['t_ext'].append(t_ext)
                res['t_bound'].append(t_bound)
                res['max_IoU'].append(IoUs)
                res['centroid_IoU'].append(centroid_IoUS[e])
                res['reference_field_IDs'].append(field_IDs[e])
                res['reference_field_sizes'].append(field_sizes[e])
    
    # export results
    df  = pd.DataFrame(data = res)
    df.to_csv(f'{result_dir}_{tile}_IoU_hyperparameter_tuning_full.csv', index=False)

    print(f'Finished tile {tile}')

######### prepare job-list


# create lists that will be passed on to the joblist
tile_list = []
extent_true_list = []
extent_pred_list = []
boundary_pred_list = []
result_dir_list = []

# load the predictions and labels
predictions =  '/data/fields/output/predictions/FORCE/BRANDENBURG/vrt/256_20_chipsvrt.vrt' # predictions straight from GPU 
reference =  '/data/fields/IACS/Auxiliary/GSA-DE_BRB-2019_All_agromask_linecrop_prediction_extent.tif' # mask from IACS
result_dir = '/data/fields/Auxiliary/grid_search/Brandenburg/' + predictions.split('/')[-1].split('.')[0] + '_' + reference.split('/')[-1].split('.')[0]


# tile predictions in prds --> total extent encompasses 90 Force Tiles (+ a few rows and cols that will be neglected as they are outside of study area)
pred_ds = gdal.Open(predictions)
rows, cols = pred_ds.RasterYSize, pred_ds.RasterXSize

# set the number by which rows and cols will be divided --> determines the number of tiles // also set border limit (dont sample fields too close to tile borders) and sample size
slicer = 10
border_limit = 10
sample_size  = 10000
row_start = [i for i in range(0, rows, math.floor(rows/slicer))]
row_end = [i for i in range (math.floor(rows/slicer), rows, math.floor(rows/slicer))]
row_start = row_start[:len(row_end)] 

col_start = [i for i in range(0, cols, math.floor(cols/slicer))]
col_end = [i for i in range (math.floor(cols/slicer), cols, math.floor(cols/slicer))]
col_start = col_start[:len(col_end)] 


# load IACS reference mask and label it 
ref_ds = gdal.Open(reference)
extent_true = ref_ds.GetRasterBand(1).ReadAsArray() 
binary_true = extent_true > 0
instances_true = measure.label(binary_true, background=0, connectivity=1)


# sample fields
# build a mask to exclude fields that are in border_limit to tile borders
power_mask = np.zeros(instances_true.shape)
for i in range(len(row_end)):
    for j in range(len(col_end)):
            power_mask[row_start[i]:row_start[i] + border_limit, :] = 1
            power_mask[:, col_start[j]:col_start[j] + border_limit] = 1
            power_mask[row_end[-1] - border_limit:power_mask.shape[0], :] = 1
            power_mask[:, col_end[-1] - border_limit:power_mask.shape[1]] = 1

# makeTif_np_to_matching_tif(power_mask, reference, result_dir, 'powermask.tif',0)
# get IDs from labelled reference
IDs_to_skip = np.unique(instances_true[power_mask==1])

# get distribution of field sizes after segmentation
unique_IDs, counts = np.unique(instances_true, return_counts=True)

# exlcude fields that are too close to tile borders
mask = ~np.isin(unique_IDs, IDs_to_skip)
unique_IDs = unique_IDs[mask]
counts = counts[mask]

# exlude 0 (background) and 1 (super-small fields) from sample
mask = (unique_IDs != 0) & (counts > 1)
unique_IDs = unique_IDs[mask]
counts = counts[mask]


# get deciles and draw equally from them
deciles = [perc for perc in range(10,100,10)]
deciles_values = np.percentile(counts, deciles)
decs = [0] + deciles_values.tolist() + [np.max(counts)]
bin_ids = []
for ind in range(len(decs) -1):
    # get the unique_IDS of those fields, whose count (size) is within bin
    bin_ids.append(np.random.choice(unique_IDs[(counts > decs[ind]) & (counts <= decs[ind + 1])], int(sample_size/10), replace=False))

mask = np.isin(instances_true, np.concatenate(bin_ids))
# set everything to 0 except samples
instances_true[~mask] = 0
# makeTif_np_to_matching_tif(instances_true, reference, result_dir, 'samples.tif',0)

# read in vrt in tiles
for i in range(len(row_end)):
    for j in range(len(col_end)):
        
        ######### fill the lists with tiled data

        
        #subset the prediction of fields read-in
        extent_pred = pred_ds.GetRasterBand(1).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i]) # goes into InstSegm --> image of crop probability
        # check if prediction subset of fields actually contains data
        if len(np.unique(extent_pred)) == 1:
            continue
        # check if tile contains a sample of reference/label data
        extent_true = instances_true[row_start[i]:row_end[i], col_start[j]:col_end[j]]
        if len(np.unique(extent_true)) == 1:
            continue
        
        extent_true_list.append(extent_true)
        extent_pred_list.append(extent_pred)

        # make identifier for tile for csv
        tile_list.append(f'{str(i)}_{str(j)}')
        # load predicted boundary prob subset // goes into InstSegm --> image of boundary probability
        boundary_pred_list.append(pred_ds.GetRasterBand(2).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) 
        # output folder
        result_dir_list.append(result_dir)

jobs = [[tile_list[i], extent_true_list[i], extent_pred_list[i], boundary_pred_list[i], result_dir_list[i], border_limit]  for i in range(len(result_dir_list))]


if __name__ == '__main__':
    starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("Starting process, time:" + starttime)
    print("")

    Parallel(n_jobs=30)(delayed(get_IoUs_per_Tile)(i[0], i[1], i[2], i[3], i[4], i[5]) for i in jobs)

    print("")
    endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("start : " + starttime)
    print("end: " + endtime)
    print("")