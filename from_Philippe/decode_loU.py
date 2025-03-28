import numpy as np
import pandas as pd
from osgeo import gdal
import os
from scipy.stats import linregress
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
import higra as hg
import scipy.ndimage as si
from skimage import measure
import glob

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

def get_IoUs(extent_true, extent_pred, boundary_pred, t_ext=0.4, 
             t_bound=0.2, plot=False):
    
    # get predicted instance segmentation
    instances_pred = InstSegm(extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound)
    instances_pred = measure.label(instances_pred, background=-1)
    
    # get instances from ground truth label
    binary_true = extent_true > 0
    instances_true = measure.label(binary_true, background=0, connectivity=1)
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(instances_true)
        ax[1].imshow(instances_pred)
        plt.show()
    
    # loop through true fields
    field_values = np.unique(instances_true)
    best_IoUs = []
    field_sizes = []
    
    for field_value in field_values:
        if field_value == 0:
            continue # move on to next value
            
        this_field = instances_true == field_value
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
        for intersect_value in intersect_values:
            if intersect_value == 0:
                continue # move on to next value
            pred_field = instances_pred == intersect_value
            union = this_field + pred_field > 0
            intersection = (this_field * pred_field) > 0
            IoU = np.sum(intersection) / np.sum(union)
            field_IoUs.append(IoU)
    
        # take maximum IoU - this is the IoU for this true field
        if len(field_IoUs) > 0:
            best_IoUs.append(np.max(field_IoUs))
        else:
            best_IoUs.append(0)
    
    return best_IoUs, field_sizes

def get_IoUs_scores(extent_true, extent_pred, boundary_pred, t_ext=0.4, t_bound=0.2, t_semc=0.8, plot=False):
    
    # get predicted instance segmentation
    instances_pred = InstSegm(extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound)
    instances_scor = InstScores(instances_pred, extent_pred)
    instances_pred[instances_scor<t_semc] = -1
    instances_pred = measure.label(instances_pred, background=-1)

    # get instances from ground truth label
    binary_true = extent_true > 0
    instances_true = measure.label(binary_true, background=0, connectivity=1)
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(instances_true)
        ax[1].imshow(instances_pred)
        plt.show()
    
    # loop through true fields
    field_values = np.unique(instances_true)
    best_IoUs = []
    field_sizes = []
    
    for field_value in field_values:
        if field_value == 0:
            continue # move on to next value
            
        this_field = instances_true == field_value
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
        for intersect_value in intersect_values:
            if intersect_value == 0:
                continue # move on to next value
            pred_field = instances_pred == intersect_value
            union = this_field + pred_field > 0
            intersection = (this_field * pred_field) > 0
            IoU = np.sum(intersection) / np.sum(union)
            field_IoUs.append(IoU)
    
        # take maximum IoU - this is the IoU for this true field
        if len(field_IoUs) > 0:
            best_IoUs.append(np.max(field_IoUs))
        else:
            best_IoUs.append(0)
    
    return best_IoUs, field_sizes

model = 'fractal-resunet_100_from-scratch_GE_rgb_nfilter-32_depth-6_bs-12_lr-0.001_trainval-70-30_norm-none_lossftnmt_masked_moz_tst_03'
model = 'airbus_france_india'
model = 'fractal-resunet_25_from-scratch_GE_rgb_nfilter-32_depth-6_bs-12_lr-0.0001_trainval-70-30_norm-none_lossftnmt_masked_moz_tst_04_i02_bf_005pos'

prds = sorted(glob.glob(f'/data/Aldhani/cv_fields/preds/descartes_tiles/mozambique/{model}/*_preds.tif'))
refs = sorted(glob.glob('/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/human/*mtsk.tif'))

# hyperparameter values
t_exts = np.linspace(0.0, 0.5, 6)
t_bounds = np.linspace(0.0, 0.5, 6)

print(model)
label_dir = '/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/human/'
results_dir = f'/data/Aldhani/cv_fields/preds/descartes_tiles/mozambique/{model}/'

mIoUs = []
mnIoUs = []
IoU_50s = []
IoU_80s = []
for t_ext in t_exts:
    for t_bound in t_bounds:
        print('thresholds: ' + str(t_ext) + ', ' +str(t_bound))
        IoUs = []

        for i in range(len(refs)):
            label_path = refs[i]
            if os.path.exists(label_path):
                reference = gdal.Open(refs[i]).ReadAsArray()
                extent_true = np.squeeze(reference[0])

                prediction = gdal.Open(prds[i]).ReadAsArray()
                extent_pred = np.squeeze(prediction[0])
                boundary_pred = np.squeeze(prediction[1])

                img_IoUs, _ = get_IoUs(extent_true, extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound)
                #img_IoUs, _ = get_IoUs_scores(extent_true, extent_pred, boundary_pred, t_ext=t_ext, t_bound=t_bound, t_semc=t_semc)
                IoUs = IoUs + img_IoUs

        mIoUs.append(np.median(IoUs))
        mnIoUs.append(np.mean(IoUs))
        IoU_50s.append(np.sum(np.array(IoUs) > 0.5) / len(IoUs))
        IoU_80s.append(np.sum(np.array(IoUs) > 0.8) / len(IoUs))

hp_df = pd.DataFrame({
    't_ext': np.repeat(t_exts, len(t_bounds)),
    't_bound': np.tile(t_bounds, len(t_exts)),
    'medianIoU': mIoUs,
    'meanIoU': mnIoUs,
    'IoU_50': IoU_50s,
    'IoU_80': IoU_80s
})
hp_df.to_csv(os.path.join(results_dir, 'IoU_hyperparameter_tuning_full.csv'), index=False)
print(hp_df.iloc[hp_df['meanIoU'].idxmax()])