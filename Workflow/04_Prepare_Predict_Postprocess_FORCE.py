import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *


####################################################### Prepare

# make vrts from force outputs for easier processing
reduced_files = reduce_force_to_validmonths('/data/force/output/BRANDENBURG/', 3, 8)
ordered_files = force_order_BGRBNR(reduced_files)
force_to_vrt(reduced_files, ordered_files, '/data/fields/Auxiliary/vrt', True)

# load vrts into npdstack
dat = loadVRTintoNumpyAI4(f'/data/fields/Auxiliary/vrt/{get_forcetiles_range(reduced_files)}')

# set tiling scheme and chip size on which prediction will be undertaken
chipsize = 128*1 # 5 is the maximum with GPU in basement
overlap  = 20

row_col_ind = get_row_col_indices(chipsize, overlap, dat.shape[2:][0], dat.shape[2:][1])

####################################################### Predict

predicted_chips_list = predict_on_GPU('/data/fields/output/model_state_All_but_LU_transformed_42.pth', row_col_ind, dat)

####################################################### Postprocess

# export the predicted chips (masked and not masked)
export_GPU_predictions(predicted_chips_list, 
                       '/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                       f'/data/fields/Auxiliary/vrt/{get_forcetiles_range(reduced_files)}',row_col_ind, 
                       '/data/fields/output/predictions/FORCE/BRANDENBURG/')


# make vrt of predicted image chips
for chip in ['chips/', 'masked_chips/']:
    predicted_chips_to_vrt(f'/data/fields/output/predictions/FORCE/BRANDENBURG/{chip}', chipsize, overlap,
                           '/data/fields/output/predictions/FORCE/BRANDENBURG/vrt/', pyramids=True)

# make a subset of the reference mask (extent FORCE output) to the extent of the prediction

subset_mask_to_prediction_extent('/data/fields/IACS/4_Crop_mask/GSA-DE_BRB-2019_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                                 f'/data/fields/output/predictions/FORCE/BRANDENBURG/vrt/{chipsize}_{overlap}_{chip}.vrt')