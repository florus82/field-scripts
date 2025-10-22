import sys
import os
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *

# sys.path.append('/media/helperToolz')

# from helpsters import *

####################################################### Prepare

origin = 'Docker'

if origin == 'Docker':
    prefix = '/home/'
else:
    prefix = '/data/Aldhani/eoagritwin/'

# make vrts from force outputs for easier processing
year = 2023
reduced_files = reduce_forceTSA_output_to_validmonths(f'{prefix}force/output/BRANDENBURG/{year}/', 3, 8)
ordered_files = force_order_Colors_for_VRT(reduced_files, ['BLU', 'GRN', 'RED', 'BNR'], [f'MONTH-{d:02d}' for d in range(3,9,1)])

vrt_out = f'{prefix}fields/Auxiliary/vrt/{year}/'
if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
    print('VRT seems to be already computated, probably to create masks based on IACS')
else:
    force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])

# # load vrts into npdstack
print('Load vrt into numpy for prediction')
dat = loadVRTintoNumpyAI4(f'{prefix}/fields/Auxiliary/vrt/{year}/{getFORCExyRangeName(get_forcetiles_range(reduced_files))}')

# set tiling scheme and chip size on which prediction will be undertaken
chipsize = 128*2 # 5 is the maximum with GPU in basement
overlap  = 20

row_col_ind = get_row_col_indices(chipsize, overlap, dat.shape[2:][0], dat.shape[2:][1])

####################################################### Predict
print('start prediction')
predicted_chips_list = predict_on_GPU(f'{prefix}fields/output/models/model_state_All_but_LU_transformed_42.pth', row_col_ind, dat)

####################################################### Postprocess
print('start exporting')
# export the predicted chips (masked and not masked)
export_GPU_predictions(predicted_chips_list, 
                       f'{prefix}fields/IACS/4_Crop_mask/{year}/GSA-DE_BRB-{year}_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                       f'{prefix}fields/Auxiliary/vrt/{year}/{getFORCExyRangeName(get_forcetiles_range(reduced_files))}',
                       row_col_ind, 
                       f'{prefix}fields/output/predictions/FORCE/BRANDENBURG/{year}/',
                       chipsize, overlap)


# make vrt of predicted image chips
for chip in ['chips/', 'masked_chips/']:
    predicted_chips_to_vrt(f'{prefix}fields/output/predictions/FORCE/BRANDENBURG/{year}/{chip}', chipsize, overlap,
                           f'{prefix}fields/output/predictions/FORCE/BRANDENBURG/vrt/{year}/', pyramids=True)


# make a subset of the reference mask (extent FORCE output) to the extent of the prediction
subset_mask_to_prediction_extent(f'{prefix}fields/IACS/4_Crop_mask/{year}/GSA-DE_BRB-{year}_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                                 f'{prefix}fields/output/predictions/FORCE/BRANDENBURG/{year}/vrt/{chipsize}_{overlap}_chips.vrt')

subset_mask_to_prediction_extent(f'{prefix}fields/IACS/4_Crop_mask/{year}/GSA-DE_BRB-{year}_cropMask_lines_touch_true_lines_touch_true_linecrop.tif', 
                                 f'/data/Aldhani/eoagritwin/fields/output/predictions/FORCE/BRANDENBURG/{year}/vrt/{chipsize}_{overlap}_chips.vrt')