import sys
import os
# sys.path.append('/home/potzschf/repos/')
# from helperToolz.helpsters import *

sys.path.append('/media/')
from helperToolz.helpsters import *

####################################################### Prepare

origin = 'Docker'

if origin == 'Docker':
    prefix = '/workspace/'
else:
    prefix = '/data/Aldhani/eoagritwin/'

print(f'working in {origin}')
# make vrts from force outputs for easier processing
year = 2023
model_name = 'model_state_AI4_RGB_exclude_False_47'
predict_master_folder = path_safe(f"{prefix}fields/output/predictions/FORCE/BRANDENBURG/{model_name.split('_state_')[-1]}/{year}/")

reduced_files = reduce_forceTSA_output_to_validmonths(f'{prefix}force/output/BRANDENBURG/{year}/', 3, 8)
ordered_files = force_order_Colors_for_VRT(reduced_files, ['BLU', 'GRN', 'RED', 'BNR'], [f'MONTH-{d:02d}' for d in range(3,9,1)])

vrt_out = f'{prefix}fields/Auxiliary/vrt/{year}/'

if os.path.isdir(vrt_out):
    if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
        print('VRT seems to be already computated, probably to create masks based on IACS')
    else:
        force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])
else:
    os.makedirs(vrt_out)
    force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])


# # load vrts into npdstack
print('Load vrt into numpy for prediction')
dat = loadVRTintoNumpyAI4(f'{prefix}/fields/Auxiliary/vrt/{year}/{getFORCExyRangeName(get_forcetiles_range(reduced_files))}')

# print(f'loaded np array has the shape:{dat.shape}')

# set tiling scheme and chip size on which prediction will be undertaken
chipsize = 128*2 # 5 is the maximum with GPU in basement
overlap  = 20

row_col_ind = get_row_col_indices(chipsize, overlap, dat.shape[2:][0], dat.shape[2:][1])

# ####################################################### Predict
# print('start prediction')
# predicted_chips_list = predict_on_GPU(f'{prefix}fields/output/models/{model_name}.pth', row_col_ind, dat, 
#                                       temp_path=f'{predict_master_folder}temp/')# model_state_All_but_LU_transformed_42

####################################################### Postprocess
print('start exporting')

    # export the predicted chips (masked and not masked)

with open(f'{predict_master_folder}temp/preds.pkl', 'rb') as f:
    predicted_chips_list = pickle.load(f)
export_GPU_predictions(predicted_chips_list, 
                    [crop_mask_file for crop_mask_file in \
                     getFilelist(f'{prefix}fields/IACS/4_Crop_mask/{year}/', '.tif') if 'prediction_extent' not in crop_mask_file], #f'{prefix}fields/IACS/4_Crop_mask/{year}/GSA-DE_BRB-{year}_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                    f'{prefix}fields/Auxiliary/vrt/{year}/{getFORCExyRangeName(get_forcetiles_range(reduced_files))}',
                    row_col_ind, 
                    path_safe(f'{predict_master_folder}chips_folder/'),
                    chipsize, overlap)


# # make vrt of predicted image chips
for chip in dirfinder(f'{predict_master_folder}chips_folder/'):
    predicted_chips_to_vrt(f'{predict_master_folder}chips_folder/', chip,  chipsize, overlap,
                        path_safe(f'{predict_master_folder}vrt/'), pyramids=True)


# adapt for case that extent_prection is already in there

# make a subset of the reference mask (extent FORCE output) to the extent of the prediction
for crop_mask_file in getFilelist(f'{prefix}fields/IACS/4_Crop_mask/{year}/', '.tif'): 
    subset_mask_to_prediction_extent(crop_mask_file,
    getFilelist(f'{predict_master_folder}vrt/', '.vrt')[0])
