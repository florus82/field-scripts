# import packages 
import sys
dataFolder = '/data/fields/'
sys.path.append('/home/potzschf/repos/')
from helperToolz.helper import *
import xarray as xr 
from helperToolz.feevos.rocksdbutils_copy import *
from math import ceil as mceil


img_size = 128
## create metadata file
metadata = {
    'inputs': {
        'inputs_shape': (4, 6, img_size, img_size),  
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

def names2array_function16(names):

    variables2use=['B2','B3','B4','B8']#,'NDVI']

    image_path, label_path = names
    # load image
    img = xr.open_dataset(image_path)
    image = np.concatenate([img[var].values[None] for var in variables2use],0)
    
    # load label
    ds = xr.open_dataset(label_path)
    label = np.asarray(ds['band_data'].values)
    label[np.isnan(label)] = 0
    return [image.astype(np.float16), label.astype(np.float16)]  

## create list of images and labels

# database for AI4Boundaries
folders = ['LU', 'AT', 'ES', 'FR', 'NL', 'SE', 'SI']
imgs = [getFilelist(dataFolder + 'ai4boundaries/sentinel2/images/' + folder, '.nc') for folder in folders]
imgs = [img for list in imgs for img in list]
labs = [getFilelist(dataFolder + 'ai4boundaries/sentinel2/masks/' + folder, '.tif') for folder in folders]
labs = [lab for list in labs for lab in list]
print(len(imgs), len(labs))
imgs.sort()
labs.sort()

## exclude completely empty images
exclude = False
if exclude == True:
    aa = checkemptyNC(labs)
    # exclude images with empty labels and their labels
    labs = [lab for i, lab in enumerate(labs) if i not in aa]
    imgs = [img for i, img in enumerate(imgs) if i not in aa]

print(len(imgs), len(labs))


img_lab_paths = []
for i in range(len(imgs)):
    img_lab_paths.append((imgs[i], labs[i]))


## create db

output_dir = f'{dataFolder}output/rocks_db/AI4_RGB_NDVI_exclude_{exclude}.db'
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