import cv2
import glob

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

######################
# path
path = '/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/human/'
check = True
# create dataset in memory using geotransform specified in ref_pth
def create_mem_ds(ref_pth, n_bands):
    drvMemR = gdal.GetDriverByName('MEM')
    ds = gdal.Open(ref_pth)
    mem_ds = drvMemR.Create('', ds.RasterXSize, ds.RasterYSize, n_bands, gdal.GDT_Float32)
    mem_ds.SetGeoTransform(ds.GetGeoTransform())
    mem_ds.SetProjection(ds.GetProjection())
    return mem_ds

# create copy
def copy_mem_ds(pth, mem_ds):
    copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0, options=['COMPRESS=LZW'])
    copy_ds = None

######################
# multi-taks labels from boundaries
def get_boundary(label, kernel_size = (2,2)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_distance(label):
    tlabel = label.astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)

    # get unique objects
    output = cv2.connectedComponentsWithStats(crop, 4, cv2.CV_32S)
    num_objects = output[0]
    labels = output[1]

    # min/max normalize dist for each object
    for l in range(num_objects):
        dist[labels==l] = (dist[labels==l]) / (dist[labels==l].max())

    return dist

def get_crop(image, kernel_size = (3,3)):

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # invert
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # erode & dilate
    img_erosion = cv2.erode(im_floodfill, kernel, iterations=1)
    return cv2.dilate(img_erosion, kernel, iterations=1) - 254

# visual check
if check == True:
    print(path+'/*.tif')
    files = glob.glob(path+'/*.tif')

    edge = cv2.imread(files[1], cv2.IMREAD_GRAYSCALE);
    crop = get_crop(edge)
    dist = get_distance(crop)
    edge = cv2.dilate(edge, np.ones((2,2), np.uint8), 1)
    label = np.stack([crop, edge, dist])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.moveaxis(label, 0, 2))
    plt.tight_layout()


# run across all label tiles
files = glob.glob(f'{path}/*.tif')
# run across label files
for file in files:
    # read image
    edge = cv2.imread(file, cv2.IMREAD_GRAYSCALE);
    crop = get_crop(edge)
    dist = get_distance(crop)
    dist = dist * crop
    edge = cv2.dilate(edge, np.ones((2,2), np.uint8), 1) # dilating mask to better match geometry of labels

    label = np.stack([crop, edge, dist])

    mem_ds = create_mem_ds(file, 3)

    # write outputs to bands
    for b in range(3):
        mem_ds.GetRasterBand(b+1).WriteArray(label[b,:,:])

    # create physical copy of ds
    out = file[:-4]+'_mtsk.tif'
    print(out)
    copy_mem_ds(out, mem_ds)
