import os
from osgeo import gdal, ogr, osr
import glob

img_path = '/data/Aldhani/cv_fields/images/descartes_tiles/mozambique'
files = glob.glob(f'{img_path}/*.tif')
print(len(files))


##### open field vector file
field = ogr.Open(f'/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/human/moz_hl_lines.gpkg')
field_lyr = field.GetLayer(0)

for f in files:

    output = f'/data/Aldhani/cv_fields/labels/descartes_tiles/mozambique/human/fields_{os.path.basename(f)[:-4]}.tif'
    print(output)
    if not os.path.exists(output):

        ds = gdal.Open(f)
        target_ds = gdal.GetDriverByName('GTiff').Create(output, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())

        gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = ["ALL_TOUCHED=TRUE"])
        target_ds = None
