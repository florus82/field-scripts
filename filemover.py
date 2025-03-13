import os
import shutil

def getFilelist(originpath, ftyp):
    files = os.listdir(originpath)
    out   = []
    for i in files:
        if i.split('.')[-1] in ftyp:
            if originpath.endswith('/'):
                out.append(originpath + i)
            else:
                out.append(originpath + '/' + i)
        # else:
        #     print("non-matching file - {} - found".format(i.split('.')[-1]))
    return out

path1 = '/home/ai4boundaries/sentinel2/images/LU'
path1TO = '/home/ai4boundaries/sentinel2/images/predict/'
imgs = getFilelist(path1, '.nc')
print(len(imgs))
imgs.sort()

for i in range(15):
    shutil.copyfile(imgs[i], path1TO + imgs[i].split('/')[-1])
    

path2 = '/home/ai4boundaries/sentinel2/masks/LU'
path2TO = '/home/ai4boundaries/sentinel2/masks/predict/'
imgs = getFilelist(path2, '.tif')
imgs.sort()
print(len(imgs))
for i in range(15):
    shutil.copyfile(imgs[i], path2TO + imgs[i].split('/')[-1])