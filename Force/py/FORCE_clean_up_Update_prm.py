import sys
sys.path.append('/home/potzschf/repos/')
from helperToolz.helpsters import *
import shutil


######################################################## year and paths
year = 2019
tile_path = f'/data/Aldhani/eoagritwin/force/output/Guzinski/{year}/tiles/'
prm_file = f'/data/Aldhani/eoagritwin/force/parameterfile/Guzinski/{year}_10_day_comp_for_magic.prm'
prm_file_clean = f'/data/Aldhani/eoagritwin/force/parameterfile/Guzinski/{year}_10_day_comp_for_magic_cleanUp.prm'
magic_path = '/data/Aldhani/eoagritwin/force/parameterfile/Guzinski/magic_subs'

number_of_expected_bands = 423

# check which tiles were already processed
folders = [os.path.join(tile_path, name) for name in os.listdir(tile_path) if os.path.isdir(os.path.join(tile_path, name))]
folders_to_kill = [folder for folder in folders if len(getFilelist(folder, '.tif', deep=True)) != number_of_expected_bands]
folders_to_kill.sort()
_ = [shutil.rmtree(folder) for folder in folders_to_kill]

# replace the first lines in prm to update the magic_file
folders = [os.path.join(tile_path, name) for name in os.listdir(tile_path) if os.path.isdir(os.path.join(tile_path, name))]

with open(prm_file, 'r') as file:
    for line in file:
        if line.startswith("%XX%:"):
            xx_values = list(map(int, line.split(":", 1)[1].strip().split()))
        elif line.startswith("%YY%:"):
            yy_values = list(map(int, line.split(":", 1)[1].strip().split()))


combis_on_disc = ['_'.join([x[0][-2:], x[1][-2:]]) for x in [folder.split('/')[-1].split('_') for folder in folders]]
combis_all = [f'{x}_{y}' for x, y in zip(xx_values, yy_values)]
indices = [i for i, val in enumerate(combis_all) if val not in combis_on_disc]

if len(indices) == 0:
    _ = [os.remove(file) for file in getFilelist(magic_path, '.prm')]
    print('finished')
else:
    pairs = [(x, y) for i, (x, y) in enumerate(zip(xx_values, yy_values)) if i in indices]
    xx = [str(x) for x, y in pairs]
    yy = [str(y) for x, y in pairs]

    # Read the file
    with open(prm_file, 'r') as f:
        lines = f.readlines()

    # Rewrite lines with filtered pairs
    new_line1 = '%XX%: ' + ' '.join(xx) + '\n'
    new_line2 = '%YY%: ' + ' '.join(yy) + '\n'

    # Replace the lines in the list
    lines[0] = new_line1
    lines[1] = new_line2

    # Write back to file (or to a new file)
    with open(prm_file_clean, 'w') as f:
        f.writelines(lines)

    # delete all old 
    _ = [os.remove(file) for file in getFilelist(magic_path, '.prm')]
    print('continue')
