import netCDF4 as nc
import numpy as np
import glob
import os
import re


dir_name='/data_ballantine01/miyoshi-t/nowcast_pawr/test_kobe/out/2019'


list_of_files = sorted( filter( os.path.isfile, glob.glob(dir_name + '/09/**/**/**/**/rain_cart_0002.nc', recursive=True) ) )


print(list_of_files)

for file_path in list_of_files:
    print(file_path) 
    d=nc.Dataset(file_path)
    arr=d.variables['rain'][0]
    file_name="".join(re.findall(r'\d+',file_path)[1:-1])
    print(file_name)
    np.save('2019/{}.npy'.format(file_name), arr.filled(0))
