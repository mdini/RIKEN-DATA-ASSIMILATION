import netCDF4 as nc
import numpy as np
import glob
import os
import re


dir_name='/data_ballantine02/miyoshi-t/amemiya/SCALE-LETKF-rt-private/result/exp/d4_500m_np8/dafcst/20210729'


list_of_files = sorted( filter( os.path.isfile, glob.glob(dir_name + '/2*mean*', recursive=True) ) )


print(list_of_files)

for file_path in list_of_files:
    print(file_path) 
    #arr=
    file_name=file_path[-24:]
    #print(arr)
    #np.save('2019/{}.npy'.format(file_name), arr.filled(0))