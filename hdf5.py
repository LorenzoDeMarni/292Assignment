import numpy as np
import pandas as pd
import h5py

kaykay_raw1=pd.read_csv('walking_kaykay.csv')
kaykay_raw2=pd.read_csv('jumping_kaykay.csv')
lorenzo_raw1=pd.read_csv('walking_lorenzo.csv')
lorenzo_raw2=pd.read_csv('jumping_lorenzo.csv')
# daniil_raw1=pd.read_csv()
# daniil_raw2=pd.read_csv()

with h5py.File('data_structure.h5', 'w') as hdf:
    G1 = hdf.create_group('Raw Data')

    G1.create_dataset('KayKay Raw Walking Data', data=kaykay_raw1.values)
    G1.create_dataset('KayKay Raw Jumping Data', data=kaykay_raw2.values)
    G1.create_dataset('Lorenzo Raw Walking Data', data=lorenzo_raw1.values)
    G1.create_dataset('Lorenzo Raw Jumping Data', data=lorenzo_raw2.values)
    # G1.create_dataset('Daniil Raw Jumping Data', data=daniil_raw2.values)
    # G1.create_dataset('Daniil Raw Jumping Data', data=daniil_raw2.values)

    G1_subgroup = G1.create_group('Raw Data Subgroups')
    G1_subgroup.create_dataset('KayKay Raw Walking Data', data=kaykay_raw1.values)
    G1_subgroup.create_dataset('KayKay Raw Jumping Data', data=kaykay_raw2.values)
    G1_subgroup.create_dataset('Lorenzo Raw Walking Data', data=lorenzo_raw1.values)
    G1_subgroup.create_dataset('Lorenzo Raw Jumping Data', data=lorenzo_raw2.values)
    # G1_subgroup.create_dataset('Daniil Raw Walking Data', data=)
    # G1_subgroup.create_dataset('Daniil Raw Jumping Data', data=)
