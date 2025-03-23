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

    G1_kaykay = raw_data_group.create_group('KayKay Raw Data')
    G1_lorenzo = raw_data_group.create_group('Lorenzo Raw Data')
    G1_daniil = raw_data_group.create_group('Daniil Raw Data')
    
    G1_kaykay.create_dataset('KayKay Walking Raw', data=kaykay_raw1.values)
    G1_kaykay.create_dataset('KayKay Jumping Raw', data=kaykay_raw2.values)
    
    G1_lorenzo.create_dataset('Lorenzo Walking Raw',data=lorenzo_raw1.values)
    G1_lorenzo.create_dataset('Lorenzo Jumping Raw', data=lorenzo_raw2.values)
    
    # G1_daniil.create_dataset('Daniil Walking Raw', data=daniil_raw1.values)
    # G1_daniil.create_dataset('Daniil Jumping Raw', data=daniil_raw2.values)
