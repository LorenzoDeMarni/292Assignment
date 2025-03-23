import numpy as np
import h5py

matrix1=np.random.random(size=(1000,1000))
matrix2=np.random.random(size=(1000,1000))
matrix3=np.random.random(size=(1000,1000))
matrix4=np.random.random(size=(1000,1000))
matrix5=np.random.random(size=(1000,1000))
matrix6=np.random.random(size=(1000,1000))
matrix7=np.random.random(size=(1000,1000))
matrix8=np.random.random(size=(1000,1000))
matrix9=np.random.random(size=(1000,1000))
matrix10=np.random.random(size=(1000,1000))
matrix11=np.random.random(size=(1000,1000))

with h5py.File('data_structure.h5', 'w') as hdf:
    G1 = hdf.create_group('Raw Data')
    G1.create_dataset('Raw Data', data=matrix1)
    G1_subgroup = G1.create_group('Raw Data Subgroups')
    G1_subgroup.create_dataset('KayKay Raw Data', data=matrix4)
    G1_subgroup.create_dataset('Lorenzo Raw Data', data=matrix5)
    G1_subgroup.create_dataset('Daniil Raw Data', data=matrix6)

    G2 = hdf.create_group('Pre-processed Data')
    G2.create_dataset('Pre-processed Data', data=matrix2)
    G2_subgroup = G2.create_group('Pre-processed Data Subgroups')
    G2_subgroup.create_dataset('KayKay Pre-processed Data', data=matrix7)
    G2_subgroup.create_dataset('Lorenzo Pre-processed Data', data=matrix8)
    G2_subgroup.create_dataset('Daniil Pre-processed Data', data=matrix9)

    G3 = hdf.create_group('Segmented Data')
    G3.create_dataset('Segmented Data', data=matrix3)
    G3_subgroup=G3.create_group('Segmented Data Subgroups')
    G3_subgroup.create_dataset('Train', data=matrix10)
    G3_subgroup.create_dataset('Test', data=matrix11)
