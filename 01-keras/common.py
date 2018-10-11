import os

# File Names
root_model_folder = './model/'

if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)

model_file_name = root_model_folder + 'model.h5'
model_checkpoint_file_name = root_model_folder + 'model-checkpoint.hdf5'

# Column Names
X_colum_names = ['x']
Y_colum_names = ['y']
