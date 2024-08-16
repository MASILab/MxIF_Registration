# Deformable registration of MxIF images using DAPI stains as anchors
This pipeline can be used to perform deformable registration for MxIF images using DAPI stains as anchors. It has 3 main functions:
1. It identifies and masks out areas of tissue deformation and tissue loss by composing the forward and inverse registration fields from the first round of DAPI to the last.
2. It uses the individual rounds of DAPI as anchors and registers the stains associated with each round of DAPI into a unified space.
3. After registering all of the DAPI rounds back to the initial round, it adjusts the intensity differences between the different rounds of DAPI and averages them to make a representative DAPI image that accounts for all rounds.

# Installation
Please clone this repo and use conda to create a python environment using the yml file provided.
```
conda env create --name mxif_registration -f environment.yml
```

# Directory Structure and Naming conventions
Each of the MxIF subjects should be in a separate directory and should follow the naming conventions metioned in example_filenames.txt. Additionaly the images should be rigidly registered before performing deformable registration. Each DAPI image should have ROUND_XX in it's filename indicating which round it was acquired in. Each maker or background image acquired in conjunction with a DAPI image should also have it's corresponding round number mentioned in the filename.

#Usage
Once the images have been rigidly registered and the filenames are representative of the round of acquisition, you can register the images in the following way.
```
data_dir="/path/to/data/dir"
model_path="/path/to/trained/voxelmorph/model"
tissue_name="Name_of_Subject"
output_path="path/where/you/want/output/to/be/stored"

#Perform registration
python final_registration.py \
	--path $data_dir \
	--tissue_name $tissue_name
	--output_path $output_path
	--model_path $model_path
```
