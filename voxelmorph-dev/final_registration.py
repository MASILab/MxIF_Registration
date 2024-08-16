import torch
import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import matplotlib.pyplot as plt
import glob
import re
import csv
from tifffile import imwrite
import numpy as np
from PIL import Image
import fnmatch
import shutil
from my_utils_v2 import Utils_v2        
from tqdm import tqdm
import subprocess
from collections import defaultdict
import gc

#Use argparse to get the path to the folder containing the images
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path to the folder containing the images")
parser.add_argument("tissue_name", help="name of the tissue")
parser.add_argument("output_path", help="path to the folder where the output will be saved")
parser.add_argument("model_path", help="path to the trained model")
args = parser.parse_args()
tissue_name = args.tissue_name
tissue_path = args.path
output_path = args.output_path
model_path = args.model_path


all_tissues=[]
for file in os.listdir(tissue_path):
    if fnmatch.fnmatch(file, f'{tissue_name}_*'):
        all_tissues.append(os.path.join(tissue_path, file))
print(len(all_tissues))

filtered_tissues = list(filter(lambda x: 'RetentionMask' not in x, all_tissues))
print(len(filtered_tissues))
# Group the strings by round number
tissues_by_round = defaultdict(list)
for tissue in filtered_tissues:
    round_number = int(re.search('ROUND_(\\d+)', tissue).group(1))
    tissues_by_round[round_number].append(tissue)

# Sort the dictionary items by key (round number)
sorted_items = sorted(tissues_by_round.items())

# Sort each group so that the string containing "DAPI_DAPI" comes first, and convert the sorted dictionary items to a list
grouped_tissues = [sorted(group, key=lambda x: f'{tissue_name}_DAPI' not in x) for _, group in sorted_items]
round0_dapi = grouped_tissues[0][0]
round0_dapi,height,width = Utils_v2.load_tissues_for_overlap_v2(round0_dapi)

round0_dapi_for_composition = grouped_tissues[0][0]
round_last_dapi_for_compositin = grouped_tissues[-1][0]
block_size = 1024

round0_dapi_for_composition=np.array(Image.open(round0_dapi_for_composition)) / 255.
round_last_dapi_for_compositin=np.array(Image.open(round_last_dapi_for_compositin)) /255.
original_height, original_width = round0_dapi_for_composition.shape

pad_size0 = block_size - round0_dapi_for_composition.shape[0] % block_size
pad_size1 = block_size - round0_dapi_for_composition.shape[1] % block_size

# Pad the images
round0_dapi_for_composition = np.pad(round0_dapi_for_composition, ((0, pad_size0), (0, pad_size1)))
round_last_dapi_for_compositin = np.pad(round_last_dapi_for_compositin, ((0, pad_size0), (0, pad_size1)))

model,device=Utils_v2.load_model(model_path) #Write this code

L2_norm=Utils_v2.L2_norm_mask(round0_dapi_for_composition,round_last_dapi_for_compositin,model,device)

mask = L2_norm > 10

mask_image = (mask * 255).astype(np.uint8)
mask_image = mask_image[:original_height, :original_width]

height1,width1=round0_dapi.shape
pad_height=height1-mask_image.shape[0]
pad_width=width1-mask_image.shape[1]

mask_image=np.pad(mask_image,((0,pad_height),(0,pad_width)))
round0_dapi[mask_image == 255] = 0

new_dir = os.path.join(output_path, tissue_name)

os.makedirs(new_dir, exist_ok=True)

for file_path in grouped_tissues[0]:
    file,height,width=Utils_v2.load_tissues_for_overlap_v2(file_path)
    file=file*255.
    file = file[:height, :width]
    Image.fromarray(np.uint8(file)).save(os.path.join(new_dir, os.path.basename(file_path)))
i=0
for group in grouped_tissues[1:]:
    stains=len(group)
    i+=1
    print(f"Registering Round {i}, which has {stains} number of stains")
    if stains==1:
        print("Bruh where are all the tissues")
    elif stains==2:
        roundi_dapi = group[0]
        stain1=group[1]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        registered_dapi,registered_stain1=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
    elif stains==3:
        roundi_dapi = group[0]
        stain1=group[1]
        stain2=group[2]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        stain2,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain2)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        stain2[mask_image == 255] = 0
        registered_dapi,registered_stain1,registered_stain2=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,stain2=stain2,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_stain2 = registered_stain2*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        registered_stain2 = registered_stain2[:height, :width]
    
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
        Image.fromarray(np.uint8(registered_stain2)).save(os.path.join(new_dir,os.path.basename(group[2])))
    elif stains==4:
        roundi_dapi = group[0]
        stain1=group[1]
        stain2=group[2]
        stain3=group[3]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        stain2,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain2)
        stain3,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain3)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        stain2[mask_image == 255] = 0
        stain3[mask_image == 255] = 0
        registered_dapi,registered_stain1,registered_stain2,registered_stain3=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,stain2=stain2,stain3=stain3,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_stain2 = registered_stain2*255.
        registered_stain3 = registered_stain3*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        registered_stain2 = registered_stain2[:height, :width]
        registered_stain3 = registered_stain3[:height, :width]
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
        Image.fromarray(np.uint8(registered_stain2)).save(os.path.join(new_dir,os.path.basename(group[2])))
        Image.fromarray(np.uint8(registered_stain3)).save(os.path.join(new_dir,os.path.basename(group[3])))
    elif stains==5:
        roundi_dapi = group[0]
        stain1=group[1]
        stain2=group[2]
        stain3=group[3]
        stain4=group[4]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        stain2,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain2)
        stain3,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain3)
        stain4,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain4)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        stain2[mask_image == 255] = 0
        stain3[mask_image == 255] = 0
        stain4[mask_image == 255] = 0
        registered_dapi,registered_stain1,registered_stain2,registered_stain3,registered_stain4=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,stain2=stain2,stain3=stain3,stain4=stain4,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_stain2 = registered_stain2*255.
        registered_stain3 = registered_stain3*255.
        registered_stain4 = registered_stain4*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        registered_stain2 = registered_stain2[:height, :width]
        registered_stain3 = registered_stain3[:height, :width]
        registered_stain4 = registered_stain4[:height, :width]
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
        Image.fromarray(np.uint8(registered_stain2)).save(os.path.join(new_dir,os.path.basename(group[2])))
        Image.fromarray(np.uint8(registered_stain3)).save(os.path.join(new_dir,os.path.basename(group[3])))
        Image.fromarray(np.uint8(registered_stain4)).save(os.path.join(new_dir,os.path.basename(group[4])))
    elif stains==6:
        roundi_dapi = group[0]
        stain1=group[1]
        stain2=group[2]
        stain3=group[3]
        stain4=group[4]
        stain5=group[5]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        stain2,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain2)
        stain3,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain3)
        stain4,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain4)
        stain5,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain5)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        stain2[mask_image == 255] = 0
        stain3[mask_image == 255] = 0
        stain4[mask_image == 255] = 0
        stain5[mask_image == 255] = 0
        registered_dapi,registered_stain1,registered_stain2,registered_stain3,registered_stain4,registered_stain5=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,stain2=stain2,stain3=stain3,stain4=stain4,stain5=stain5,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_stain2 = registered_stain2*255.
        registered_stain3 = registered_stain3*255.
        registered_stain4 = registered_stain4*255.
        registered_stain5 = registered_stain5*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        registered_stain2 = registered_stain2[:height, :width]
        registered_stain3 = registered_stain3[:height, :width]
        registered_stain4 = registered_stain4[:height, :width]
        registered_stain5 = registered_stain5[:height, :width]
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
        Image.fromarray(np.uint8(registered_stain2)).save(os.path.join(new_dir,os.path.basename(group[2])))
        Image.fromarray(np.uint8(registered_stain3)).save(os.path.join(new_dir,os.path.basename(group[3])))
        Image.fromarray(np.uint8(registered_stain4)).save(os.path.join(new_dir,os.path.basename(group[4])))
        Image.fromarray(np.uint8(registered_stain5)).save(os.path.join(new_dir,os.path.basename(group[5])))
    elif stains==7:
        roundi_dapi = group[0]
        stain1=group[1]
        stain2=group[2]
        stain3=group[3]
        stain4=group[4]
        stain5=group[5]
        stain6=group[6]
        roundi_dapi,_,_ = Utils_v2.load_tissues_for_overlap_v2(roundi_dapi)
        stain1,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain1)
        stain2,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain2)
        stain3,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain3)
        stain4,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain4)
        stain5,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain5)
        stain6,_,_ = Utils_v2.load_tissues_for_overlap_v2(stain6)
        roundi_dapi[mask_image == 255] = 0
        stain1[mask_image == 255] = 0
        stain2[mask_image == 255] = 0
        stain3[mask_image == 255] = 0
        stain4[mask_image == 255] = 0
        stain5[mask_image == 255] = 0
        stain6[mask_image == 255] = 0
        registered_dapi,registered_stain1,registered_stain2,registered_stain3,registered_stain4,registered_stain5,registered_stain6=Utils_v2.register_multiple_tissues_with_overlap(dapi_round_last=roundi_dapi,dapi_round0=round0_dapi,
                                                                                          stain1=stain1,stain2=stain2,stain3=stain3,stain4=stain4,stain5=stain5,stain6=stain6,model=model,device=device)
        registered_dapi = registered_dapi*255.
        registered_stain1 = registered_stain1*255.
        registered_stain2 = registered_stain2*255.
        registered_stain3 = registered_stain3*255.
        registered_stain4 = registered_stain4*255.
        registered_stain5 = registered_stain5*255.
        registered_stain6 = registered_stain6*255.
        registered_dapi = registered_dapi[:height, :width]
        registered_stain1 = registered_stain1[:height, :width]
        registered_stain2 = registered_stain2[:height, :width]
        registered_stain3 = registered_stain3[:height, :width]
        registered_stain4 = registered_stain4[:height, :width]
        registered_stain5 = registered_stain5[:height, :width]
        registered_stain6 = registered_stain6[:height, :width]
        Image.fromarray(np.uint8(registered_dapi)).save(os.path.join(new_dir,os.path.basename(group[0])))
        Image.fromarray(np.uint8(registered_stain1)).save(os.path.join(new_dir,os.path.basename(group[1])))
        Image.fromarray(np.uint8(registered_stain2)).save(os.path.join(new_dir,os.path.basename(group[2])))
        Image.fromarray(np.uint8(registered_stain3)).save(os.path.join(new_dir,os.path.basename(group[3])))
        Image.fromarray(np.uint8(registered_stain4)).save(os.path.join(new_dir,os.path.basename(group[4])))
        Image.fromarray(np.uint8(registered_stain5)).save(os.path.join(new_dir,os.path.basename(group[5])))
        Image.fromarray(np.uint8(registered_stain6)).save(os.path.join(new_dir,os.path.basename(group[6])))
print(f"Registration Done. Output Saved at {new_dir}")

# Clear GPU memory 
del model  # Remove the model from memory
torch.cuda.empty_cache()  # Clear the cached memory
gc.collect()  # Force garbage collection

files = glob.glob(os.path.join(new_dir, "*DAPI_DAPI*"))
#files = glob.glob(os.path.join(new_dir, "*DAPI_UV*")) #This is for Set01
print(files)
def get_round_number(filename):
    match = re.search(r'ROUND_(\d+)', filename)
    return int(match.group(1)) if match else 0

# Sort the files by the round number
files = sorted(files, key=get_round_number)
print(files[0])
file_zer0=files[0]
image0=np.array(Image.open(file_zer0))
image_accumulator=np.zeros(image0.shape)

for file in tqdm(files):
    image=np.array(Image.open(file))
    image,_=Utils_v2.adjust_intensity(image0,image)
    image_accumulator+=image
image_accumulator=image_accumulator/len(files)

Image.fromarray(np.uint8(image_accumulator)).save(os.path.join(new_dir,f"AverageDAPI_{tissue_name}.tif"))