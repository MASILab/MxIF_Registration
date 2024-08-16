import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
from scipy.ndimage import map_coordinates
from skimage import exposure
from skimage.util import view_as_windows
from math import ceil
Image.MAX_IMAGE_PIXELS = None

class Utils_v2:
    @staticmethod
    def adjust_intensity(original_image, target_image):
        original_image = original_image.astype(float)
        target_image = target_image.astype(float)
        ratio = np.where(target_image != 0, original_image / target_image, 0)
        counts, bins = np.histogram(ratio[ratio != 0].ravel(), bins=255)
        max_count_index = np.argmax(counts)
        start = bins[max_count_index]
        end = bins[max_count_index + 1]
        ratios_in_bin = ratio[(ratio > start) & (ratio < end)]
        factor = ratios_in_bin.mean()
        adjusted_target_image = target_image * factor
        adjusted_target_image_new = np.minimum(adjusted_target_image, original_image.max())
        return adjusted_target_image_new,factor

    @staticmethod
    def load_tissues_for_overlap(tissues,mask):
        """"
        Args:
        tissues: tissue path
        mask: Path to the mask image
        Returns: Padded tissues, height and width of the mask
        """
        mask=np.array(Image.open(mask))
        height, width = mask.shape
        patch_size = 1024
        overlap = 200
        n_patches_height = ceil((mask.shape[0] - overlap) / (patch_size - overlap))
        n_patches_width = ceil((mask.shape[1] - overlap) / (patch_size - overlap))
        pad_height = n_patches_height * (patch_size - overlap) + overlap - mask.shape[0]
        pad_width = n_patches_width * (patch_size - overlap) + overlap - mask.shape[1]
        tissue=np.array(Image.open(tissues))
        tissue=(tissue*mask)/255.
        tissue = np.pad(tissue, ((0, pad_height), (0, pad_width)), mode='constant')

        return tissue,height,width
    @staticmethod
    def load_tissues_for_overlap_v2(tissues):
        """"
        Args:
        tissues: tissue path
        mask: Path to the mask image
        Returns: Padded tissues, height and width of the mask
        """
        mask=tissues
        mask=np.array(Image.open(mask))
        height, width = mask.shape
        patch_size = 1024
        overlap = 200
        n_patches_height = ceil((mask.shape[0] - overlap) / (patch_size - overlap))
        n_patches_width = ceil((mask.shape[1] - overlap) / (patch_size - overlap))
        pad_height = n_patches_height * (patch_size - overlap) + overlap - mask.shape[0]
        pad_width = n_patches_width * (patch_size - overlap) + overlap - mask.shape[1]
        tissue=np.array(Image.open(tissues))
        tissue = np.pad(tissue, ((0, pad_height), (0, pad_width)), mode='constant')

        return tissue,height,width
    
    @staticmethod
    def register_tissues_with_overlap(dapi_round_last, dapi_round0, model, device,stain1=None):
        """
        Register two images with overlap
        Args:
        dapi_round_last: dapi_round_last image
        dapi_round0: dapi_round0 image
        model: Model
        device: Device
        stain: Stain image
        Returns: Registered tissue and stain image
        """
        block_size = (1024, 1024)
        overlap = 200
        stride = (block_size[0] - overlap, block_size[1] - overlap)
        height, width = dapi_round_last.shape

        # Accumulator arrays for averaging overlapping areas
        full_tissue = np.zeros_like(dapi_round_last, dtype=np.float32)
        stain1 = np.zeros_like(dapi_round_last, dtype=np.float32)
        count_map = np.zeros_like(dapi_round_last, dtype=np.float32)

        num_blocks_x = (width - overlap) // stride[1] + 1
        num_blocks_y = (height - overlap) // stride[0] + 1

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                y_start = i * stride[0]
                x_start = j * stride[1]
                y_end = min(y_start + block_size[0], height)
                x_end = min(x_start + block_size[1], width)

                dapi_round_last_block = dapi_round_last[y_start:y_end, x_start:x_end]
                fixed_block = dapi_round0[y_start:y_end, x_start:x_end]
                stain_block = stain1[y_start:y_end, x_start:x_end]

                dapi_round_last_block = dapi_round_last_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                stain_block = stain_block[np.newaxis, ..., np.newaxis]
                if dapi_round_last_block.shape!=(1, 1024, 1024, 1):
                    continue
                dapi_round_last_block = torch.from_numpy(dapi_round_last_block).to(device).float().permute(0, 3, 1, 2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0, 3, 1, 2)

                fwd_pred,fwd_pred_field = model(dapi_round_last_block, fixed_block, registration=True)
                stain1_block = model.transformer(stain_block, fwd_pred_field)

                # Update full image and field accumulators
                full_tissue[y_start:y_end, x_start:x_end] += fwd_pred.detach().cpu().numpy().squeeze()
                stain1[y_start:y_end, x_start:x_end] += stain1_block.detach().cpu().numpy().squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1

        # Averaging the accumulated values
        full_tissue /= count_map
        stain1 /= count_map

        return full_tissue,stain1
    
    @staticmethod
    def register_multiple_tissues_with_overlap(dapi_round_last, dapi_round0, model, device,stain1=None,stain2=None,stain3=None,stain4=None,stain5=None,stain6=None):
        """
        Register two images and corresponding stains with overlap
        Args:
        dapi_round_last: dapi_round_last image
        dapi_round0: dapi_round0 image
        stain1: First stain image
        stain2: Second stain image
        stain3: Third stain image
        model: Model
        device: Device
        stain: Stain image
        Returns: Registered tissue and stain image
        """
        block_size = (1024, 1024)
        overlap = 200
        stride = (block_size[0] - overlap, block_size[1] - overlap)
        height, width = dapi_round_last.shape

        # Accumulator arrays for averaging overlapping areas
        full_tissue = np.zeros_like(dapi_round_last, dtype=np.float32)
        stain1_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain1 is not None else None
        stain2_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain2 is not None else None
        stain3_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain3 is not None else None
        stain4_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain4 is not None else None
        stain5_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain5 is not None else None
        stain6_accumulator = np.zeros_like(dapi_round_last, dtype=np.float32) if stain6 is not None else None
        count_map = np.zeros_like(dapi_round_last, dtype=np.float32)
        L2_map=np.zeros_like(dapi_round_last, dtype=np.float32)

        num_blocks_x = (width - overlap) // stride[1] + 1
        num_blocks_y = (height - overlap) // stride[0] + 1

        for i in range(num_blocks_y):
            for j in range(num_blocks_x):
                y_start = i * stride[0]
                x_start = j * stride[1]
                y_end = min(y_start + block_size[0], height)
                x_end = min(x_start + block_size[1], width)

                dapi_round_last_block = dapi_round_last[y_start:y_end, x_start:x_end]
                fixed_block = dapi_round0[y_start:y_end, x_start:x_end]
                stain1_block = stain1[y_start:y_end, x_start:x_end] if stain1 is not None else None
                stain2_block = stain2[y_start:y_end, x_start:x_end] if stain2 is not None else None
                stain3_block = stain3[y_start:y_end, x_start:x_end] if stain3 is not None else None
                stain4_block = stain4[y_start:y_end, x_start:x_end] if stain4 is not None else None
                stain5_block = stain5[y_start:y_end, x_start:x_end] if stain5 is not None else None
                stain6_block = stain6[y_start:y_end, x_start:x_end] if stain6 is not None else None

                dapi_round_last_block = dapi_round_last_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                if stain1_block is not None:
                    stain1_block = stain1_block[np.newaxis, ..., np.newaxis]
                if stain2_block is not None:
                    stain2_block = stain2_block[np.newaxis, ..., np.newaxis]
                if stain3_block is not None:
                    stain3_block = stain3_block[np.newaxis, ..., np.newaxis]
                if stain4_block is not None:
                    stain4_block = stain4_block[np.newaxis, ..., np.newaxis]
                if stain5_block is not None:
                    stain5_block = stain5_block[np.newaxis, ..., np.newaxis]
                if stain6_block is not None:
                    stain6_block = stain6_block[np.newaxis, ..., np.newaxis]
                if dapi_round_last_block.shape!=(1, 1024, 1024, 1):
                    continue
                dapi_round_last_block = torch.from_numpy(dapi_round_last_block).to(device).float().permute(0, 3, 1, 2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0, 3, 1, 2)

                fwd_pred,fwd_pred_field = model(dapi_round_last_block, fixed_block, registration=True)
                if stain1_block is not None:
                    stain1_block = torch.from_numpy(stain1_block).to(device).float().permute(0, 3, 1, 2)
                    stain1_block = model.transformer(stain1_block, fwd_pred_field)
                    stain1_accumulator[y_start:y_end, x_start:x_end] += stain1_block.detach().cpu().numpy().squeeze()
                if stain2_block is not None:
                    stain2_block = torch.from_numpy(stain2_block).to(device).float().permute(0, 3, 1, 2)
                    stain2_block = model.transformer(stain2_block, fwd_pred_field)
                    stain2_accumulator[y_start:y_end, x_start:x_end] += stain2_block.detach().cpu().numpy().squeeze()
                if stain3_block is not None:
                    stain3_block = torch.from_numpy(stain3_block).to(device).float().permute(0, 3, 1, 2)
                    stain3_block = model.transformer(stain3_block, fwd_pred_field)
                    stain3_accumulator[y_start:y_end, x_start:x_end] += stain3_block.detach().cpu().numpy().squeeze()
                if stain4_block is not None:
                    stain4_block = torch.from_numpy(stain4_block).to(device).float().permute(0, 3, 1, 2)
                    stain4_block = model.transformer(stain4_block, fwd_pred_field)
                    stain4_accumulator[y_start:y_end, x_start:x_end] += stain4_block.detach().cpu().numpy().squeeze()
                if stain5_block is not None:
                    stain5_block = torch.from_numpy(stain5_block).to(device).float().permute(0, 3, 1, 2)
                    stain5_block = model.transformer(stain5_block, fwd_pred_field)
                    stain5_accumulator[y_start:y_end, x_start:x_end] += stain5_block.detach().cpu().numpy().squeeze()
                if stain6_block is not None:
                    stain6_block = torch.from_numpy(stain6_block).to(device).float().permute(0, 3, 1, 2)
                    stain6_block = model.transformer(stain6_block, fwd_pred_field)
                    stain6_accumulator[y_start:y_end, x_start:x_end] += stain6_block.detach().cpu().numpy().squeeze()
                # Update full image and field accumulators
                full_tissue[y_start:y_end, x_start:x_end] += fwd_pred.detach().cpu().numpy().squeeze()
                count_map[y_start:y_end, x_start:x_end] += 1

        # Averaging the accumulated values
        full_tissue /= count_map
        if stain6 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            stain5_accumulator /= count_map
            stain6_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator,stain5_accumulator,stain6_accumulator
        elif stain5 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            stain5_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator,stain5_accumulator
        elif stain4 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            stain4_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator,stain4_accumulator
        elif stain3 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            stain3_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator,stain3_accumulator
        elif stain2 is not None:
            stain1_accumulator /= count_map
            stain2_accumulator /= count_map
            return full_tissue,stain1_accumulator,stain2_accumulator
        elif stain1 is not None:
            stain1_accumulator /= count_map
            return full_tissue,stain1_accumulator
        else:
            return full_tissue
    
    @staticmethod
    def generate_average_DAPI(dir_path):
        pass

    @staticmethod
    def load_model(model_path):
        """"
        Args:
        model_path: Path to the model
        Returns: Model and device
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = vxm.networks.VxmDense.load(model_path, device)
        model.to(device)
        model.eval()
        return model,device
    
    @staticmethod
    def calculate_ncc(array1, array2):
        """
        Calculate normalized cross correlation
        Args:
        array1: 1D array of your image1. Use np.rave() to convert 2D image to 1D
        array2: 1D array of your image2.
        Returns: Normalized cross correlation
        """
        array1 = (array1 - np.mean(array1)) / (np.std(array1) * len(array1))
        array2 = (array2 - np.mean(array2)) / (np.std(array2))
        ncc = np.correlate(array1, array2)
        return ncc

    @staticmethod
    def combine_displacement_fields(D1, D2):
        assert D1.shape == D2.shape, "Displacement fields must have the same shape"
        
        D_combined = np.zeros_like(D1)
        
        _, height, width = D1.shape
        
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        
        X_displaced = X + D1[0]
        Y_displaced = Y + D1[1]
        
        X_displaced_clipped = np.clip(X_displaced, 0, width - 1)
        Y_displaced_clipped = np.clip(Y_displaced, 0, height - 1)
        
        D2_x_interpolated = map_coordinates(D2[0], [Y_displaced_clipped, X_displaced_clipped], order=1)
        D2_y_interpolated = map_coordinates(D2[1], [Y_displaced_clipped, X_displaced_clipped], order=1)
        
        D_combined[0] = D1[0] + D2_x_interpolated
        D_combined[1] = D1[1] + D2_y_interpolated
        
        return D_combined
    
    @staticmethod
    def L2_norm_mask(moving, fixed, model, device):
        block_size = (1024, 1024)
#        num_blocks_x = moving.shape[0] // block_size[0]
#        num_blocks_y = moving.shape[1] // block_size[1]
#        original_tissue_cropped = moving[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
#        fixed_cropped = fixed[:num_blocks_x * block_size[0], :num_blocks_y * block_size[1]]
        moving_tissue_blocks = view_as_blocks(moving, block_shape=block_size)
        fixed_tissue_blocks = view_as_blocks(fixed, block_shape=block_size)
        pred_blocks_tissue = []
        pred_blocks_field=[]

        for i in range(moving_tissue_blocks.shape[0]):
            row_blocks_tissues = []
            row_blocks_field = []
            for j in range(moving_tissue_blocks.shape[1]):
                moving_block = moving_tissue_blocks[i, j]
                fixed_block = fixed_tissue_blocks[i, j]
                moving_block = moving_block[np.newaxis, ..., np.newaxis]
                fixed_block = fixed_block[np.newaxis, ..., np.newaxis]
                moving_block = torch.from_numpy(moving_block).to(device).float().permute(0,3,1,2)
                fixed_block = torch.from_numpy(fixed_block).to(device).float().permute(0,3,1,2)
                fwd_pred = model(moving_block,fixed_block ,registration=True)
                inv_pred = model(fixed_block,moving_block, registration=True)
                composite_field = Utils_v2.combine_displacement_fields(fwd_pred[1].detach().cpu().numpy().squeeze(), inv_pred[1].detach().cpu().numpy().squeeze())
                L2_norm_combined = np.sqrt(composite_field[0]**2 + composite_field[1]**2)
                row_blocks_tissues.append(fwd_pred[0].detach().cpu().numpy())
                row_blocks_field.append(L2_norm_combined)
            pred_blocks_tissue.append(row_blocks_tissues)
            pred_blocks_field.append(row_blocks_field)

        reconstructed_tissue = np.block(pred_blocks_tissue)
        composed_warp = np.block(pred_blocks_field)
        reconstructed_tissue = reconstructed_tissue.squeeze().squeeze()
        return composed_warp