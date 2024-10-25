import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom


def has_lesion(mask_window, threshold=0.5, N=1):
    num_positive_pixels = (mask_window > threshold).sum()     
    return num_positive_pixels >= N

def delete_images(size,N,mode):
    dir = 'results/x'+ str(size) + '/' + str(N) + '/' + mode + '/train/lesion/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
    dir = 'results/x'+ str(size)  + '/' + str(N) + '/' + mode +  '/train/nonLesion/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
    dir = 'results/x'+ str(size) + '/' + str(N) + '/' + mode +  '/test/lesion/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = 'results/x'+ str(size) + '/' + str(N) + '/' + mode + '/test/nonLesion/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def resize_mask(mask):
    final_shape = (208,320,320)
    # Redimensionar la máscara usando scipy.ndimage.zoom
    zoom_factors = [final_shape[i] / mask.shape[i] for i in range(3)]
    resized_mask = zoom(mask, zoom_factors, order=0, mode='constant', cval=0)  # order=0 para interpolación más cercana

    return resized_mask

def is_black(img, threshold=20):
    pixels = np.array(img)
    
    total_pixels = pixels.shape[0] * pixels.shape[1]
    black_pixels = np.sum(pixels < threshold)

    percentage_black = black_pixels / total_pixels
    
    return percentage_black >= 0.85

def delete_black_images(size,N,mode, threshold=20):
    dirs = ['results/x' + str(size) +'/' + str(N) + '/' + mode + '/train/lesion/',
            'results/x' + str(size) +'/' + str(N) + '/' + mode + '/train/nonLesion/'.format(size),
            'results/x' + str(size) +'/' + str(N) + '/' + mode + '/test/lesion/'.format(size),
            'results/x' + str(size) +'/' + str(N) + '/' + mode + '/test/nonLesion/'.format(size)]

    for dir in dirs:
        for f in os.listdir(dir):
            img_path = os.path.join(dir, f)
            img = Image.open(img_path)
            if is_black(img, threshold):
                os.remove(img_path)

patients = [0, 2, 3, 5, 8, 9, 13, 14, 15, 17, 19, 23, 26, 31, 32, 33, 37, 39, 42, 43, 46, 47, 49, 52, 54, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 71, 72, 73, 75, 76, 77, 79, 80, 82, 86, 88, 89, 90, 91, 94, 96, 97, 98, 99, 100, 102, 104, 106, 107, 108, 111, 114, 115, 119, 120, 121, 122, 124, 125, 127, 129, 130, 131, 132, 133, 135, 137, 138, 139, 140, 141, 143, 144, 145]

def is_train(subject, test_size=0.1, random_state=42):
    train_patients, test_patients = train_test_split(patients, test_size=0.1, random_state=random_state)

    return subject in train_patients


def random_split_image_3d(image, mask, patch_size, num_windows):
    patches = []
    patches_mask = []
    patch_sizes = (1, patch_size, patch_size, patch_size)
    image_shape=image.shape

    for _ in range(num_windows):
        
        i = np.random.randint(0, image_shape[0] - patch_size + 1)
        j = np.random.randint(0, image_shape[1] - patch_size + 1)
        k = np.random.randint(0, image_shape[2] - patch_size + 1)

        patch = image[ i:i+patch_size, j:j+patch_size, k:k+patch_size]
        mask_patch = mask[i:i+patch_size, j:j+patch_size, k:k+patch_size] 
        patches.append(patch)
        patches_mask.append(mask_patch)

    return patches, patches_mask


def generate_patches(patch_size,N,mode):
    data_dir = 'DATASET/'
   
    if(patch_size==128):
        num_windows = 4
    elif(patch_size==96):
        num_windows = 6
    elif(patch_size==64):
        num_windows = 8
    elif(patch_size==48):
        num_windows =25
    else:
        num_windows = 25


    print("Generating images...")

    j=0
    for sub in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, sub)):
            dir_anat = os.path.join(data_dir, sub, 'anat')
            files_t1w = [file for file in os.listdir(dir_anat) if file.endswith('_T1w.nii.gz')]
            files_roi = [file for file in os.listdir(dir_anat) if file.endswith('_FLAIR_roi.nii.gz')]
            
            # Verify the presence of both files in 'anat'
            if files_t1w and files_roi:

                training = is_train(j)
                
                route_t1w = os.path.join(dir_anat, files_t1w[0])
                data_t1w = nib.load(route_t1w)
                img = data_t1w.get_fdata()
                data_t1w.uncache()
                
                route_roi = os.path.join(dir_anat, files_roi[0])
                data_roi = nib.load(route_roi)
                mask = data_roi.get_fdata()
                mask = resize_mask(mask)
                data_roi.uncache()
                img_slices, mask_slices = random_split_image_3d(img, mask, patch_size, num_windows)

                output_dir = 'results/x' + str(patch_size) + '/' + str(N) + '/' + mode + '/'
                os.makedirs(output_dir, exist_ok=True)
                for i, (img_slice, mask_slice) in enumerate(zip(img_slices, mask_slices)):
                    for z, (slice, mask) in enumerate(zip(img_slice, mask_slice)):

                        if has_lesion(mask, 0.5, N):
                            slice_array = np.array(slice)
                            if training:
                                plt.imsave(os.path.join(output_dir, f'train/lesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slice_array, cmap='gray')
                            else:
                                plt.imsave(os.path.join(output_dir, f'test/lesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slice_array, cmap='gray')

                        else:
                            slice_array = np.array(slice)
                            if training:
                                plt.imsave(os.path.join(output_dir, f'train/nonLesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slice_array, cmap='gray')
                            else:
                                plt.imsave(os.path.join(output_dir, f'test/nonLesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slice_array, cmap='gray')

                        
            j += 1   


def generate_3d_patches(patch_size,N,mode):

    data_dir = 'DATASET/'

    if(patch_size==128):
        num_windows = 4
    elif(patch_size==96):
        num_windows = 6
    elif(patch_size==64):
        num_windows = 10
    elif(patch_size==48):
        num_windows =16
    else:
        num_windows = 25

    print("Generating patches...")

    j = 0
    for sub in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, sub)):
            dir_anat = os.path.join(data_dir, sub, 'anat')
            files_t1w = [file for file in os.listdir(dir_anat) if file.endswith('_T1w.nii.gz')]
            files_roi = [file for file in os.listdir(dir_anat) if file.endswith('_FLAIR_roi.nii.gz')]
            
            # Verify the presence of both files in 'anat'
            if files_t1w and files_roi:
                training = is_train(j)
                
                route_t1w = os.path.join(dir_anat, files_t1w[0])
                data_t1w = nib.load(route_t1w)
                img = data_t1w.get_fdata()
                data_t1w.uncache()
                
                route_roi = os.path.join(dir_anat, files_roi[0])
                data_roi = nib.load(route_roi)
                mask = data_roi.get_fdata()
                mask = resize_mask(mask)
                data_roi.uncache()
                
                img_slices, mask_slices = random_split_image_3d(img, mask, patch_size, num_windows)
                output_dir = 'results/x' + str(patch_size) + '/' + str(N) + '/' + mode + '/'
                os.makedirs(output_dir, exist_ok=True)
                
                for i, (img_slice, mask_slice) in enumerate(zip(img_slices, mask_slices)):
                    # Iterate with stride 3
                    for z in range(0, len(img_slice)-2):
                        combined_slices = img_slice[z:z+3]  
                        combined_masks = mask_slice[z:z+3]   
                        if has_lesion(np.concatenate(combined_masks), 0.5, N):
                            slices_array = np.stack(combined_slices, axis=0) 
                            slices_array = np.transpose(slices_array, (1, 2, 0))
                            max_pixel_value = np.max(slices_array)
                            if max_pixel_value != 0:
                                slices_array = slices_array.astype(np.float32) / max_pixel_value

                            if training:
                                plt.imsave(os.path.join(output_dir, f'train/lesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slices_array)
                            else:
                                plt.imsave(os.path.join(output_dir, f'test/lesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slices_array)
                        else:
                            slices_array = np.stack(combined_slices, axis=0) 
                            slices_array = np.transpose(slices_array, (1, 2, 0))
                            max_pixel_value = np.max(slices_array)
                            if max_pixel_value != 0:
                                slices_array = slices_array.astype(np.float32) / max_pixel_value

                            if training:
                                plt.imsave(os.path.join(output_dir, f'train/nonLesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slices_array)
                            else:
                                plt.imsave(os.path.join(output_dir, f'test/nonLesion/p{j}_{z}_{patch_size}_{patch_size}_{i}.png'), slices_array)
            j += 1
        
    print("Patches generated")


def labelnum(label):
    if(label=='nonLesion'):
            return 0
    elif(label=='lesion'):
            return 1
    else:
            return 0


def imgsToCsv(patch_size,N,mode):
    data_dir = os.path.expanduser('results/x' + str(patch_size) + '/' + str(N) + '/' + mode + '/')

    train_data_dir = os.path.join(data_dir, 'train/')
    test_data_dir = os.path.join(data_dir, 'test/')

    # for training
    train_df = pd.DataFrame(columns=["img_name","subject","label"])

    filenames = glob.glob(os.path.join(train_data_dir, '**/*.png'))
    train_df["img_name"] = filenames
    train_df["subject"] =  [re.search(r'p(\d+)', filename).group(1) for filename in filenames]

    classes = (os.path.basename(os.path.dirname(name)) for name in filenames)
    classes = list(classes)

    for i,name in enumerate(filenames):
        label = labelnum(classes[i])
        train_df["label"][i] = label

    filename = r'testData/' + str(N) + '/' + mode + '/trainx' + str(patch_size) + '.csv'
    # Delete the file if exists
    if os.path.isfile(filename):
        os.remove(filename)
        
    train_df.to_csv (filename, index = False, header=True)
   

    # For tests
    test_df = pd.DataFrame(columns=["img_name","subject","label"])

    filenames = glob.glob(os.path.join(test_data_dir, '**/*.png'))
    test_df["img_name"] = filenames
    test_df["subject"] =   [re.search(r'p(\d+)', filename).group(1) for filename in filenames]

    classes = (os.path.basename(os.path.dirname(name)) for name in filenames)
    classes = list(classes)
    for i,name in enumerate(filenames):
        label = labelnum(classes[i])
        test_df["label"][i] = label

    filename = r'testData/' + str(N) + '/' + mode + '/testx' + str(patch_size) + '.csv'
    # Delete the file if exists
    if os.path.isfile(filename):
        os.remove(filename)
      
    test_df.to_csv (filename, index = False, header=True)

