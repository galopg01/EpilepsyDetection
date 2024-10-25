import openneuro as on
import os
import zipfile

def unzip_file(zip_filepath, dest_dir):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

if not os.path.exists('./DATASET'):
    os.makedirs('./DATASET', exist_ok=True)

if len(os.listdir('./DATASET')) == 0:
    if os.path.exists('./DATASET.ZIP'):
        print("Extracting dataset...")
        unzip_file('./DATASET.ZIP', './DATASET')
        print("Extraction completed")
    else:
        print("Downloading dataset...")
        on.download(dataset="ds004199", target_dir="./DATASET")
        print("Download completed")


print("Creating project structure...")
sizes = ['x32','x48','x64','x96','x128']
N = [1,3,5,7,9,10,11,15]
modes = ['normal','noSharpen','sharpen']

os.makedirs('./metrics/', exist_ok=True)

os.makedirs('./graphics/', exist_ok=True)
for size in sizes:
    os.makedirs(f'./graphics/{size}', exist_ok=True)
    for i in N:
        os.makedirs(f'./graphics/{size}/{i}', exist_ok=True)
        for mode in modes:
            os.makedirs(f'./graphics/{size}/{i}/{mode}', exist_ok=True)

os.makedirs('./models/', exist_ok=True)
for size in sizes:
    os.makedirs(f'./models/{size}', exist_ok=True)
    for i in N:
        os.makedirs(f'./models/{size}/{i}', exist_ok=True)
        for mode in modes:
            os.makedirs(f'./models/{size}/{i}/{mode}', exist_ok=True)

os.makedirs('./results/', exist_ok=True)
for size in sizes:
    os.makedirs(f'./results/{size}', exist_ok=True)
    for i in N:
        os.makedirs(f'./results/{size}/{i}', exist_ok=True)
        for mode in modes:
            os.makedirs(f'./results/{size}/{i}/{mode}', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/test', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/test/lesion', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/test/nonLesion', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/train', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/train/lesion', exist_ok=True)
            os.makedirs(f'./results/{size}/{i}/{mode}/train/nonLesion', exist_ok=True)

os.makedirs('./testData/', exist_ok=True)
for i in N:
    os.makedirs(f'./testData/{i}', exist_ok=True)
    for mode in modes:
        os.makedirs(f'./testData/{i}/{mode}', exist_ok=True)

print("Project structure created")