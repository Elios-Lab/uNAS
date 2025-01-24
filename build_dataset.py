from pathlib import Path
import shutil
import tarfile
import csv

path_to_dataset = Path('/media/pigo/22F2EE2BF2EE0341/wake_vision/')

print("path_to_dataset: ", path_to_dataset)

folders_and_file_names = list()

#extract validation images metadata
img_names_0 = set()
img_names_1 = set()

with open(f'{path_to_dataset}/wake_vision_validation.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:]))
    for image_path, category, *_ in data[1:] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'validation/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'validation/1', 'file_names': img_names_1})

# #extract test images metadata
img_names_0 = set()
img_names_1 = set()

with open(f'{path_to_dataset}/wake_vision_test.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:-1]))
    for image_path, category, *_ in data[1:-1] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'test/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'test/1', 'file_names': img_names_1})

#extract train (large) images metadata
# img_names_0 = set()
# img_names_1 = set()

# with open(f'{path_to_dataset}/wake_vision_train_large.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)
#     print(len(data[1:-1]))
#     for image_path, category, *_ in data[1:] :
#         if '0' == category :
#             img_names_0.add(Path(image_path))
#         elif '1' == category :
#             img_names_1.add(Path(image_path))
#         else :
#             print('Unknown image category')
#             exit()

# folders_and_file_names.append({'folder': path_to_dataset / 'train_large/0', 'file_names': img_names_0})
# folders_and_file_names.append({'folder': path_to_dataset / 'train_large/1', 'file_names': img_names_1})

# extract train (quality) images metadata
img_names_0 = set()
img_names_1 = set()

with open(f'{path_to_dataset}/wake_vision_train_bbox.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    print(len(data[1:-1]))
    for image_path, category, *_ in data[1:] :
        if '0' == category :
            img_names_0.add(Path(image_path))
        elif '1' == category :
            img_names_1.add(Path(image_path))
        else :
            print('Unknown image category')
            exit()

folders_and_file_names.append({'folder': path_to_dataset / 'train_quality/0', 'file_names': img_names_0})
folders_and_file_names.append({'folder': path_to_dataset / 'train_quality/1', 'file_names': img_names_1})


for element in folders_and_file_names :
    element['folder'].mkdir(parents=True)

#extract all compressed images and copy them in the corresponding folders
for zipped_file in path_to_dataset.glob('*.tar.gz') :
    print("Extracted file:", zipped_file)
    tar = tarfile.open(zipped_file, 'r:gz')
    tar.extractall()
    tar.close()
    images = set(Path('.').glob('*.jpg'))
    
    for folder in folders_and_file_names :
        for image in images & folder['file_names'] :
            shutil.copy(image, folder['folder'])
    #delete extracted images
    for image in images : 
        image.unlink()
        
    #delete compressed file
    zipped_file.unlink()
