"""
    cropBBox - Python code to crop and save the bounding box annotations for images
    Saves the cropped images in given destination dir
    
    Modify - path, images_path, dst_dir for custom images 
    OUTPUT - cropped images in dst_dir/1,2,3,4 according to levels
"""

import json
from pycocotools.coco import COCO
import cv2
import os
import pandas as pd
import openpyxl

path = r"C:\Users\pmj02\sca\car-damage-detection-AIHUB\data\datainfo\part_train.json"
#path = r"C:\Users\pmj02\sca\car-damage-detection-AIHUB\data\datainfo\part_test.json"
# images_path = "../../data/custom/test/"
images_path = r"C:\Users\pmj02\sca\car-image-data\data\1.Training\1.origin\damage_part/"
#images_path = r"C:\Users\pmj02\sca\car-damage-detection-AIHUB\data\custom\test/"
file_path = r"C:\Users\pmj02\sca\car-image-data\data\1.Training\1.origin-xls"
dst_dir = r"C:\Users\pmj02\sca\car-damage-detection-AIHUB\dat/"


def classification(ann):
    level = 0
    img_id = img_names[ann['image_id']-1].replace(".jpg", "")
    part = ann['part'].replace(' ','')

    # Store xls file
    xls_path = file_path + '/' + img_id.split("_")[1] + '.xls'
    xls = pd.read_excel(xls_path, usecols=[2, 4], skiprows=range(0, 22))
    if xls.empty:
        return False
    xls.columns = ['part', 'level']

    # Translate part name and match
    if part == 'Frontbumper':
        _xls = xls[xls['part'].str.contains('?:프런트범퍼|프런트 범퍼|앞범퍼|후론트 범퍼|후론트범퍼')]
    elif part == 'Rearbumper':
        _xls = xls[xls['part'].str.contains('?:리어범퍼|리어 범퍼|뒤범퍼')]
    elif part == 'Frontfender(R)':
        _xls = xls[xls['part'].str.contains('?:프런트펜더(우)|프런트휀다(우)|앞펜더(우)|앞휀다(우)|앞휀더(우)|후론트휀다(우)|후론트 휀다(우)')]
    elif part == 'Frontfender(L)':
        _xls = xls[xls['part'].str.contains('?:프런트펜더(좌)|프런트 펜더(좌)|프런트휀다(좌)|앞펜더(좌)|앞휀다(좌)|앞휀더(좌)|후론트휀다(좌)|후론트 휀다(좌)')]
    elif part == 'Rearfender(R)':
        _xls = xls[xls['part'].str.contains('?:리어펜더(우)|리어휀다(우)|리어 휀다(우)|뒤펜더(우)|뒤휀다(우)|뒤휀더(우)')]
    elif part == 'Trunklid':
        _xls = xls[xls['part'].str.contains('?:트렁크')]
    elif part == 'Bonnet':
        _xls = xls[xls['part'].str.contains('?:본넷|본네트')]
    elif part == 'Rearfender(L)':
        _xls = xls[xls['part'].str.contains('?:리어펜더(좌)|리어휀다(좌)|리어 휀다(좌)|뒤펜더(좌)|뒤휀다(좌)|뒤휀더(좌)')]
    elif part == 'Reardoor(R)':
        _xls = xls[xls['part'].str.contains('?:리어도어|도어(뒤')]
    elif part == 'Headlights(R)':
        _xls = xls[xls['part'].str.contains('?:헤드라이트(우)|헤드램프(우)')]
    elif part == 'Headlights(L)':
        _xls = xls[xls['part'].str.contains('?:헤드라이트(좌)|헤드램프(좌)')]
    elif part == 'FrontWheel(R)':
        _xls = xls[xls['part'].str.contains('?:휠')]
    elif part == 'Frontdoor(R)':
        _xls = xls[xls['part'].str.contains('?:프런트도어|도어(앞|후론트 도어')]
    elif part == 'Sidemirror(R)':
        _xls = xls[xls['part'].str.contains('?:사이드미러')]
    else: # part is not exist in categories
        return False
    print("here?")
    # level classfication
    if _xls.empty: # is not exist in 견적서
        print("non")
        return False
    else:
        print(_xls)
        if _xls['level'][0] == '교환' or '1/2OH':
            level = 4
        elif _xls['level'][0] == '판금' or '오버홀':
            level = 3
        elif _xls['level'][0] == '도장' or '탈착':
            level = 1

    return dst_dir + str(level) + '/'+ img_id +'_'+ part +'.jpg'


# Store img file names
img_names = []
f = open(path)
data = json.load(f)
for i in data['images']:
    img_names.append(i['file_name'])


# Load coco format annotations
coco=COCO(path)
image_ids = coco.getImgIds()
annotation_ids = coco.getAnnIds()
anns = coco.loadAnns(annotation_ids)

# ann = anns[2]
# if ann['part'] != None:
#     dst = classification(ann)
#     print(dst)

# Draw boxes and add label to each box
for ann in anns:

    # create destination directories 
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    image = cv2.imread(images_path + img_names[ann['image_id']-1])
    #print("Dest: " + dst_dir+ img_names[ann['image_id']-1].replace(".jpg", "")+'_'+ann['part'].replace(' ','')+'.jpg')

    try: # for invalid bounding boxes
        [x,y,w,h] = ann['bbox']
        cropped_image = image[y:y+h,x:x+w]
        resized_image = cv2.resize(cropped_image, (244, 244))

        if ann['part'] != None:
            dst = classification(ann)
            if dst == False:
                pass
            else:
                cv2.imwrite(dst, resized_image)
    except:
        pass


print("Cropped Done")