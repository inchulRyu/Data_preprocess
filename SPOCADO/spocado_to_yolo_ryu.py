import os
import json
from tqdm import tqdm


def bboxNormalize(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0

    w = box[2]
    h = box[3]

    w = w*1.02
    h = h*1.02

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (round(x, 3), round(y, 3), round(w, 3), round(h, 3))

# label_file_path = '/DL_data/Spocado/baseball_annotations'
# label_file_path = '/DL_data/Spocado/data/2008026_상무_vs_삼성_마스터/json'
# yolo_label_save_path = '/DL_data/Spocado/data/2008026_상무_vs_삼성_마스터/labels'
# img_path = '/DL_data/Spocado/data/2008026_상무_vs_삼성_마스터/images'

# label_file_path = '/DL_data/Spocado/data/json/3'
# yolo_label_save_path = '/DL_data/Spocado/data/labels/ryu/3'
# img_path = '/DL_data/Spocado/data/images/3'

# label_file_path = '/DL_data/Spocado/data/2008026_상무_vs_삼성_마스터/json'
# yolo_label_save_path = '/DL_data/Spocado/data/labels/ryu/2008026_상무_vs_삼성_마스터'
# img_path = '/DL_data/Spocado/data/images/2008026_상무_vs_삼성_마스터'

label_file_path = '/DL_data/icryu/baseball_annotations'
yolo_label_save_path = '/DL_data/Spocado/data/labels/ryu/demo1'
img_path = '/DL_data/icryu/baseball_images'

files = os.listdir(label_file_path)

### 파일 열여서 data에 정리
cls_dic = {'0' : 0, # umpire_in_chief
           '1': 0, # assistant_umpire
           '2' : 0, # assistant_umpire
           '3' : 0, # assistant_umpire
           '4' : 1, # manager
           '5' : 1, # coach
           '6' : 1, # player
           '7' : 1, # batter
           '8' : 1, # runner
           '9' : 1, # pitcher
           '10' : 1, # catcher
           '11' : 1, # fielder
           '12' : 1, # fielder
           '13' : 2, # home_base
           '14' : 3, # bases
           '15' : 3, # bases
           '16' : 3, # bases
           '17' : -1, # mound
           '18' : -1, # pitchers_plate
           '19' : -1, # on_deck_circle
           '20' : -1, # on_deck_circle
           '21' : -1, # batters_box
           '22' : -1, # batters_box
           '23' : -1, # outfield
           '24' : -1, # infield
           '25' : -1, # bench
           '26' : -1, # ball
           '27' : 4, # bat
           '28' : -1, # uniform
           '29' : 5, # glove
           '30' : -1, # hat
           '31' : -1 # helmet
           }

data = {}

# cnt = 0
for each_file in tqdm(files):
    filename = os.path.splitext(each_file)[0]
    file_ext = os.path.splitext(each_file)[1]

    # print(filename)

    if file_ext == '.json':
        with open(os.path.join(label_file_path, filename+file_ext), 'r') as jf:
            contents = json.load(jf)

        image_size = (contents['image']['width'], contents['image']['height'])

        for each_obj in contents['annotations']:
            if filename not in data:
                data[filename] = []
            if each_obj['category_id'] <= 31:
                # each_obj['bbox'] = (x1, y1, width, height)
                # bbox 좌표들 정규화 시킴.
                normalized_bbox = bboxNormalize(image_size, each_obj['bbox'])
                # yolo format으로 새로운 line 만듦.
                class_id = cls_dic[str(each_obj["category_id"])]
                if class_id in [-1]:
                    continue
                new_line = f'{class_id} {normalized_bbox[0]} {normalized_bbox[1]} {normalized_bbox[2]} {normalized_bbox[3]}'
                
                data[filename].append(new_line)

# print(cnt)
print(len(data.values()))


for each_file in files:
    filename = os.path.splitext(each_file)[0]
    if filename not in list(data.keys()):
        print(filename)

for key in tqdm(data):
    with open(os.path.join(yolo_label_save_path, ''.join([key, '.txt'])), 'w') as output_txt_file:
        for each_anno in data[key]:
            output_txt_file.write(each_anno + '\n')

        