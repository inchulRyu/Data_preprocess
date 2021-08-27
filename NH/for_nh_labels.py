import os
import numpy as np
import json

def coordinateCvt2YOLO(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]

    # (xmin + xmax / 2)
    x = (box[0] + box[1]) / 2.0
    # (ymin + ymax / 2)
    y = (box[2] + box[3]) / 2.0

    # (xmax - xmin) = w
    w = box[1] - box[0]
    # (ymax - ymin) = h
    h = box[3] - box[2]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (round(x, 3), round(y, 3), round(w, 3), round(h, 3))


# json_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/label_1/nh.json'
json_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/label_1/ai-hub.json'
# manifest_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/manifest.txt'
manifest_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/manifest_2.txt'
# img_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/target_label_for_NH/1~15000/'
img_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/target_label_for_NH/15001~30000/'
# img_type = '.png'
img_type = '.jpg'
# save_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/label_1_converted/'
save_path = '/local_hdd/works/ICRyu_workspace/for_yolo/dataset/data1/label_1_converted_2/'



cls_dic = {'person_s' : 0, 'person_l' : 0, 'car' : 1, 'bicycle' : 2}
data = {}

with open(json_path, 'r') as f:
    json_f = json.load(f)

cnt = 0

for each_anno in json_f["annotations"]:
    for each_image in json_f["images"]:

        if each_image['id'] == each_anno['image_id']:
            img_file_path = each_image['file_name']
            img_size = (each_image['width'], each_image['height'])

    img_file_name = img_file_path.split('/')[1].split('.')[0]

    # (x1, x2, y1, y2)
    box = (each_anno['bbox'][0], each_anno['bbox'][0] + each_anno['bbox'][2], each_anno['bbox'][1], each_anno['bbox'][1] + each_anno['bbox'][3])
    box_normalized = coordinateCvt2YOLO(img_size, box)

    for each_cate in json_f["categories"]:
        if each_cate['id'] == each_anno['category_id']:
            obj_cls_name = each_cate['name']

    obj_cls = cls_dic[obj_cls_name]
    newline = f'{obj_cls} {box_normalized[0]} {box_normalized[1]} {box_normalized[2]} {box_normalized[3]}'

    try:
        data[img_file_name].append(newline)
    except KeyError:
        data[img_file_name] = []
        cnt += 1
        data[img_file_name].append(newline)

print(cnt)


if os.path.isdir(manifest_path):
    manifest_abspath = os.path.join(manifest_path, "manifest.txt")
else:
    manifest_abspath = manifest_path

with open(os.path.abspath(manifest_abspath), "w") as manifest_file:

    for key in data:
        manifest_file.write(os.path.abspath(os.path.join(
            img_path, "".join([key, img_type, "\n"]))))

        with open(os.path.abspath(os.path.join(save_path, "".join([key, ".txt"]))), "w") as output_txt_file:
            for each_anno in data[key]:
                output_txt_file.write(each_anno+'\n')