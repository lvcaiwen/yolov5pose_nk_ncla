import json
import os
import random
from tqdm import tqdm
import argparse
import cv2
import shutil
import numpy as np


def comparison(o,r):
    json_name = [i.split('.')[0] for i in o]
    img_name = [i.split('.')[0] for i in r]
    for name in img_name:
        if name not in json_name:
            print(name,' not json')

def xyxy2xywh(xyxy):
    '''
    左上右下转化为左上宽高
    '''
    w = xyxy[2]-xyxy[0]
    h = xyxy[3]-xyxy[1]
    return [xyxy[0],xyxy[1],w,h]

def xyxy2xywh1(xyxy,wh):
    '''
    左上右下转化为中心点坐标和宽高
    '''
    iw,ih=wh
    x=(xyxy[0]+xyxy[2])/2/iw
    y=(xyxy[1]+xyxy[3])/2/ih
    w = xyxy[2]-xyxy[0]/iw
    h = xyxy[3]-xyxy[1]/ih
    return [x,y,w,h]

def read_json(json_name):
    json_path = os.path.join(file_path, json_name)
    with open(json_path,'r') as f:
        labelme = json.load(f)
    num = int(labelme['shape']/5)
    info =[]
    for i in range(num):
        inf = []
        point_dict = {}
        for shape in labelme['shape']:
            if shape['group_id']==i:
                if shape['label']=='face':
                    xyxy=[shape['points'][0][0],shape['points'][0][1],shape['points'][1][0],shape['points'][1][1]]
                    bbox = xyxy2xywh1(xyxy,[labelme['imageHeight'], labelme['imageWidth']])
                else:
                    if shape['label'] not in point_dict.keys():
                        point_dict[shape['label']] = shape['points'][0] + [1]
        point = point_dict['nose'] + point_dict['l_mouth'] + point_dict['r_mouth'] + point_dict['l_eye'] + point_dict['r_eye']
        area = bbox[2] * bbox[3]
        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        flag = True
        info.append([flag, [labelme['imagePath'], labelme['imageHeight'], labelme['imageWidth'], bbox, point, area, center]])
    return info


class imgattribute(object):
    def __init__(self,json_name):
        json_path = os.path.join(file_path,json_name)
        flag,info = read_json(json_path)
        self.flag = flag
        if flag:
            self.file_name,self.height,self.width,self.bbox,self.keypoints,self.area,self.center= info
        else:
            self.file_name, self.height, self.width, self.bbox, self.keypoints, self.area, self.center=None,None,None,[],[],None,[]

def loadinfo(json_list,file_path):
    id = 0
    ann_info = {'images':[],
                'annotations':[],
                'categories':[{"supercategory": "person", "id": 1, "name": "face", "keypoints": [], "skeleton": []}]}
    for json_path in tqdm(json_list,total=len(json_list)):
        json_path = os.path.join(file_path,json_path)
        flag,info = read_json(json_path)

    return ann_info

def writeinfo(save_path,info):
    with open(save_path,'w') as f:
        json.dump(info,f,indent=4,ensure_ascii=False)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_root', type=str, default='', help='face data json in dir')
    parser.add_argument('--txt_path', type=str, default='', help='10w data path txt path')
    parser.add_argument('--save_root', type=str, default='/data/FQA/data/yolopose_data', help='save data root path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')


    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def polt(imgpath,labelme,arg):
    img = cv2.imread(imgpath)
    for shape in labelme['shape']:
        if shape['shape_type'] == 'rectangle':
            cv2.rectangle(img, shape['points'][0], shape['points'][1], (255, 0, 0), 2)
        else:
            cv2.circle(img, shape['points'][0], 1, (255, 255, 0), -1)
    cv2.imwrite(arg.save_root+'/errorimg/'+imgpath.split('/')[-1], img)


def check_json(arg,json_list,face_path,save_img,save_json,img_list,):
    '''
    # 1.删选掉不满足条件的json文件
        # 1.删选掉没有目标的json图片
        # 2.删选掉关键点不足5个的标签，并画出图及路径
    '''

    for json_name in tqdm(json_list,total=len(json_list)):
        json_path = os.path.join(face_path,json_name)
        with open(json_path,'r') as f:
            labelme = json.load(f)
        if len(labelme['shape'])==0:
            os.remove(json_path)
        elif len(labelme['shape']) % 5!=0:
            print(json_name,' 关键点数量不足！')
            suffix = [x for x in img_list if json_name.split('.')[0] in x][0].split('.')[1]
            img_path = str(json_path.split('.')[0])+'.'+suffix
            polt(img_path,labelme)

            shutil.copy(img_path,arg.save_root+'/no_points/'+img_path.split('/')[-1])
            shutil.copy(json_path,arg.save_root+'/no_points/'+json_name)
            continue
        else:
            num = int(labelme['shape'] / 5)
            total_txt = ''
            for i in range(num):
                point_dict = {}
                for shape in labelme['shape']:
                    if shape['group_id'] == i:
                        if shape['label'] == 'face':
                            xyxy = [shape['points'][0][0], shape['points'][0][1], shape['points'][1][0],
                                    shape['points'][1][1]]
                            bbox = xyxy2xywh1(xyxy,[labelme['imageHeight'], labelme['imageWidth']])
                            bbox.insert(0,4)
                        else:
                            if shape['label'] not in point_dict.keys():
                                point_dict[shape['label']] = np.append(np.array(shape['points'][0])/np.array([[labelme['imageHeight'], labelme['imageWidth']]] + 2))
                point = point_dict['nose'].tolist() + point_dict['l_mouth'].tolist() + point_dict['r_mouth'].tolist() + point_dict['l_eye'].tolist() + \
                        point_dict['r_eye'].tolist()
                txt=''
                for i in bbox+point:
                    txt+=(str(i)+'/t')
                txt = txt[0:-1]+'n'
                total_txt+=txt
            with open(save_json,'w') as f:
                f.write(total_txt)
            shutil.copy(img_path, save_img + img_path.split('/')[-1])






if __name__ == '__main__':
    # 1.删选掉不满足条件的json文件
        # 1.删选掉没有目标的json图片
        # 2.删选掉关键点不足5个的标签，并画出图及路径
    # 2.将facejson写成yolopose格式
    # 3.将普通图片已经转化成txt标签的加入yoloposetxt文件中(除了face，及其他目标加入0.0 0.0 0.0)

    opt = parse_opt()
    # opt.txt_path
    face_img_path = os.path.join(opt.face_root,'images')
    save_data_img_path = os.path.join(opt.save_data_root,'images')
    save_data_lab_path = os.path.join(opt.save_root,'labels')
    os.makedirs(opt.save_root,exist_ok=True)
    os.makedirs(save_data_img_path, exist_ok=True)
    os.makedirs(save_data_lab_path, exist_ok=True)
    os.makedirs(opt.save_root+'/errorimg')
    os.makedirs(opt.save_root+'/no_points')


    # 1.删选掉不满足条件的json文件
        # 1.删选掉没有目标的json图片
    face_ij_list=os.listdir(face_img_path)
    face_json_list = [i for i in face_ij_list if i.split('.')[1]=='json']
    face_img_list = [i for i in face_ij_list if i.split('.')[1] != 'json']
    print('miss json: ',len(face_img_list)-len(face_json_list))
    check_json(opt,face_json_list,face_img_path,save_data_img_path,save_data_lab_path,face_img_list)























































    face_root_path = r'E:\face_dataset\test'
    file_path = os.path.join(root_path,'images')
    anno_path = os.path.join(root_path,'annotations')
    os.makedirs(anno_path,exist_ok=True)
    file_list = os.listdir(file_path)

    json_list = [i for i in file_list if i.split('.')[1] == 'json']
    img_list = [i for i in file_list if i.split('.')[1] != 'json']
    print('miss json: ',len(img_list)-len(json_list))
    comparison(json_list,img_list)
    random.shuffle(json_list)
    n = int(len(json_list)*0.8)
    train_list = random.sample(json_list,n)
    test_list = [i for i in json_list if i not in train_list]
    save_train = os.path.join(anno_path,'face_landmarks_wflw_train.json')
    save_test = os.path.join(anno_path, 'face_landmarks_wflw_test.json')
    info = loadinfo(train_list)
    writeinfo(save_train,info)
    info = loadinfo(test_list)
    writeinfo(save_test,info)









