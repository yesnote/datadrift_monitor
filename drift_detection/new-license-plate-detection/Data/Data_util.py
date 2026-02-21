import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import os
import cv2
import json
import torch
import random
from OD_models.yolov5.models.yolo_v5_object_detector import YOLOV5TorchObjectDetector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.io as io
import seaborn as sns
"""
This module is used for upload, explore and evaluate datasets.
"""
id = 1600
coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

mmdet_coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

SuperStore_classes =['__background__','Agnesi Polenta','Almond Milk','Snyders','Calvin Klein','Dr Pepper','Flour',
'Groats','Jack Daniels','Nespresso','Oil','Paco Rabanne','Pixel4','Samsung_s20','Greek Olives','Curry Spice',
'Chablis Wine','Lindor','Piling Sabon','Tea','Versace']

SuperStore_classes_YOLO =['Jack Daniels','Nespresso','Agnesi Polenta','Snyders','Calvin Klein',
                          'Dr Pepper','Flour','Pixel4','Groats','Almond Milk','Oil','Paco Rabanne','Samsung_s20',
                          'Greek Olives','Chablis Wine','Lindor','Piling Sabon','Tea','Versace','Curry Spice']
def load_dataset(dataset_path, model_algorithm='Faster_RCNN', batch_size=1):
    """
    Function that load image dataset
    :param dataset_path: required. str. The path to the image folder.
    :param batch_size: optional. int. batch size.
    :return: numpy array containing the images (in numpy format).
    """
    images_paths =[os.path.join(dataset_path,img_file) for img_file in os.listdir(dataset_path)]
    image_dataset = ImageDataset(images_paths=images_paths, backbone_name=model_algorithm)
    return DataLoader(image_dataset,batch_size=batch_size,num_workers=6,pin_memory=True)

def load_dataset_attack_format(dataset_path,algorithm_name):
    images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
    images_paths = sorted(images_paths)
    images = [load_image(image,algorithm_name) for image in images_paths]
    file_names = [os.path.splitext(os.path.basename(image_path))[0] for image_path in images_paths]
    return images,file_names

def load_images(dataset_path):
    """
    Function that load images from a given path.
    :param dataset_path:  required. str. The path to the image folder.
    :return: numpy array containing the images (in numpy format).
    """
    images = []
    for file in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path,file)
        image = load_image(image_path)
        images.append(image)
    return images

def load_images_yolo(dataset_path):
    images_paths = [os.path.join(dataset_path, img_file) for img_file in os.listdir(dataset_path)]
    images_paths = sorted(images_paths)
    images = [cv2.imread(image) for image in images_paths]
    processed_images = [ yolo_preprocessing(image) for image in images]
    return processed_images

def image_preprocess(image,algorithm_name):
    if algorithm_name == 'YOLOv5':
        image = yolo_preprocessing(image)
    else:
        image = faster_rcnn_preprocessing(image)
    return image


def yolo_preprocessing(img):
    if len(img.shape) != 4:
        img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)
    img = np.array(YOLOV5TorchObjectDetector.yolo_resize(img.squeeze(0), new_shape=(640, 640))[0])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img / 255.0
    img = img.unsqueeze(0)
    return img

def faster_rcnn_preprocessing(img):
    image_float_np = np.float32(img) / 255
    transform = transfomations()
    image = transform(image_float_np)
    image = image.unsqueeze(0)
    return image

def faster_rcnn_preprocessing_old_version(img):
    img = img.astype(np.uint8)
    img = np.array(YOLOV5TorchObjectDetector.yolo_resize(img, new_shape=(640, 640))[0])
    img = img.transpose((2, 0, 1))
    image_float_np = np.float32(img) / 255
    image = torch.from_numpy(image_float_np)
    return image

def load_image(image,algorithm_name):
    """
    Function that load 1 image from a given path.
    :param image_path: required. str. The path to the image file.
    :return: The image in numpy format.
    """
    if isinstance(image, str):
        image = np.array(Image.open(image))
    return image_preprocess(image,algorithm_name)


def process_image(image):
    image_float_np = np.float32(image) / 255
    transform = transfomations()
    image = transform(image_float_np)
    image = image.unsqueeze(0)
    return image

def process_image_for_video_framework(image):
    image_float_np = np.float32(image) / 255
    image_transpose = np.transpose(image_float_np, (2, 0, 1))
    image = torch.from_numpy(image_transpose)
    image = image.unsqueeze(0)
    return image

def process_image_for_real_time_pipeline(image):
    image_float_np = np.float32(image) / 255
    image_transpose = np.transpose(image_float_np,(0,3,1,2))
    image = torch.from_numpy(image_transpose)
    return image

def transfomations():
    """
    Transformation function that process the image.
    :return:
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform


def get_image_paths(dataset_path, image_width=640, image_height=640):
    """
    Load images from the given path list to the RAM.
    :param images_path: Required. list of strings describing each image path.
    :param image_width: Optional. int. The image width.
    :param image_height: Optional. int. The image height.
    :return: A list of images (4D numpy array).
    """
    images = []
    for image in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path,image)
        images.append(image_path)
    # Stack images
    images = np.array(sorted(images))
    return images

def load_image_in_art_format(image_path, image_width=640, image_height=640):
    """
    Load images from the given path list to the RAM.
    :param images_path: Required. list of strings describing each image path.
    :param image_width: Optional. int. The image width.
    :param image_height: Optional. int. The image height.
    :return: A list of images (4D numpy array).
    """

    image = io.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # resize the images to the predefined size.
    # image = np.transpose(image, (1, 2, 0))
    # image = cv2.resize(image,
    #                      dsize=(image_width, image_height),
    #                      interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    return image

def plot_image_with_boxes(img, boxes, pred_cls,confidence, output_path, image_id,save=True):
    """
    A function that plot the prediction on the input scene and saved it on
    a given output path.
    :param img: Required. 3D Numpy array. The input scene.
    :param boxes: Required. list of bounding boxes. Each bounding box is a list
    with 4 coordinated in Faster RCNN
    format (
    x1,y1,x2,y2).
    :param pred_cls: Required. List of strings represent the classification
    of the corresponding object.
    :param output_path: Required. String of the output path to save the plot.
    :param image_id: An id of the given img.
    :return: Saved the input image with the prediction in the given output
    path.
    """
    text_size = 0.6
    text_th = 2
    rect_th = 6
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0),
              (128, 0, 128), (255, 105, 180), (0, 255, 0)]

    for i in range(len(boxes)):
        # img = img*255
        boxes_as_int = [int(coordinate) for coordinate in boxes[i]]
        start_point = (boxes_as_int[0],boxes_as_int[1])
        end_point = (boxes_as_int[2],boxes_as_int[3])
        # Draw Rectangle with the coordinates
        color_idx = random.randint(0, len(colors)-1)
        cv2.rectangle(img, start_point, end_point, color=colors[color_idx], thickness=rect_th)

        # Write the prediction class

        curr_confidence = "%.2f" % confidence[i]
        prediction_text = f'{pred_cls[i]} {curr_confidence}'
        start_point = (boxes_as_int[0]+5,boxes_as_int[1]+20)
        cv2.putText(img, prediction_text, start_point, cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0, 150, 255), thickness=text_th)
    if save:
        image_id = os.path.splitext(os.path.basename(str(image_id)))[0]
        fig = plt.figure(figsize=(10, 7))
        plt.axis("off")
        plt.imshow(img, interpolation="nearest")
        # plt.show()
        plt.savefig(f'{output_path}/{image_id}.jpg', dpi=fig.dpi)
    else:
        return img

def process_prediction(images,prediction_dicts,original_images):
    process_prediction_dicts = []
    for image,prediction_dict,original_image in zip(images,prediction_dicts,original_images):
        image_height, image_width = image.size(2),image.size(3)
        original_image_height,original_image_width  = (original_image['height'],original_image['width'])
        width_diff = original_image_width/image_width
        height_diff = original_image_height/image_height
        process_prediction_dict = process_bbox(prediction_dict,width_diff,height_diff)
        process_prediction_dicts.append(process_prediction_dict)
    return prediction_dicts

def process_bbox(prediction_dict,width_diff,height_diff):
        process_boxes = []
        for box in prediction_dict['boxes']:
            x_1 = int(np.round(box[0]*width_diff))
            x_2 = int(np.round(box[2]*width_diff))
            y_1 = int(np.round(box[1]*height_diff))
            y_2 = int(np.round(box[3]*height_diff))
            # x_c = (x_1+x_2)/2
            # y_c = (y_1+y_2)/2
            # w = x_2-x_1
            # h = y_2-y_1
            process_boxes.append([x_1,y_1,x_2,y_2])
        prediction_dict['boxes'] = process_boxes
        return prediction_dict

class ImageDataset(Dataset):
    """
    This class represent image dataset object. Will be useful when using Dataloaders.
    """
    def __init__(self, images_paths,backbone_name):
        self.images_paths = images_paths
        self.backbone_name = backbone_name
    def __len__(self):
        return len(self.images_paths)
    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = image_preprocess(image, self.backbone_name)
        return image


    def transfomations(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        return transform



class MS_COCO_util():
    """
    Class representing MSCOCO util object. Create an instance of this class to used functionality on MSCOCO dataset.
    """

    def __init__(self,dataset_path,annotations_file_path,images_path):
        """
        Args:
            dataset_path: required. str. path to MSCOCO dataset (folders of images).
            annotations_file_path: req. str. path to MSCOCO annotations file.
            ids_list_path: req. str. path to txt file containing specific image ids to be loaded.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.annotations_file_path = annotations_file_path
        self.coco = self.create_coco_object()
        # self.ids_list = self.set_ids_list(ids_list_path)
        self.ids_list = self.set_ids_list(images_path)
        self.image_dict = self.load_coco_dataset_from_ids_list()

    def create_coco_object(self):
        """
        Create an COCO instance using pycocotools.
        Returns: COCO instance.
        """
        return COCO(self.annotations_file_path)

    def set_ids_list(self, images_path):
        """
        Open ids file to ids list.
        Args:
            ids_list_path: req. str. path to txt file containing specific image ids to be loaded.
        Returns: image ids as a list instance.

        """
        return [int(image_file.split('/')[-1].split('.')[0]) for image_file in images_path]



    def get_imgs_dict_by_id_list(self):
        """
        Get image data as dictionary from list of ids.
        Returns: image data as dictionary.
        """
        imgIds = self.coco.getImgIds(imgIds=self.ids_list)
        imgIds = sorted(imgIds)
        imgIds = self.coco.loadImgs(imgIds)
        return imgIds

    def get_img_by_img_dict(self,img_dict):
        """
        Get image as numpy array from an image dict instance.
        Args:
            img_dict: req. dict of information about a specific image.
        Returns: image as numpy array
        """
        return io.imread(img_dict['coco_url'])

    def get_img_annotations_by_img_dict(self, img_dict):
        """
        use COCO instance to load dictionary representation the annotations of a specific image
        (segmentation, bounding box etc.)
        Args:
            img_dict:  req. dict of information about a specific image.

        Returns: Dictionary representation the annotations of a specific image
        """
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=[],
                                     iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        return anns

    def plot_image_bbox(self,img_dict,catIds):
        """
        Plot the image with its corresponding bounding boxes of each detected object.
        Args:
            img_dict: req. dict of information about a specific image.
            catIds: category ids.
        Returns: A plot of the image with its corresponding bounding boxes of each detected object.

        """
        I = self.get_img_by_img_dict(img_dict)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(I)
        plt.axis('off')
        annIds = self.coco.getAnnIds(imgIds=img_dict['id'], catIds=catIds,
                                   iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        for i, ann in enumerate(anns):
            plt.gca().add_patch(Rectangle((anns[i]['bbox'][0], anns[i][
                'bbox'][1]), anns[i]['bbox'][2], anns[i]['bbox'][3],
                         edgecolor = 'green',facecolor='none'))
            ax.text(anns[i]['bbox'][0], anns[i]['bbox'][1],
                    coco_names[anns[i]['category_id']],
                                                 style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
        plt.show()

    def label_list_to_indexes(self,label_list):
        """
        Transform labels to indexing using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            label_list: req. list of labels (strings).

        Returns: list of indexes in COCO format.
        """
        return [self.label_to_index(label) for label in label_list]

    def label_to_index(self,label):
        """
        Transform label to index using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            label: req. label (str).
        Returns: index in COCO format.

        """
        return coco_names.index(label)

    def index_to_label(self,index):
        """
        Transform index to label using COCO_INSTANCE_CATEGORY_NAMES.
        Args:
            index: req. int.
        Returns: Corresponding label of the given index in COCO format.
        """
        return coco_names[index]

    def display_coco_categories(self):
        """
        Display COCO categories and super-categories
        Returns: COCO categories and super-categories.
        """
        #
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

    def load_coco_dataset_from_ids_list(self):
        """
        Loads coco dataset from ids list.
        Returns: images dictionary.
        """
        images_dict = self.get_imgs_dict_by_id_list()
        for img_dict in images_dict:
            img_dict['annotations'] = self.get_img_annotations_by_img_dict(img_dict)
            img_dict['image'] = load_image_in_art_format(img_dict['coco_url'])

        return images_dict

    def eval(self,cocoDt,annType,cat_ids):
        """
        Evaluate object detector predictions.
        Args:
            cocoDt: req. coco detections.
            annType: req. str. "bbox" or "segmentations".
            cat_ids: req. list of categories ids.
        Returns: evaluation report produced by pycocotools.
        """
        coco_eval = COCOeval(self.coco,cocoDt,annType)
        imgIds = sorted(self.coco.getImgIds(imgIds=self.ids_list))
        coco_eval.params.imgIds = imgIds
        coco_eval.params.catIds = cat_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def transformations(self):
        """
        Transformation function for the given dataset (mainly transform from
        numpy to torch and normalize).
        :return: Transformations function.
        """
        return ComposeSingle([
            FunctionWrapperSingle(np.moveaxis, source=-1, destination=0),
            FunctionWrapperSingle(normalize_01)
        ])

    def coco_to_dataloader(self, inputs, batch_size=1):
        """
        Form a pytorch dataloader instance for real time detection.
        :param inputs: req. numpy array representing an image.
        :param batch_size: optinal. tha number of images in a batch (default
        is 1).
        :return: pytorch dataloader instance.
        """

        # transformations
        transforms = self.transformations()

        # create dataset and dataloader
        dataset = ObjectDetectionDatasetSingleFromNumpy(inputs=inputs,
                                                        transform=transforms,
                                                        use_cache=False,
                                                        )

        dataloader_prediction = DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0)
        return dataloader_prediction

    def image_dict_to_json(self,images_dict,output_path):
        """
        Transform images dictionary to Json file.
        Args:
            images_dict: req. dict of information about a COCO images.
            output_path: req. str. path to save the json file.

        Returns: Save the json file in the given path.
        """
        images = []
        categories = []
        annotations = []
        for image_dict in images_dict:
            images.append(self.process_image(image_dict,output_path))
            for idx,annotation in enumerate(image_dict['annotations']):
                annotations.append(self.process_annotations(image_dict,annotation))
        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations


    def process_image(self,image_dict,output_path):
        """
        process image to mscoco format.
        Args:
            image_dict: req. dict of information about a specific COCO image.
            output_path: req. str. path to save the image file.
        Returns: dictionary in mscoco format.
        """
        image = {}
        image["height"] = image_dict['height']
        image["width"] = image_dict['width']
        image["id"] = image_dict['id']
        image["file_name"] = image_dict['file_name']
        output_image = cv2.cvtColor(image_dict['image'],cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{output_path}/{image_dict['id']}.jpeg",output_image)
        return image

    def process_annotations(self,image_dict,curr_annotation):
        """
         process annotations to mscoco format.
        Args:
            image_dict: req. dict of information about a specific COCO image.
            curr_annotation: req. dict of annotations about a specific COCO image.
        Returns: dictionary of annotations in mscoco format.

        """
        annotation = {}
        area = image_dict['width'] * image_dict['height']
        annotation["segmentation"] = []
        annotation["iscrowd"] = curr_annotation['iscrowd']
        annotation["area"] = float(area)
        annotation["image_id"] = curr_annotation['image_id']
        annotation["bbox"] = curr_annotation['bbox']
        annotation["category_id"] = curr_annotation['category_id']
        annotation["id"] = curr_annotation['id']
        return annotation

    def get_ids_from_specific_class(self, classes):
        """
        Function that return instances from specific class.
        Args:
            classes: req. list of COCO classes (strings).
        Returns: COCO image id list selected from the given classes list.
        """
        ids_list = []
        for class_name in classes:
            catIds = self.coco.getCatIds(catNms=class_name)
            imgIds = self.coco.getImgIds(catIds=catIds)
            imgIds = imgIds[:300]
            ids_list.append(imgIds)
        self.ids_list = [item for sublist in ids_list for item in sublist]

    def transform_ground_truth_to_dict(self):
        ground_truth_dicts = []
        for image_dict in self.image_dict:
            boxes = np.array([self.xywh2x1y1x2y2(annotation['bbox']) for annotation in image_dict['annotations']])
            labels = np.array([annotation['category_id'] for annotation in image_dict['annotations']])
            scores = np.ones(len(labels))
            ground_truth_dict = {
               'boxes': boxes,
               'labels': labels,
                'scores':scores
            }
            ground_truth_dicts.append(ground_truth_dict)
        return ground_truth_dicts

    def xywh2x1y1x2y2(self,bbox):
        x_center = bbox[0]
        y_center = bbox[1]
        width = bbox[2]
        hight = bbox[3]
        x1 = x_center-width/2
        x2 = x_center+width/2
        y1 = y_center-hight/2
        y2 = y_center+hight/2
        return np.array([x1,y1,x2,y2])

def transform_to_yolo_format(bbox):
    x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
    image_width= 640
    image_height = 360
    # Calculate the center coordinates
    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)

    # Calculate the width and height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return [round(x_center,3), round(y_center,3), round(width,3), round(height,3)]

def edit_dataset(input_path,output_path):
    for label_file_path in os.listdir(input_path):
        f = open(os.path.join(input_path,label_file_path), "r")
        data = json.loads(f.read())
        boxes = transform_to_yolo_format(data['boxes'][0])
        label_file_path = label_file_path.split('.')[0]
        text_file = open(f"{os.path.join(output_path,label_file_path)}.txt", "w")
        n = text_file.write(f'{13} {boxes[0]} {boxes[1]} {boxes[2]} {boxes[3]}')
        text_file.close()
        f.close()

def plot_image(input_image,input_labels):
    cv2.imread(input_image)


def check_duplicates(dir1, dir2):
    dir1_files = set(os.listdir(dir1))
    dir2_files = set(os.listdir(dir2))

    unique_files = dir1_files - dir2_files

    for file in unique_files:
        print(file)

def change_labels(labels_file_path):

    original_classes = ['Jack Daniels','Nespresso','Agnesi Polenta','Snyders','Calvin Klein','Dr Pepper','Flour','Pixel4','Groats','Almond Milk','Oil',
                        'Paco Rabanne','Samsung_s20','Greek Olives','Chablis Wine','Lindor','Piling Soap','Tea','Versace','Curry Spice']
    new_classes = ['Agnesi Polenta','Almond Milk','Calvin Klein','Chablis Wine','Curry Spice','Dr Pepper','Flour','Greek Olives','Groats','Jack Daniels','Lindor','Nespresso','Oil',
                    'Paco Rabanne','Piling Soap','Pixel4','Samsung_s20','Snyders','Tea','Versace']

    for label_file_path in os.listdir(labels_file_path):
        label_file = os.path.join(labels_file_path,label_file_path)
        f = open(label_file, "r")
        lines = f.readlines()
        f.close()
        new_labels = []
        for line in lines:
            label = line.split(' ')
            class_label = label[0]
            class_label = new_classes[int(class_label)]
            transform_class_label = original_classes.index(class_label)
            new_labels.append(f'{transform_class_label} {label[1]} {label[2]} {label[3]} {label[4]}')
        with open(label_file, 'w') as file:
            for line,new_label in zip(lines,new_labels):
                file.write(new_label)

def print_files_name(path):
    for file_path in os.listdir(path):
        print(file_path.split(".")[0])


def create_dict(idx,file_name):
    return {
            "height": 360,
            "width": 640,
            "id": idx+1600,
            "file_name": file_name
            }

def xcycwh2x1y1wh(anno):
    xc,yc,w,h = anno
    h= float(h.replace("\n", ""))
    xc,yc,w = float(xc),float(yc),float(w)
    x1 = (xc-(w/2))*640
    y1 = (yc - (h / 2))*360
    w = w*640
    h = h*360
    return x1,y1,w,h

def create_anno_dict(idx,line,id):
    anno = line.split(" ")
    category_id = anno[0]
    x1,y1,w,h = xcycwh2x1y1wh(anno[1:])
    id= id+1
    return {
        "segmentation": [],
        "iscrowd": 0,
        "area": 230400.0,
        "image_id": 1600+idx,
        "bbox": [
            x1,
            y1,
            w,
            h
        ],
        "category_id": int(SuperStore_classes.index(SuperStore_classes_YOLO[int(category_id)])),
        "id": id
    }




def data_disturbution(array,mode):
    np_array = np.array(array)
    thresholds = np.linspace(10, 100, 10)
    percentiles = [np.percentile(np_array, thr, axis=0) for thr in thresholds]
    counts = []
    print('check')
    for threshold in percentiles:
      if mode=='TN':
        count = np.sum(array <= threshold)
      else:
        count = np.sum(array > threshold)
      counts.append(count)
    # Convert counts to percentages
    total = 800
    percentages = [count / total * 100 for count in counts]
    print(percentages)
    cumulative_percentages = np.cumsum(percentages)
    np_array = np.array(array)

    percentiles = [round(threshold,5) for threshold in percentiles]
    # Plotting
    sns.set_style("darkgrid", {'font.family':'serif', 'font.serif':'Times New Roman'})
    plt.figure(figsize=(10,6))
    # percentages[1] = percentages[1]+0.1
    percentiles[1] = 0.0001
    sns.barplot(x=percentiles, y=percentages, color='skyblue', edgecolor='black')
    plt.title(f'Clean {mode} per DiL value', fontsize=15)
    plt.xlabel('DiL value', fontsize=12)
    plt.ylim((0, 100) )
    plt.ylabel('CLean Data Percentage', fontsize=12)
    plt.show()

def upload_annotation_file(path):
    with open(path) as json_file:
        data = json.load(json_file)
    original_image_ids = [image['file_name'].split('.')[0].split('-')[1] for image in data['images']]
    for idx in range(len(data['annotations'])):
        image_id = data['annotations'][idx]['image_id']
        x1,y1,width,height = data['annotations'][idx]['bbox']
        x2,y2 = x1+width, y1+height
        data['annotations'][idx]['bbox'] = [x1,y1,x2,y2]
        anno_category = data['categories'][data['annotations'][idx]['category_id']]['name']
        # data['annotations'][idx]['category_id'] = SuperStore_classes.index(anno_category)
        data['annotations'][idx]['category_id'] = coco_names.index(anno_category)
        data['annotations'][idx]['original_image_id'] = original_image_ids[image_id]


