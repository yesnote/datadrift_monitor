import numpy as np
import imutils
import json
from tqdm import tqdm
from pathlib import Path
import sys
import torch
import os
from Attack.dpatch_robust import RobustDPatch
from Attack.dpatch import DPatch
from Attack.art_object_detection import PyTorchFasterRCNN
# from dpatch_robust import RobustDPatch
# from dpatch import DPatch
# from art_object_detection import PyTorchFasterRCNN
from Config import configs
from Data.Data_util import coco_names,load_dataset_attack_format,yolo_preprocessing
from OD_models.Object_detector import Object_detection_model
import datetime

"""
This module is used to craft and apply adversarial attack against object detection models.
Includes: attack creation using the Dpatch and Lee & Kotler adversarial aatcks, apply a crafted patch on a scene, etc...
"""

class Attack_object_detector():
    def __init__(self, config):
        self.attack_params = config.attack_params
        self.model_params = config.model_params
        self.dataset_params = config.dataset_params
        self.od_model = Object_detection_model(model_params=self.model_params,
                                               decision_threshold=self.model_params['decision_threshold'],
                                               device=self.model_params['device'],
                                               img_size=self.dataset_params['img_size'],
                                               target_model_path = self.model_params['model_path'])


    def load_model(self):
        if self.model_params['model_algorithm'] == 'Faster_RCNN':
            frcnn = PyTorchFasterRCNN(
                clip_values=(0, 255),
                channels_first=False,
                attack_losses=self.attack_params['attack_losses'])
            return frcnn

    def rubust_dpatch_init(self,od_model):
        # if not patch_location is None:
        #     self.attack_params["patch_location"] = patch_location
        # if not patch_shape is None:
        #     self.attack_params["patch_shape"] = patch_shape
        attack_conf = RobustDPatch(
            od_model,
            patch_shape=self.attack_params["patch_shape"],
            patch_location=self.attack_params["patch_location"],
            crop_range=self.attack_params["crop_range"],
            brightness_range=self.attack_params["brightness_range"],
            rotation_weights=self.attack_params["rotation_weights"],
            sample_size=self.attack_params["sample_size"],
            learning_rate=self.attack_params["learning_rate"],
            max_iter=self.attack_params['max_iter'],
            batch_size=self.attack_params["batch_size"],
            targeted=self.attack_params['targeted'],
            source_class= self.attack_params['source_class'],
            target_class= self.attack_params['target_class'],
            verbose=self.attack_params['verbose'],
            patch_save_dir=self.attack_params['saved_patch_dir']
        )
        return attack_conf

    def dpatch_init(self,od_model):
        attack_conf = DPatch(
            od_model,
            patch_shape=self.attack_params["patch_shape"],
            learning_rate=self.attack_params["learning_rate"],
            max_iter=self.attack_params['max_iter'],
            batch_size=self.attack_params["batch_size"],
            patch_save_dir=self.attack_params['saved_patch_dir'],
            apply_patch_on_main_object = self.attack_params['apply_patch_on_main_object'],
            verbose=self.attack_params['verbose']
         )
        return attack_conf


    def predict_batch(self, model, image_dicts,mode='benign',dataset_mode='train'):
        pred_dicts = []
        images_id = [image_dict['id'] for image_dict in image_dicts]
        if mode == 'benign':
            images = [image_dict['image'] for image_dict in image_dicts]
        else:
            images = [image_dict['patch_image'] for image_dict in image_dicts]
        for id,image in tqdm(zip(images_id,images), desc="MS coco attack evaluation"):
        # for image_path in image_paths:
        #     image = load_image_in_art_format(image_path=image_path)
            images = np.expand_dims(image, 0)
            pred_dict = self.predict(model,images)
            pred_dicts.append(pred_dict[0])
            pred_dict = Object_detection_model.extract_prediction_faster_rcnn_dict_format(pred_dict,
                                                                                          threshold=0.1)
            # if dataset_mode=='train':
            #     output_path = self.attack_params['saved_train_clean_images_dir']
            # else:
            #     output_path = self.attack_params['saved_val_clean_images_dir']
            # if self.attack_params['plot_images_prediction']:
            #     plot_image_with_boxes(image,
            #                           boxes=pred_dict[0]['boxes'],
            #                           pred_cls=pred_dict[0]['labels'],
            #                           confidence=pred_dict[0]['scores'],
            #                           output_path=output_path,
            #                           image_id=id)
        return pred_dicts

    def predict(self,model,image):
        return model.predict(x=image, batch_size=1)

    @staticmethod
    def get_patch_location_faster_rcnn_format(bbox, patch):

        object_width = bbox[2] - bbox[0]
        object_height = bbox[3] - bbox[1]
        object_min_size = min(object_width,object_height)
        if object_min_size*0.8<patch.shape[1]:
            width = int(object_min_size*0.7)
            patch = imutils.resize(patch,width=width,height=width)

        object_x_middle = (bbox[2] + bbox[0]) / 2
        object_y_middle = (bbox[3] + bbox[1]) / 2
        x_patch_location = object_x_middle - (patch.shape[0] / 2)
        y_patch_location = object_y_middle - (patch.shape[1] / 2)
        return (int(x_patch_location), int(y_patch_location)), patch

    @staticmethod
    def apply_patch_to_single_image(image, patch, patch_location):
        image_with_patch = image.copy()
        image_with_patch = np.squeeze(image_with_patch)
        image_with_patch = np.transpose(image_with_patch, (1, 2, 0))

        patch_local = patch.copy()
        # Apply patch:
        x_1, y_1 = patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]
        if x_2 > image_with_patch.shape[1] or y_2 > image_with_patch.shape[0]:  # pragma: no cover
            raise ValueError("The patch (partially) lies outside the image.")
        try:
            image_with_patch[y_1:y_2, x_1:x_2, :] = patch_local
            # np.transpose(patch_local(0,1,2))
        except:
            print('Patch assertion failed')
            x_2,y_2 = Attack_object_detector.adjust_patch_location(x_1, x_2, y_1, y_2, patch_local)
            image_with_patch[y_1:y_2, x_1:x_2, :] = patch_local
        image_with_patch = np.transpose(image_with_patch, (2, 0, 1))
        image_with_patch = np.expand_dims(image_with_patch,axis=0)
        return torch.tensor(image_with_patch)

    @staticmethod
    def adjust_patch_location(x_1,x_2,y_1,y_2,patch_local):
        if x_2 - x_1 != patch_local.shape[1]:
            x_2 += 1
            if x_2 - x_1 != patch_local.shape[1]:
                x_2  = x_2-2
        if y_2-y_1 != patch_local.shape[0]:
            y_2+=1
            if y_2 - y_1 != patch_local.shape[0]:
                y_2 = y_2-2
        return x_2,y_2
    def place_patch_on_image(self, preds, images):
        patched_images = self.generate_patch_images(preds, self.patch, images)
        return patched_images

    def generate_patch_images(self, preds, patch,images,dataset_mode='val'):
        patch_images = []
        patch = patch/255
        counter = 0
        for image,pred in tqdm(zip(images,preds), desc="Put patch on images"):
            counter+=1
            # print(counter)
            bbox = self.select_object_to_attack(pred)
            try:
                patch_loc, new_patch = self.get_patch_location_faster_rcnn_format(bbox, patch)
                # patch_loc, new_patch = self.get_patch_location_faster_rcnn_format([50, 50, 300, 300], patch)
            except:
                print("patch custom apply failed")
                patch_loc, new_patch = self.get_patch_location_faster_rcnn_format([200,200,350,350], patch)
            try:
                patched_image = self.apply_patch_to_single_image(image.detach().cpu().numpy(), new_patch, patch_loc)
            except:
                print("patch custom apply failed again")
                patched_image = image
            patch_images.append(patched_image)
            # patch_images[image_path.split('/')[-1].split('.')[0]] = patched_image
        return patch_images

    def select_object_to_attack(self,pred):
        for bbox in pred['boxes']:
            object_width = bbox[2] - bbox[0]
            object_height = bbox[3] - bbox[1]
            max_size = max(object_width,object_height)
            if max_size>120:
                return bbox




    def evaluate_predictions(self, dataset,predictions):
        image_ids = [image_dict['id'] for image_dict in dataset.image_dict]
        preds_list = self.process_preds_for_eval(predictions,image_ids)
        coco_detections = dataset.coco.loadRes(preds_list)
        dataset.eval(coco_detections,'bbox',list(range(0, len(coco_names))))

    def process_preds_for_eval(self,predictions,image_ids):
        preds_list = []
        for id,preds in tqdm(zip(image_ids,predictions), desc="Processing predictions for evaluation"):
            for i in range(len(preds['boxes'])):
                if float(preds['scores'][i]) > self.model_params['decision_threshold']:
                    pred_dict = {}
                    pred_dict['image_id'] = id
                    pred_dict['category_id'] = int(preds['classes'][i])
                    pred_dict['bbox'] = self.transform_bbox_to_yolo_format(preds['boxes'][i])
                    pred_dict['score'] = preds['scores'][i]
                    preds_list.append(pred_dict)

        return preds_list

    def transform_bbox_to_yolo_format(self, bbox_list):
        x1 = bbox_list[0]
        y1 = bbox_list[1]
        w = bbox_list[2] - x1
        h = bbox_list[3] - y1
        return [x1, y1, w, h]

    def change_prediction_dict_by_target(self, pred_dicts,source_class,target_class, keep_only_main_object = True):
        updated_pred_dicts = []
        for pred_dict in pred_dicts:
            if keep_only_main_object:
                max_confidence_of_specific_class = self.find_max_confidence_of_specific_class(pred_dict=pred_dict,
                                                                                              source_class=source_class)
                pred_dict['boxes'] = np.expand_dims(pred_dict['boxes'][max_confidence_of_specific_class],axis=0)
                pred_dict['labels'] =  np.expand_dims(pred_dict['labels'][max_confidence_of_specific_class],axis=0)
                pred_dict['labels'][0] = target_class
                pred_dict['scores'] = np.expand_dims(pred_dict['scores'][:max_confidence_of_specific_class],axis=0)
            else:
                temp_labels = []
                for i in range(len(pred_dict['labels'])):
                    temp_labels.append(target_class)
                pred_dict['labels'] = np.array(temp_labels)
            updated_pred_dicts.append(pred_dict)
        return updated_pred_dicts

    @staticmethod
    def find_max_confidence_of_specific_class(pred_dict, source_class):
        source_class_idx = np.where(pred_dict['labels']==source_class)[0]
        max_confidence_idx_of_source_class = np.argmax(pred_dict['scores'][source_class_idx])
        return source_class_idx[max_confidence_idx_of_source_class]

    def find_max_size_of_specific_class(self, pred_dict, source_class):
        source_class_idx = np.where(pred_dict['labels'] == source_class)[0]
        sizes = self.get_bbox_sizes(pred_dict,source_class_idx)
        max_size_idx_of_source_class = source_class_idx[np.argmax(sizes)]
        return max_size_idx_of_source_class

    def get_bbox_sizes(self,pred_dict,idx_list):
        bboxes = pred_dict['boxes']
        sizes = np.array([(box[2]-box[0])*(box[3]-box[1]) for box in bboxes[idx_list]])
        max_size_idx = np.argmax(sizes)
        return idx_list[max_size_idx]

    def dump_configs(self):
        with open(f'{self.attack_params["saved_patch_dir"]}/attacks_configs.json', 'w') as fp:
            json.dump(self.attack_params, fp)

    def load_patch(self):
        patch = np.load(self.attack_params['patch_path'])
        return patch

    def digital_attack_train_demo_COCO(self):

        od_model = self.load_model()
        self.dump_configs()
        train_image_paths = self.dataset_params['attack_training_set_path']
        images, file_names = load_dataset_attack_format(os.path.join(train_image_paths, 'images'),
                                                               self.model_params['model_algorithm'])
        prediction_dicts = self.od_model.predict_wrapper(image_dataloader=images,
                                                                detection_threshold=self.model_params[
                                                                    'decision_threshold'],
                                                                use_grad=False)
        images = [self.convert_torch_to_numpy(image) for image in images]
        prediction_dicts = [self.prepare_predicitons_for_attack(prediction_dict) for prediction_dict in prediction_dicts]
        attack_conf = self.dpatch_init(od_model = od_model)
        self.patch = attack_conf.generate(x=np.array(images),
                                     y=prediction_dicts)


    def digital_attack_train_demo_SuperStore(self):

        self.init_experiment_folder()
        self.dump_configs()
        images = load_dataset_attack_format(self.dataset_params['attack_training_set_path'])

        prediction_dicts = []
        for image in images:
            prediction_dicts.append(self.od_model.predict(image,None,None)[0][0])
        #
        # prediction_dicts = self.od_model.predict_wrapper(image_dataloader=images, saliency_maps=None,
        #                                              DiL_scores=None, use_grad=False,
        #                                             detection_threshold = self.model_params['decision_threshold'])
        frcnn = PyTorchFasterRCNN(self.od_model.model, clip_values=(0, 255),
                                  channels_first=False,
                                  attack_losses=self.attack_params['attack_losses'])
        attack_conf = self.dpatch_init(od_model=frcnn)
        images = torch.cat(images)
        images = torch.permute(images, (0, 2, 3, 1))
        np_images = images.detach().cpu().numpy()*255
        self.patch = attack_conf.generate(x=np_images,
                                          y=prediction_dicts)
        # self.digital_attack_evaluate_demo(resume=True)

    def init_experiment_folder(self):

        """
        Function for crating output folders.
        :return: Create output folders.
        """
        time = datetime.datetime.now().strftime("%d-%m-%Y_%H;%M")
        output_path = f"{self.attack_params['experiment_folder']}/{time}"
        Path(output_path).mkdir(parents=True, exist_ok=True)

        folder_param = {
            'saved_patch_dir':'patch',
            'saved_val_clean_images_dir': 'val_clean_images',
            'saved_val_patched_images_dir':'val_patched_images',
            'saved_train_clean_images_dir':'train_clean_images',
            'saved_train_patched_images_dir':'train_patched_images',
            'saved_val_dynamic_images_dir':'val_dynamic_images'
        }
        [self.create_folder(output_path=output_path,dir_name=dir_name,param_name=param_name) for param_name,dir_name in folder_param.items()]

    def create_folder(self, output_path, dir_name, param_name):
        curr_dir = f'{output_path}/{dir_name}'
        Path(curr_dir).mkdir(parents=True, exist_ok=True)
        self.attack_params[param_name] = curr_dir


    def apply_digital_attack(self):

        self.patch = self.load_patch()
        self.patch = imutils.resize(self.patch,width=150)
        images,file_names = load_dataset_attack_format(self.dataset_params['clean_validation_set_path']+'/images',algorithm_name = self.model_params['model_algorithm'])

        benign_prediction_dicts = self.od_model.predict_wrapper(image_dataloader=images,
                                                                detection_threshold=self.model_params['decision_threshold'],
                                                                use_grad=False)
        patched_images = self.place_patch_on_image(benign_prediction_dicts, images)
        print('Predict patched scenes:')
        adversarial_prediction_dicts = self.od_model.predict_wrapper(image_dataloader=patched_images,
                                                                     detection_threshold=self.model_params['decision_threshold'],
                                                                     use_grad=False)
        return images,patched_images,benign_prediction_dicts,adversarial_prediction_dicts,file_names


    def processed_image_for_yolov5_model(self, images):
        return [yolo_preprocessing(image) for image in images]

    def convert_torch_to_numpy(self, image):
        image = image.detach().cpu().numpy()*255
        image = image.squeeze(0)
        image = np.transpose(image,(1,2,0))
        return image

    def prepare_predicitons_for_attack(self, prediction_dict):
        prediction_dict['labels'] = np.array([self.convert_label_to_idx(label) for label in prediction_dict['labels']])
        return prediction_dict

    def convert_label_to_idx(self, label):
        label = coco_names.index(label)
        return label


if __name__ == "__main__":
    config = configs()
    attack_module = Attack_object_detector(config)
    # attack_module.digital_attack_train_demo()
    attack_module.digital_attack_train_demo_COCO()
    # attack_module.digital_attack_train_demo_SuperStore()
    # attack_module.digital_attack_train_demo_COCO()
    # images,patched_images,benign_prediction_dicts,adversarial_prediction_dicts = attack_module.apply_digital_attack()
    # print("Benign evaluation results:")
    # attack_module.digital_attack_evaluate_demo(images,benign_prediction_dicts)
    # print('Adversarial evaluation results:')
    # attack_module.digital_attack_evaluate_demo(patched_images,adversarial_prediction_dicts,mode='Adversarial')

