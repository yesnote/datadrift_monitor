import os.path

import torch
from OD_models.Object_detector import Object_detection_model

class configs():
    """
    Config module -- contains all configuration setting for DiL project.
    Includes: the model type, dataset type, XAI settings, attack settings etc...
    """

    def __init__(self,model_dataset='COCO'):

        Object_detection_model.init_seeds()

        self.model_params = {
            # the model algorithm that will be used Faster_RCNN or YOLOv5 or MMDetection
            'model_algorithm': 'Faster_RCNN',
            # the dataset that the model was trained on : COCO or SuperStore
            'model_dataset': model_dataset,

        }
        # decision threshold works best with : COCO: 0.8 to Faster RCNN and 0.5 to YOLOv5, SuperStore: 0.7 to faster RCNN and 0.7 to YOLOv5
        self.init_decision_thresholds()
        # The number of classes of each dataset
        self.model_params['num_of_classes'] = 20 if self.model_params['model_dataset']=='SuperStore' else 80
        # get pretrained models for each dataset
        self.COCO_models,self.Superstore_models = self.init_model_zoo()
        # select the specific model to use in the current experiment from: double_heads_faster_rcnn, grid_rcnn, yolox_l,cascade_rcnn_r101_fpn,yolo_f,yolo_v3,cascade_rpn_r50_fpn
        self.COCO_model = self.COCO_models['grid_rcnn']
        # select the specific model to use in the current experiment from: faster_rcnn_seed_42,grid_rcnn_r50_fpn,double_head_rcnn_r50_fpn,yolov5f,yolox_x,cascade_rpn,yolo_f,cascade_rcnn, yolo_v3
        self.SuperStore_model = self.Superstore_models['faster_rcnn_seed_42']
        self.model_params['model_path'] = self.SuperStore_model if self.model_params['model_dataset']=='SuperStore' else self.COCO_model

        # Use GPU or CPU (cuda or cpu)
        self.model_params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #  Root data folder - change the root dataset folder
        self.dataset_root_folder = 'datasets'
        self.COCO_dataset_params = {
            # dataset input path, should consist of images png/jpeg/jpg formats
            # If you wish to use pycocotools fill out those fields:
            'overall_dataset_dir': 'MSCOCO/images...',
            'annotations_file_path': '/MSCOCO/annotations...',
            'img_size' : 640,
            'names': None,
            'batch_size': 1,
            'COCO_dataset_root_folder': os.path.join(self.dataset_root_folder,'COCO')
        }
        self.COCO_dataset_params['clean_validation_set_path'] = os.path.join(self.COCO_dataset_params['COCO_dataset_root_folder'],'Clean')
        self.COCO_dataset_params['Unrealistic_PO'] = os.path.join(self.COCO_dataset_params['COCO_dataset_root_folder'],'Unrealistic PO')
        self.COCO_dataset_params['Realistic_PO'] = os.path.join(self.COCO_dataset_params['COCO_dataset_root_folder'],'Realistic PO')
        self.COCO_dataset_params['OOD'] = os.path.join(self.COCO_dataset_params['COCO_dataset_root_folder'],'OOD')

        self.SuperStore_dataset_params = {
            'batch_size': 1,
            'img_size': 640,
            'super_store_dataset_root_folder':os.path.join(self.dataset_root_folder,'SuperStore')
        }
        self.SuperStore_dataset_params['clean_validation_set_path'] = os.path.join(self.SuperStore_dataset_params['super_store_dataset_root_folder'],'Clean')
        self.SuperStore_dataset_params['partial_occlusion_validation_set_path'] = os.path.join(self.SuperStore_dataset_params['super_store_dataset_root_folder'],'PO')
        self.SuperStore_dataset_params['OOD_validation_set_path'] = os.path.join(self.SuperStore_dataset_params['super_store_dataset_root_folder'],'OOD')
        self.SuperStore_dataset_params['adversarial_faster_validation_set_path'] = os.path.join(self.SuperStore_dataset_params['super_store_dataset_root_folder'],'Adv faster rcnn')
        self.SuperStore_dataset_params['adversarial_yolo_validation_set_path'] = os.path.join(self.SuperStore_dataset_params['super_store_dataset_root_folder'],'Adv_yolo')

        self.dataset_params = self.SuperStore_dataset_params if self.model_params['model_dataset']=='SuperStore' else self.COCO_dataset_params

        self.XAI_params = {

            #The desired XAI method to use (out of be GradCAM or GradCAM++ or EigenCAM or EigenGradCAM or
            #GradCAMElementWise or LayerCAM or XGradCAM or HiresCAM)
            'XAI_method' : 'GradCAM',
            # based the saliency methods on the objectness score (default is true, in YOLOv5 also logits is possible)
            'bbox_normalization': False
        }
        self.XAI_params['saliency-based-on-objectness'] = True
        self.XAI_params['eigen_smooth']=False if self.XAI_params['saliency-based-on-objectness'] else True

        self.attack_params = {
            'target_model_in_art_wrapper': False,
            'attack_losses': ["loss_objectness"],
            # 'attack_losses': ("loss_classifier", "loss_box_reg", "loss_objectness",
            #                  "loss_rpn_box_reg"),
            # 'attack_losses': ["loss_classifier", "loss_box_reg",
            #               "loss_rpn_box_reg"],
            'cuda_visible_devices': '1',
            'patch_shape': [150, 150, 3],
            'patch_location': [200, 200],
            'apply_patch_on_main_object':True,
            'crop_range': [0, 0],
            'brightness_range': [1.0, 1.0],
            'rotation_weights': [1, 0, 0, 0],
            'sample_size': 8,
            'learning_rate': 25.0,
            'max_iter': 200,
            'batch_size': 1,
            'image_width': 640,
            'image_height': 640,
            'targeted': True,
            'source_class':1,
            'target_class': 0,
            'verbose': True,
            'plot_images_prediction':False,
            # if you wish to generate an adversarial attack, fill here the path to which the final patch will be saved on
            'saved_patch_dir':'saved_patch_dir',
            'experiment_folder': 'Experiments_output',
        }
        self.attack_params['patches'] = self.get_patches()
        if self.model_params['model_algorithm'] == 'Faster_RCNN' or 'MMdetection':
            self.attack_params['patch_path'] = self.attack_params['patches']['faster_patch_coco']
        else:
            self.attack_params['patch_path'] = self.attack_params['patches']['yolo_patch_coco']

        #  Additional patches
        # self.attack_params['patch_path'] = self.attack_params['patches']['RandomNoisePatch']
        # self.attack_params['patch_path'] = self.attack_params['patches']['faster_patch_super_store']
        # self.attack_params['patch_path'] = self.attack_params['patches']['yolo_patch_super_store']

        self.uncertainty_robustness_params = {
            'Dil_threshold':0.5,
            'Maximum_degradation': 0.2,
            'saliency_map_masking_threshold':0.3
        }

        self.evaluation_params = {
            'experiment_folder': 'Experiments_output',
        }


    def init_model_zoo(self):
        # Super Store model zoo, dict with models path
        # Model root folder - change the models root folder
        models_root_folder = 'models_weights'
        super_store_models_root_path = os.path.join(models_root_folder,'SuperStore')
        super_store_model_zoo = {}
        super_store_model_zoo['faster_rcnn_seed_42'] = [os.path.join(super_store_models_root_path, 'faster_rcnn/target_model_rcnn_seed_42.pt')]
        super_store_model_zoo['yolov5f'] = [os.path.join(super_store_models_root_path, 'YOLOv5/finetune/best.pt')]

        super_store_models_root_path = os.path.join(models_root_folder,'mmdetection/configs')
        super_store_model_zoo['faster_rcnn_caffe_fpn_50'] = [os.path.join(super_store_models_root_path, 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_superstore.pth'),
                                                             os.path.join(super_store_models_root_path, 'faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_superstore.py')]
        super_store_model_zoo['grid_rcnn'] = [
            os.path.join(super_store_models_root_path, 'grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_superstore.pth'),
            os.path.join(super_store_models_root_path, 'grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_superstore.py')]

        super_store_model_zoo['double_heads_faster_rcnn'] = [
            os.path.join(super_store_models_root_path, 'double_heads/dh-faster-rcnn_r50_fpn_1x_superstore.pth'),
            os.path.join(super_store_models_root_path, 'double_heads/dh-faster-rcnn_r50_fpn_1x_superstore.py')]

        super_store_model_zoo['cascade_rpn'] = [
            os.path.join(super_store_models_root_path, 'cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_superstore.pth'),
            os.path.join(super_store_models_root_path, 'cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_superstore.py')]

        super_store_model_zoo['cascade_rcnn'] = [
            os.path.join(super_store_models_root_path,'cascade_rcnn/cascade-rcnn_r50_fpn_1x_superstore.pth'),
            os.path.join(super_store_models_root_path,'cascade_rcnn/cascade-rcnn_r50_fpn_1x_superstore.py')]

        super_store_model_zoo['yolox_x'] = [
            os.path.join(super_store_models_root_path,'yolox/yolox_x_8xb8-300e_superstore.pth'),
            os.path.join(super_store_models_root_path,'yolox/yolox_x_8xb8-300e_superstore.py')]

        super_store_model_zoo['yolo_f'] = [
            os.path.join(super_store_models_root_path, 'yolof/yolof_r50-c5_8xb8-1x_superstore.pth'),
            os.path.join(super_store_models_root_path, 'yolof/yolof_r50-c5_8xb8-1x_superstore.py')]

        super_store_model_zoo['yolo_v3'] = [
            os.path.join(super_store_models_root_path, 'yolo/yolov3_d53_8xb8-ms-608_superstore.pth'),
            os.path.join(super_store_models_root_path, 'yolo/yolov3_d53_8xb8-ms-608_superstore.py')]

        # COCO model zoo, dict with models path
        COCO_models_root_path = os.path.join(models_root_folder,'mmdetection/configs')
        COCO_model_zoo = {}
        COCO_model_zoo['faster_rcnn_r50_fpn'] = [
            os.path.join(COCO_models_root_path, 'faster_rcnn/faster-rcnn_r50_fpn_2x_coco.pth'),
            os.path.join(COCO_models_root_path, 'faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py')]
        COCO_model_zoo['cascade_rpn'] = [
            os.path.join(COCO_models_root_path, 'cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco_test.pth'),
            os.path.join(COCO_models_root_path, 'cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco_test.py')]
        COCO_model_zoo['cascade_rcnn'] = [
            os.path.join(COCO_models_root_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.pth'),
            os.path.join(COCO_models_root_path, 'cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py')]
        COCO_model_zoo['double_heads_faster_rcnn'] = [
            os.path.join(COCO_models_root_path, 'double_heads/dh-faster-rcnn_r50_fpn_1x_coco.pth'),
            os.path.join(COCO_models_root_path, 'double_heads/dh-faster-rcnn_r50_fpn_1x_coco.py')]
        COCO_model_zoo['grid_rcnn'] = [
            os.path.join(COCO_models_root_path, 'grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.pth'),
            os.path.join(COCO_models_root_path, 'grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py')]

        COCO_model_zoo['yolo_f'] = [
            os.path.join(COCO_models_root_path, 'yolof/yolof_r50-c5_8xb8-1x_coco.pth'),
            os.path.join(COCO_models_root_path, 'yolof/yolof_r50-c5_8xb8-1x_coco.py')]
        COCO_model_zoo['yolo_v3'] = [
            os.path.join(COCO_models_root_path, 'yolo/yolov3_d53_8xb8-320-273e_coco.pth'),
            os.path.join(COCO_models_root_path, 'yolo/yolov3_d53_8xb8-320-273e_coco.py')]
        COCO_model_zoo['yolox_l'] = [
            os.path.join(COCO_models_root_path, 'yolox/yolox_l_8xb8_300e_coco.pth'),
            os.path.join(COCO_models_root_path, 'yolox/yolox_l_8xb8_300e_coco.py')]

        return COCO_model_zoo,super_store_model_zoo

    def get_patches(self):
        # Patches root folder
        patches_root_path = 'Adversarial_patches'
        patches = {}
        patches['faster_patch_coco'] = os.path.join(patches_root_path,"faster_patch_coco/best_patch.npy")
        patches['faster_patch_super_store'] = os.path.join(patches_root_path,"faster_patch_super_store/best_patch.npy")
        patches['yolo_patch_coco'] = os.path.join(patches_root_path,"yolo_patch_coco/best_patch.npy")
        patches['yolo_patch_super_store'] = os.path.join(patches_root_path,"yolo_patch_super_store/best_patch.npy")
        patches['RandomNoisePatch'] = os.path.join(patches_root_path,"random_noise_patch/best_patch.npy")
        return patches

    def init_decision_thresholds(self):
        if self.model_params['model_algorithm'] == 'YOLOv5':
            self.model_params['xai_decision_threshold'] = 0.0
            if self.model_params['model_dataset'] == 'COCO':
                self.model_params['decision_threshold'] = 0.5
            else:
                self.model_params['decision_threshold'] = 0.7
        else:
            if self.model_params['model_dataset'] == 'COCO':
                self.model_params['decision_threshold'] = 0.8
                self.model_params['xai_decision_threshold'] = 0.8
            else:
                self.model_params['decision_threshold'] = 0.7
                self.model_params['xai_decision_threshold'] = 0.7

