import os.path
import numpy as np
import math
import json
import csv
from Attack.Attack_util import Attack_object_detector
from OD_models.Object_detector import Object_detection_model
from Config import configs
from XAI.XAI_util import Explainer
from XAI.DiL_quantification_techniques import saliency_sum,mask_objects_saliency_map
from Evaluation.Eval_util import evaluation_module
from OD_models.Object_detector import plot_image_with_boxes
from Data.Data_util import load_dataset_attack_format

class dil_interface():
    def __init__(self, config):
        self.robustness_params = config.uncertainty_robustness_params
        self.attack_params = config.attack_params
        self.model_params = config.model_params
        self.dataset_params = config.dataset_params
        self.xai_params = config.XAI_params
        self.evaluation_params = config.evaluation_params
        self.pred_objects_predicted = None
        # Object detection model initialization
        self.od_model = Object_detection_model(model_params=self.model_params,
                                               decision_threshold=self.model_params['decision_threshold'],
                                               device=self.model_params['device'],
                                               img_size=self.dataset_params['img_size'],
                                               target_model_path = self.model_params['model_path'])
        #If using MMdetection framework, change the model to explain to faster RCNN or YOLOv5
        if self.model_params['model_algorithm'] == 'MMDetection':
            self.model_params['model_algorithm'] = 'Faster_RCNN'
            # self.model_params['model_algorithm'] = 'YOLOv5'
            # self.model_params['xai_decision_threshold'] = 0.0
            if self.model_params['model_dataset'] == 'SuperStore':
                self.model_params['model_path'] = config.Superstore_models['faster_rcnn_seed_42']
                # self.model_params['model_path'] = config.Superstore_models['yolov5f']

        # XAI Object detection model initialization
        self.xai_model = Object_detection_model(model_params=self.model_params,
                                                decision_threshold=self.model_params['xai_decision_threshold'],
                                                device=self.model_params['device'],
                                                img_size=self.dataset_params['img_size'],
                                                target_model_path = self.model_params['model_path'])


    def dil_experiment(self, images, file_names, object_counter_path, annotations):
        """
        Main function that establish an experiment on DiL estimation.
        :param images: req. list of scenes in tensor format.
        :param file_names: req. list of strings representing each file id.
        :param object_counter_path: req. string representing the path for the object counter file (used for recall computation).
        :param annotations: req. COCO format annotation dicts.
        :return: prints the model performance, the complete, background and distinctive scores and save artifacts to output folders.
        """
        self.eval_module = evaluation_module(self.od_model,self.model_params,self.dataset_params,self.evaluation_params)
        self.eval_module.init_experiment_folder()
        gt_objects_counter = self.read_csv_to_dict(object_counter_path)
        eval_dict = self.eval_module.create_evaluation_dict(images,file_names,gt_objects_counter)
        predictions = self.predict_and_evaluate_model(images, file_names, gt_objects_counter, annotations = annotations,output_plot_path=self.eval_module.evaluation_params['saved_base_model_preds'])
        for image_id,pred in zip(eval_dict.keys(),predictions):
            eval_dict[image_id]['predictions']=pred
        # filter scenes were the attack failed
        images_fool, self.file_names_fool,predictions_fool,images_dropped = self.drop_good_predictions(images, file_names,predictions,gt_objects_counter)
        self.eval_module.mAP(np.array(predictions_fool), annotations, self.file_names_fool, images_fool,
                             self.model_params['model_dataset'])
        distinctive_localization_scores,saliency_maps = self.XAI_plugin(images,file_names,
                                                            output_path=self.eval_module.evaluation_params['saved_XAI'],
                                                            base_model_preds=predictions,
                                                            mode='attacked')
        print("Model predictions and their corresponding saliency maps are saved in the Experiment_output folder")

    def dil_clean_experiment(self, images, file_names, object_counter_path, annotations):
        """
        Main function that establish an experiment on DiL estimation.
        :param images: req. list of scenes in tensor format.
        :param file_names: req. list of strings representing each file id.
        :param object_counter_path: req. string representing the path for the object counter file (used for recall computation).
        :param annotations: req. COCO format annotation dicts.
        :return: prints the model performance, the complete, background and distinctive scores and save artifacts to output folders.
        """
        self.eval_module = evaluation_module(self.od_model,self.model_params,self.dataset_params,self.evaluation_params)
        self.eval_module.init_experiment_folder()
        gt_objects_counter = self.read_csv_to_dict(object_counter_path)
        eval_dict = self.eval_module.create_evaluation_dict(images,file_names,gt_objects_counter)
        predictions = self.predict_and_evaluate_model(images, file_names, gt_objects_counter, annotations = annotations,output_plot_path=self.eval_module.evaluation_params['saved_base_model_preds'])
        for image_id,pred in zip(eval_dict.keys(),predictions):
            eval_dict[image_id]['predictions']=pred
        # filter scenes were the attack failed
        self.eval_module.mAP(np.array(predictions), annotations, file_names, images,
                             self.model_params['model_dataset'])
        distinctive_localization_scores,saliency_maps = self.XAI_plugin(images,file_names,
                                                            output_path=self.eval_module.evaluation_params['saved_XAI'],
                                                            base_model_preds=predictions)
        print("Model predictions and their corresponding saliency maps are saved in the Experiment_output folder")

    def compute_ddt(self,images, file_names, object_counter_path, annotations):
        """
        Main function that generates a dynamic decision threshold based on the DiL metric and calculate the model performance.
        :param images: req. list of scenes in tensor format.
        :param file_names: req. list of strings representing each file id.
        :param object_counter_path: req. string representing the path for the object counter file (used for recall computation).
        :param annotations: req. COCO format annotation dicts.
        :return: prints the model performance, the complete, background and distinctive scores and save artifacts to output folders.
        """
        self.eval_module = evaluation_module(self.od_model, self.model_params, self.dataset_params,
                                             self.evaluation_params)
        self.eval_module.init_experiment_folder()
        gt_objects_counter = self.read_csv_to_dict(object_counter_path)
        eval_dict = self.eval_module.create_evaluation_dict(images, file_names, gt_objects_counter)
        predictions = self.predict_and_evaluate_model(images, file_names, gt_objects_counter, annotations=annotations,
                                                      output_plot_path=self.eval_module.evaluation_params[
                                                          'saved_base_model_preds'])
        for image_id, pred in zip(eval_dict.keys(), predictions):
            eval_dict[image_id]['predictions'] = pred
        images_fool, self.file_names_fool, predictions_fool, images_dropped = self.drop_good_predictions(images,
                                                                                                         file_names,
                                                                                                         predictions,
                                                                                                         gt_objects_counter)
        # self.xai_params['XAI_method'] = 'GradCAM++'
        distinctive_localization_scores, saliency_maps = self.XAI_plugin(images, file_names,
                                                                         output_path=self.eval_module.evaluation_params[
                                                                             'saved_XAI'],
                                                                         base_model_preds=predictions)
        for image_id,dil_score in zip(eval_dict.keys(),distinctive_localization_scores):
            eval_dict[image_id]['DiL_score'] = dil_score
        dynamic_thresholds = self.generate_dynamic_threshold(
            [eval_dict[image_id]['DiL_score'] for image_id in self.file_names_fool])
        print(f'Robust solution (DDT):')
        self.predict_and_evaluate_model(images_fool, self.file_names_fool, gt_objects_counter,
                                        self.eval_module.evaluation_params['saved_robust_model_preds_manipulated'],
                                        annotations,
                                        dynamic_thresholds)


    def XAI_plugin(self, input_images,file_names,output_path,base_model_preds,mode='clean'):
        """
        Main function which generate a saliency maps for a given input scenes and calculated dil scores.
        :param images: req. list of scenes in tensor format.
        :param file_names: req. list of strings representing each file id.
        :param output_path: req. string representing the path to save the produced saliency maps.
        :param base_model_preds: req. list of dicts representing the corresponding prediction for each image.
        return: dil scores and save the produces' saliency maps to an output folder.
        """
        exp = Explainer(object_detection_model=self.xai_model,
                        XAI_method=self.xai_params['XAI_method'],
                        model_algorithm=self.model_params['model_algorithm'])
        saliency_maps,_ = exp.apply_explanations(input_images,
                                        saliency_based_on_objectness=self.xai_params['saliency-based-on-objectness'],
                                                                eigen_smooth = self.xai_params['eigen_smooth'])
        exp.visualize(original_images=input_images,
                      heatmap_cams=saliency_maps,
                      prediction_dicts=base_model_preds,
                      output_path=output_path,
                      bbox_renormalize=config.XAI_params['bbox_normalization'],
                      file_names = file_names)
        self.od_model.model.eval()
        complete_localization_score, complete_localization_list = saliency_sum(saliency_maps)
        saliency_maps_back = mask_objects_saliency_map(saliency_maps, base_model_preds)
        background_localization_score, background_localization_list = saliency_sum(saliency_maps_back)
        distinctive_localization_score = \
            [background_score/objectness_score for objectness_score,background_score in zip(complete_localization_list,background_localization_list)]
        if mode=='clean':
            print(f'Mean complete localization score: {complete_localization_score}')
            print(f'Mean background localization score: {background_localization_score}')
            print(f'Mean DiL score :{sum(distinctive_localization_score) / len(distinctive_localization_score)}')
        # Drop samples were the attack failed
        else:
            filtered_complete,filtered_cackground,filtered_dils= [],[],[]
            dil_t = 0.3
            for complete_score,background_score,dil_score,file_name in zip(complete_localization_list,background_localization_list,distinctive_localization_score,file_names):
                if dil_score>dil_t or file_name in self.file_names_fool:
                    filtered_dils.append(dil_score)
                    filtered_complete.append(complete_score)
                    filtered_cackground.append(background_score)

            print(f'Mean complete localization score: {sum(filtered_complete) / len(filtered_complete)}')
            print(f'Mean background localization score: {sum(filtered_cackground) / len(filtered_cackground)}')
            print(f'Mean DiL score: {sum(filtered_dils) / len(filtered_dils)}')
        return distinctive_localization_score,saliency_maps

    def generate_dynamic_threshold(self,dil_scores):
        """
        Main function which calculates the dynamic decision threshold based on the dil scores.
        :param confidence_thresholds: list of floats representing the dil score for each scene in the dataset.
        return the DDT for each scene
        """
        dynamic_thresholds = [self.calculate_threshold(confidence_threshold) for confidence_threshold in dil_scores]
        return dynamic_thresholds

    def calculate_threshold(self, dil_score):
        """
        Help function that calculates the DDT for a single image.
        """
        if math.isnan(dil_score):
            dil_score=1.0
        return round(self.model_params['decision_threshold'] - (self.robustness_params['Maximum_degradation']*dil_score),3)


    def plot(self,images,prediction_dicts,file_names,output_path = None):
        images = [patched_image.detach().cpu().numpy() * 255 for patched_image in images]
        for idx,(image,pred_dict,file_name) in enumerate(zip(images,prediction_dicts,file_names)):
            image = np.squeeze(image)
            image = np.transpose(image, (1, 2, 0)).astype(np.uint8).copy()
            plot_image_with_boxes(image,
                                  boxes=pred_dict['boxes'],
                                  pred_cls=pred_dict['labels'],
                                  confidence=pred_dict['scores'],
                                  output_path=output_path,
                                  image_id=file_name)


    def predict(self,images,file_names,output_path,detection_threshold=None,DiL_scores =None):
        """
        Help function for producing a prediction (uses the object_detector module).
        """
        if detection_threshold==None:
            detection_threshold=self.model_params['decision_threshold']
        preds = self.od_model.predict_wrapper(image_dataloader=images, saliency_maps=None,
                                                     DiL_scores=DiL_scores, use_grad=False,
                                                    detection_threshold = detection_threshold)
        self.plot(images,preds,file_names,output_path)
        return self.object_count_per_scene(preds, file_names),preds

    def object_count_per_scene(self, preds, files_names):
        object_count = [len(pred['labels']) for pred in preds]
        return dict(zip(files_names, object_count))

    def read_csv_to_dict(self,csv_path):
        reader = csv.reader(open(csv_path, 'r'))
        d = {}
        for row in reader:
            k, v = row
            d[k] = v
        return d

    def extract_object_counts(self, gt_objects_counter, pred_objects_predicted):
        gt_objects_counter = list(gt_objects_counter.values())[1:]
        gt_objects_counter = [int(x) for x in gt_objects_counter]
        pred_objects_predicted = list(pred_objects_predicted.values())
        # pred_objects_predicted = [int(x) for x in pred_objects_predicted]
        return gt_objects_counter,pred_objects_predicted

    def predict_and_evaluate_model(self, images, file_names, gt_objects_counter,output_plot_path,annotations,detection_threshold=None):
        pred_objects_counter, predictions = dil.predict(images, file_names, output_plot_path, detection_threshold)
        self.evaluate_recall_and_fp(gt_objects_counter, pred_objects_counter)
        return predictions

    def evaluate_recall_and_fp(self, gt_objects_counter, pred_objects_counter):
        gt_objects_counter, pred_objects_predicted = self.extract_object_counts(gt_objects_counter,
                                                                                pred_objects_counter)
        model_performance = evaluation_module.get_base_model_performance(gt_objects_counter,
                                                                         pred_objects_predicted)
        fp = evaluation_module.get_model_fp(gt_objects_counter, pred_objects_predicted)
        print(f'Recall_performance:{model_performance}')
        print(f'False positive :{fp}')
        if self.pred_objects_predicted != None:
            dif = evaluation_module.get_base_robustness_models_performance_difference(gt_objects_counter,
                                                                                      self.pred_objects_predicted,
                                                                                      pred_objects_predicted)
            print(f'Base robustness models performance difference :{dif}')
        else:
            self.pred_objects_predicted = pred_objects_predicted

    def drop_good_predictions(self, images, file_names, predictions,gt_objects_counter):
        """
        Function which remove predictions in which the attack (natural or deliberate) failed.
        """
        gt_objects_counter = list(gt_objects_counter.values())[1:]
        gt_objects_counter = [int(x) for x in gt_objects_counter]
        index_to_keep=[]
        index_to_remove =[]
        for idx,(prediction,gt_object_counter) in enumerate(zip(predictions,gt_objects_counter)):
            if len(prediction['labels'])<gt_object_counter:
                index_to_keep.append(idx)
            else:
                index_to_remove.append(idx)
        images = [images[i] for i in index_to_keep if i < len(images)]
        file_names = [file_names[i] for i in index_to_keep if i < len(file_names)]
        predictions = [predictions[i] for i in index_to_keep if i < len(predictions)]
        dropped_images = [images[i] for i in index_to_remove if i < len(images)]
        print(f'Total DiL evaluation samples:{len(images)}')
        return images,file_names,predictions,dropped_images


def generate_attack(config):
    """
    Help function for applying an adversarial attack (used in the digital adversarial use case)
    """
    attack_module = Attack_object_detector(config)
    images,patched_images,benign_prediction_dicts,adversarial_prediction_dicts,file_names = attack_module.apply_digital_attack()
    return images,patched_images,benign_prediction_dicts,adversarial_prediction_dicts,file_names


def digital_evaluation_demo(evaluation_use_case,config):
    """
    Main function to execute a digital evaluation use case.
    :param evaluation_use_case: the chosen evaluation use case
    :param config: a config module
    return: execute a digital evaluation use case
    """
    if evaluation_use_case=='clean':
        dataset_path = config.dataset_params['clean_validation_set_path']
    elif evaluation_use_case=='unrealistic partial occlusion':
        dataset_path = config.dataset_params['Unrealistic_PO']
    elif evaluation_use_case=='realistic partial occlusion':
        dataset_path = config.dataset_params['Realistic_PO']
    elif evaluation_use_case=='ood':
        dataset_path = config.dataset_params['OOD']
    elif evaluation_use_case=='adversarial_faster' or evaluation_use_case=='adversarial_yolo':
        dataset_path = config.dataset_params['clean_validation_set_path']
        images, patched_images, benign_prediction_dicts, adversarial_prediction_dicts, file_names = generate_attack(
            config)
        images = patched_images
    else:
        print("You choose an invalid evaluation use case for the digital evaluation space.")
        return

    if evaluation_use_case!='adversarial_faster' and evaluation_use_case!='adversarial_yolo':
        images_dir = config.dataset_params.get('images_dir', os.path.join(dataset_path, 'images'))
        images, file_names = load_dataset_attack_format(images_dir, config.model_params['model_algorithm'])

    return images, file_names, dataset_path


def physical_evaluation_demo(evaluation_use_case,config):
    """
   Main function to execute a physical evaluation use case.
   :param evaluation_use_case: the chosen evaluation use case
   :param config: a config module
   return: execute a physical evaluation use case
   """
    if evaluation_use_case == 'clean':
        dataset_path = config.dataset_params['clean_validation_set_path']
    elif evaluation_use_case == 'unrealistic partial occlusion':
        print(f"Physical evaluation space do not support {evaluation_use_case} use case, "
              f"please select a different evaluation space or use case.")
        return
    elif evaluation_use_case == 'realistic partial occlusion':
        dataset_path = config.dataset_params['partial_occlusion_validation_set_path']
    elif evaluation_use_case == 'ood':
        dataset_path = config.dataset_params['OOD_validation_set_path']
    elif evaluation_use_case == 'adversarial_faster':
        dataset_path = config.dataset_params['adversarial_faster_validation_set_path']
    elif evaluation_use_case == 'adversarial_yolo':
        dataset_path = config.dataset_params['adversarial_yolo_validation_set_path']
    else:
        print("You choose an invalid evaluation use case for the digital evaluation space.")
        return
    images_dir = config.dataset_params.get('images_dir', os.path.join(dataset_path, 'images'))
    images, file_names = load_dataset_attack_format(images_dir, config.model_params['model_algorithm'])
    return images, file_names,dataset_path


if __name__ == "__main__":
    ################################################################################################################
    """
    PLEASE FOLLOW THE "GETTING STARTED" SECTION IN THE README FILE BEFORE EXECUTING THE CODE.
    """
    ################################################################################################################
    """
    DiL experimental demo. 
    Please chose the evaluation space and use case from the following:
    - Evaluation space: 
        - 'digital' - digital evaluation using the COCO dataset. 
        - 'physical' - physical evaluation using the SuperStore dataset.
     - Evaluation use case:
        - 'Clean' - a benign scenes.
        - 'Unrealistic partial occlusion' - scenes with object partially occluded by a different objects in an unrealistic manner.
        - 'Realistic partial occlusion' - scenes with object partially occluded by a different objects in a realistic manner.
        - 'Out-of-distribution' - scenes with objects taken from a class that was not part of the model’s training set.
        - 'Adversarial' - scenes with adversarial patch that caused the model to misidentified the object (faster or yolo modes).      
    """

    evaluation_space_options = ['digital','physical']
    evaluation_use_case_options = ['clean','unrealistic partial occlusion','realistic partial occlusion',
                                   'ood', 'adversarial_faster', 'adversarial_yolo']

    # Select here the evaluation space and use case
    evaluation_space = evaluation_space_options[0]
    evaluation_use_case = evaluation_use_case_options[0]

    #  Upload the images and their corresponding file ids from the chosen dataset.
    if evaluation_space=='digital':
        config = configs('COCO')
        images,file_names, dataset_path= digital_evaluation_demo(evaluation_use_case,config)
    else:
        config = configs('SuperStore')
        images,file_names, dataset_path = physical_evaluation_demo(evaluation_use_case,config)

    # Init the dil interface class
    dil = dil_interface(config)

    #  Upload additional annotation files (for evaluation purpose).
    object_counter_path = config.dataset_params.get('objects_count_path', os.path.join(dataset_path,'objects_count/objects_count.csv'))
    annotations_path = config.dataset_params.get('annotations_path', os.path.join(dataset_path,'annotations/annotations.json'))

    with open(annotations_path) as json_file:
        annotations = json.load(json_file)
    if evaluation_use_case=='clean':
        dil.dil_clean_experiment(images, file_names, object_counter_path=object_counter_path, annotations = annotations)
    else:
        dil.dil_experiment(images, file_names, object_counter_path=object_counter_path, annotations = annotations)




