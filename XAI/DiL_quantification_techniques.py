import torch
import numpy as np
from tqdm import tqdm
from XAI.XAI_util import Explainer
import matplotlib.pyplot as plt
import copy

"""
This module is used to quantify a saliency map image using the distinctive localization metric.
"""

def saliency_sum(saliency):
    """
    The evaluation metric.
    Sum all the saliency maps values (and average them if the aggregation funtionality is activated).
    :param saliency: Require. tensor in the shape of (batch size,) that contains saliency maps tensors. Contrary, can be a ndarray of ndarrays.
    :return: tensor at the size of (1). The score according to the saliency maps.
    """
    num_of_saliency = saliency.shape[0] if hasattr(saliency, 'shape') else len(saliency)
    final_score = torch.zeros(1)
    final_std_score = torch.zeros(1)
    margin = 0.0000001
    scores_list = []
    for idx, saliency_map in enumerate(saliency):
        if torch.is_tensor(saliency_map):
            saliency_map = saliency_map.detach().cpu().numpy()
        saliency_map = np.asarray(saliency_map)
        saliency_map = np.nan_to_num(saliency_map, nan=0.0)
        saliency_t = torch.tensor(saliency_map)
        obj_score = torch.nanmean(saliency_t)+margin
        scores_list.append(float(obj_score.numpy()))

        if obj_score>1.0:
            obj_score = torch.log(torch.sum(saliency_t))
        if not torch.isnan(obj_score):
            final_score += obj_score
    final_score_norm = final_score / num_of_saliency
    return float(final_score_norm.numpy()),scores_list
    # return np.array(scores_list),scores_list

def mask_objects_saliency_map(saliency_maps, prediction_dicts):
    b_saliency_map = copy.deepcopy(saliency_maps)
    for saliency_map,pred in zip(b_saliency_map, prediction_dicts):
        for bbox in pred['boxes']:
            saliency_map[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = np.nan
    return b_saliency_map


def saliency_sum_evaluation(tech_name_list, input_images, od_model, prediction_dicts,config,mode= 'benign',viz_saliency_ensemble = False):
    """
    Performs the evaluation of diffrent saliency maps according to the saliency_sum metric.
    :param tech_name_list: Require. List of strings. The names of the evaluated saliency maps techniques.
    :param input_images: Require. ndarray (as outputed by the "load_images" function in data utils). The input images.
    :param od_model: Require. "Object_detection_model" instance. The explained object detector.
    :param object_detection_backbone: Require. string. The backbone used by the object detector.
    :param decision_threshold: Require. float. The threshold for bounding box filtering.
    :param agg: Optional. string. The aggregation technique. if "avg" the technique will be average. if "none" there will be no aggragation.
    :return: scores - a dict where the keys are the name of the evaluated methods (str) and the values are the corresponding scores.
            best - string. The name of the best method.
    """
    # prediction_dicts = od_model.predict(input_images, decision_threshold)
    scores = dict()
    saliency_map_dict = {}
    if mode=='benign':
        output_path =config.COCO_dataset_params['benign_output_dir']
    else:
        output_path = config.COCO_dataset_params['advresarial_output_dir']
    for tech_name in tqdm(tech_name_list, desc="Applying saliency map techniques"):
        exp = Explainer(object_detection_model=od_model,
                        XAI_method=tech_name,
                        model_algorithm=config.model_params['model_algorithm'])
        saliency_maps = exp.apply_explanations(input_images)
        saliency_map_dict[tech_name] = np.copy(saliency_maps)
        print("Complete Localization score:")
        score_objectness,objectness_list = saliency_sum(np.array(saliency_maps))
        exp.visualize(original_images=input_images,
                      heatmap_cams=saliency_maps,
                      prediction_dicts=prediction_dicts,
                      output_path=output_path,
                      bbox_renormalize=config.XAI_params['bbox_normalization'])
        saliency_maps_back = mask_objects_saliency_map(saliency_maps,prediction_dicts)
        print("Background Localization score:")
        score_background,background_list = saliency_sum(np.array(saliency_maps_back))
        scores.update({f'{tech_name}_complete_localization': score_objectness})
        scores.update({f'{tech_name}_background_localization': score_background})
        # scores.update({f'{tech_name}_distinctive_localization': score_background/score_objectness})
        # exp.visualize(original_images=input_images,
        #               heatmap_cams=saliency_maps,
        #               prediction_dicts=prediction_dicts,
        #               output_path=output_path,
        #               bbox_renormalize=config.XAI_params['bbox_normalization'])
        print(f"{tech_name} Distinctive Localization Scores")
        # for objectness_score,background_score in zip(objectness_list,background_list):
            # print(round(background_score/objectness_score, 3))
    if viz_saliency_ensemble:
        visualized_multiple_saliency_map(input_images, saliency_map_dict,exp,output_path,config,prediction_dicts)
    best = max(scores, key=scores.get)
    return scores , best, saliency_map_dict

def visualized_multiple_saliency_map(samples,saliency_map_dict,exp,output_path,config,prediction_dicts):
    saliency_ensembles = []
    for idx in range(len(samples)):
        saliency_maps = []
        for saliency_map_tech in saliency_map_dict.keys():
            saliency_maps.append(saliency_map_dict[saliency_map_tech][idx])
        saliency_ensembles.append(np.amax(np.array(saliency_maps), axis=0))
    print("-" * 145)
    print("Complete Localization score:")
    score_objectness, objectness_list = saliency_sum(np.array(saliency_ensembles))
    print("Background Localization score:")
    saliency_maps_back = mask_objects_saliency_map(saliency_ensembles, prediction_dicts)
    score_background, background_list = saliency_sum(np.array(saliency_maps_back))
    print("Distinctive Localization Scores")
    for objectness_score, background_score in zip(objectness_list, background_list):
        print(round(background_score / objectness_score, 3))
    exp.visualize(original_images=samples,
                  heatmap_cams=saliency_ensembles,
                  prediction_dicts=prediction_dicts,
                  output_path=output_path,
                  bbox_renormalize=config.XAI_params['bbox_normalization'])
    print("-" * 145)


def plot_hist(images,tech_name,mode):
    nb_bins = 256
    final_hist = np.zeros(nb_bins)
    for image in images:
        histogram, bin_edges = np.histogram(image, bins=nb_bins, range=(0, 1))
        final_hist+=histogram
    plt.figure()
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    # plt.ylim([0.0, 500000.0])  # <- named arguments do not work here
    plt.plot(bin_edges[:-1], final_hist)  # <- or here
