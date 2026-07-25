"""PPAL CCMS/diversity acquisition step."""

from pathlib import Path

import numpy as np

from methods.common.coco_pool import image_ids, read_coco_json
from methods.ppal.base import BaseALSampler
from methods.ppal.inference import load_image_distance_cache


eps = 1e-10


class DiversitySampler(BaseALSampler):
    def __init__(
        self,
        n_sample_images,
        oracle_annotation_path,
        dataset_type,
    ):
        super(DiversitySampler, self).__init__(
            n_sample_images,
            oracle_annotation_path,
            is_random=False,
            dataset_type=dataset_type)

        self.kmeans_iterations = 100

    @staticmethod
    def k_centroid_greedy(dis_matrix, K):
        N = dis_matrix.shape[0]
        centroids = []
        c = np.random.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

    @staticmethod
    def kmeans(dis_matrix, K, n_iter=100):
        N = dis_matrix.shape[0]
        centroids = DiversitySampler.k_centroid_greedy(dis_matrix, K)
        data_indices = np.arange(N)

        assign_dis_records = []
        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1).sum()
            assign_dis_records.append(assign_dis)

            new_centroids = []
            for i in range(K):
                cluster_i = data_indices[cluster_assign == i]
                assert len(cluster_i) >= 1
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            centroids = np.array(new_centroids)
        return centroids.tolist()

    def al_acquisition(self, image_dis_path, last_label_path):
        image_dis_matrix, distance_image_ids = load_image_distance_cache(image_dis_path)

        centroids = DiversitySampler.kmeans(
            image_dis_matrix,
            K=self.n_images,
            n_iter=self.kmeans_iterations,
        )

        results = read_coco_json(Path(last_label_path))

        last_labeled_img_ids = image_ids(results)
        image_hit = dict()
        for img_id in self.oracle_data.keys():
            image_hit[img_id] = 0
        for img_id in last_labeled_img_ids:
            image_hit[img_id] = 1

        rest_image_ids = []
        for img_id in self.oracle_data.keys():
            if image_hit[img_id] == 0:
                rest_image_ids.append(img_id)

        sampled_img_ids = distance_image_ids[centroids].tolist()
        for img_id in sampled_img_ids:
            rest_image_ids.remove(img_id)
        unsampled_img_ids = rest_image_ids

        metrics = {
            'image_distance_npy': str(image_dis_path),
            'distance_matrix_shape': list(image_dis_matrix.shape),
            'distance_image_count': int(distance_image_ids.shape[0]),
            'kmeans_iterations': int(self.kmeans_iterations),
        }
        return sampled_img_ids, unsampled_img_ids, metrics

    def al_round(self, result_path, image_dis_path, last_label_path, out_label_path, out_unlabeled_path):
        self.round += 1
        self.latest_labeled = last_label_path
        sampled_img_ids, rest_img_ids, metrics = self.al_acquisition(image_dis_path, last_label_path)
        counts = self.create_jsons(
            sampled_img_ids,
            rest_img_ids,
            last_label_path,
            out_label_path,
            out_unlabeled_path,
        )
        metrics.update(counts)
        return {
            'selected_image_ids': sampled_img_ids,
            'selected_count': len(sampled_img_ids),
            'metrics': metrics,
        }
