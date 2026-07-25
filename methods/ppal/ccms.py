"""PPAL CCMS/diversity acquisition step."""

import numpy as np

from methods.common.image_identity import (
    normalize_image_ids,
    validate_image_ids_subset,
)
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

    def al_acquisition(self, image_dis_path):
        image_dis_matrix, distance_image_ids = load_image_distance_cache(image_dis_path)
        oracle_image_ids = normalize_image_ids(distance_image_ids.reshape(-1).tolist())
        validate_image_ids_subset(
            oracle_image_ids,
            self.oracle_data.keys(),
            'PPAL diversity cache',
        )

        centroids = DiversitySampler.kmeans(
            image_dis_matrix,
            K=self.n_images,
            n_iter=self.kmeans_iterations,
        )

        sampled_img_ids = [oracle_image_ids[index] for index in centroids]

        metrics = {
            'image_distance_npy': str(image_dis_path),
            'distance_matrix_shape': list(image_dis_matrix.shape),
            'distance_image_count': int(distance_image_ids.shape[0]),
            'canonical_distance_image_count': len(oracle_image_ids),
            'selected_candidate_count': len(sampled_img_ids),
            'kmeans_iterations': int(self.kmeans_iterations),
        }
        return sampled_img_ids, metrics

    def al_round(self, image_dis_path, last_label_path):
        self.round += 1
        self.latest_labeled = last_label_path
        sampled_img_ids, metrics = self.al_acquisition(image_dis_path)
        return {
            'selected_image_ids': sampled_img_ids,
            'selected_count': len(sampled_img_ids),
            'metrics': metrics,
        }
