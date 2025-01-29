import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
seed = 7
np.random.seed(seed)
def label_closest_half(matrix: np.ndarray, point: np.ndarray):
    """
    Labels the closest 50% of rows in a matrix to a given point with 0, and the other 50% with 1.
    
    Parameters:
    matrix (np.ndarray): A n x d numpy matrix.
    point (np.ndarray): A 1 x d numpy array representing the point.

    Returns:
    np.ndarray: An array of labels (0 and 1).
    """
    # Calculate the Euclidean distances
    distances = np.linalg.norm(matrix - point, axis=1)
    
    # Get the indices that would sort the distances array
    sorted_indices = np.argsort(distances)
    
    # Determine the threshold index for 50%
    half_n = len(distances) // 2
    
    # Create labels array initialized to 1
    labels = np.ones(len(distances), dtype=int)
    
    # Set the first half of the sorted indices to 0
    labels[sorted_indices[:half_n]] = 0
    
    return labels

def generate_data(num_samples, num_networks, transform_scale=1):
    # Situation 1: Single Gaussian distribution for each class
    mu_class_1 = np.random.rand(num_networks)
    transform = np.zeros_like(mu_class_1)
    transform[0] = 2 * transform_scale
    mu_class_2 = mu_class_1 + transform
    cov = np.diag(np.random.rand(num_networks))
    class_1_data = np.random.multivariate_normal(mu_class_1, cov, num_samples // 2)
    class_2_data = np.random.multivariate_normal(mu_class_2, cov, num_samples // 2)
    data_situation_1 = np.vstack((class_1_data, class_2_data))

    # Situation 2: Two Gaussian distributions for each class
    mu_class_1_1 = np.random.rand(num_networks)
    transform = np.zeros_like(mu_class_1_1)
    transform[0] = 6 * transform_scale
    mu_class_1_2 = mu_class_1_1 + transform
    transform = np.zeros_like(mu_class_1_1)
    transform[1] = 3 * transform_scale
    mu_class_2_1 = mu_class_1_1 + transform
    transform = np.zeros_like(mu_class_1_1)
    transform[0] = 6 * transform_scale
    transform[1] = 3 * transform_scale
    mu_class_2_2 = mu_class_1_1 + transform
    cov_1 = np.diag(np.random.rand(num_networks) + 0.5)
    cov_2 = np.diag(np.random.rand(num_networks) + 0.5)
    class_1_data_1 = np.random.multivariate_normal(mu_class_1_1, cov_1, num_samples // 4)
    class_1_data_2 = np.random.multivariate_normal(mu_class_1_2, cov_2, num_samples // 4)
    class_2_data_1 = np.random.multivariate_normal(mu_class_2_1, cov_1, num_samples // 4)
    class_2_data_2 = np.random.multivariate_normal(mu_class_2_2, cov_2, num_samples // 4)
    data_situation_2 = np.vstack((class_1_data_1, class_1_data_2, class_2_data_1, class_2_data_2))

    # Situation 3: Two Gaussian distributions for each class (non-linearly separable)
    mu_class_1_3 = np.random.rand(num_networks)
    transform = np.zeros_like(mu_class_1_3)
    transform[:2] = 6 * transform_scale
    mu_class_1_4 = mu_class_1_3 + transform
    transform = np.zeros_like(mu_class_1_3)
    transform[:2] = -4 * transform_scale
    mu_class_2_3 = mu_class_1_3 + transform
    transform = np.zeros_like(mu_class_1_3)
    transform[:2] = 2 * transform_scale
    mu_class_2_4 = mu_class_1_3 + transform
    cov_3 = np.diag(np.random.rand(num_networks) + 0.5)
    cov_4 = np.diag(np.random.rand(num_networks) + 0.5)
    class_1_data_3 = np.random.multivariate_normal(mu_class_1_3, cov_3, num_samples // 4)
    class_1_data_4 = np.random.multivariate_normal(mu_class_1_4, cov_4, num_samples // 4)
    class_2_data_3 = np.random.multivariate_normal(mu_class_2_3, cov_3, num_samples // 4)
    class_2_data_4 = np.random.multivariate_normal(mu_class_2_4, cov_4, num_samples // 4)
    data_situation_3 = np.vstack((class_1_data_3, class_1_data_4, class_2_data_3, class_2_data_4))

    # Situation 4: Single Gaussian distribution, label closest 50% to mean as class 0, others as class 1
    mu_class_4 = np.random.rand(num_networks)
    cov_4 = np.diag(np.random.rand(num_networks) + 0.5)
    data_situation_4 = np.random.multivariate_normal(mu_class_4, cov_4, num_samples)
    labels_situation_4 = label_closest_half(data_situation_4, mu_class_4)

    return data_situation_1, data_situation_2, data_situation_3, data_situation_4, labels_situation_4



if __name__ == "__main__":
    num_samples = 1000
    num_networks = 2
    
    data_situation_1, data_situation_2, data_situation_3, data_situation_4, labels_situation_4 = generate_data(num_samples, num_networks)
    class_labels = np.repeat([1, 2], num_samples // 2)
    
    projection_coeffs = np.random.randn(2, 100)

    projected_1 = np.dot(data_situation_1, projection_coeffs)
    projected_2 = np.dot(data_situation_2, projection_coeffs)
    projected_3 = np.dot(data_situation_3, projection_coeffs)
    
    data_situation_4 = np.vstack([
        data_situation_4[(labels_situation_4 == 1)],
        data_situation_4[(labels_situation_4 == 0)],
    ])
    
    projected_4 = np.dot(data_situation_4, projection_coeffs)
    
    np.savetxt(f'simulated_data/2d/situation_1/X_100d-{seed}1.csv',projected_1)
    np.savetxt(f'simulated_data/2d/situation_1/latent_X_seed-{seed}1.csv',data_situation_1)

    np.savetxt(f'simulated_data/2d/situation_2/X_100d-{seed}1.csv',projected_2)
    np.savetxt(f'simulated_data/2d/situation_2/latent_X_seed-{seed}1.csv',data_situation_2)

#     np.savetxt(f'simulated_data/2d/situation_3/X_100d-{seed}1.csv',projected_3)
#     np.savetxt(f'simulated_data/2d/situation_3/latent_X_seed-{seed}1.csv',data_situation_3)

#     np.savetxt(f'simulated_data/2d/situation_4/X_100d-{seed}1.csv',projected_4)
#     np.savetxt(f'simulated_data/2d/situation_4/latent_X_seed-{seed}1.csv',data_situation_4)