import glob
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

from sklearn import metrics



# USED
def plot_latent_space_contours(
    ax,
    z, 
    model, 
    z_min,
    z_max,
    resolution=0.1,
    cmap='viridis', 
    alpha=0.9,
    num_levels=25,
    vmin=0,
    vmax=1
):
    """
    Plots contour of class probabilities in the latent space.

    Parameters:
    - ax: matplotlib.ax
        ax to plot on
    - z: np.ndarray
        2D array of latent space points (n_samples, 2).
    - model: torch.nn.Module
        Trained model with a `classify` method to predict class probabilities.
    - z_min: float
        Minimum value of z for mesh grid.
    - z_max: float
        Maximum value of z for mesh grid.
    - resolution: float, optional
        Resolution of the mesh grid, default is 0.1.
    - cmap: str, optional
        Colormap for the contour plot, default is 'viridis'.
    - alpha: float, optional
        Transparency level for the contour plot, default is 0.9.
    """

    # Create a mesh grid for the plot
    xx, yy = np.meshgrid(np.arange(z_min - 1, z_max + 1, resolution), np.arange(z_min - 1, z_max + 1, resolution))
    
    # Transform the grid points back to the original feature space
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict class probabilities for each point in the grid
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        y_pred_grid = model.classify(grid_points_tensor).softmax(dim=1).numpy()
    
    # Reshape the probabilities to match the grid shape
    probs = y_pred_grid[:, 1].reshape(xx.shape)
    
    # Plot the contour for the probability of the second class (class index 1)
    ax.contourf(xx, yy, probs, cmap=cmap, vmin=vmin, vmax=vmax, levels=num_levels, alpha=alpha)
    
def embed_data_from_model_path(model,model_path,X,dist_metric):
    
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.eval()
    with torch.no_grad():
        x_hat, z, y_pred = model(X)
    
    return z,y_pred, x_hat


def plot_train_loss_info(
    loss_df,
    save_str,
    par,
    fold_idx,
    kl_beta,
    classification_loss_beta,
    latent_dim,
    arch_str,
    mse_beta,
    save_img = True
):
    
    #training loss MSE and TOTAL
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    losses_to_plot = ['total_loss', 'reconstruction_loss', 'kl-divergence','valid_mse']
    plt.plot(loss_df[losses_to_plot])
    plt.legend(losses_to_plot)
    
    #classification loss
    plt.subplot(132)
    class_losses = ['classification_loss','valid_class_loss']
    plt.plot(loss_df[class_losses])
    plt.legend(class_losses)
    
    #accuracy 
    plt.subplot(133)
    acc_cols = ['train_acc','valid_acc']
    plt.plot(loss_df[acc_cols])
    plt.ylim([0,1])
    plt.legend(acc_cols)
    prior_test = round(float(loss_df['prior_test'][0]),2)
    plt.title(f'adjusted chance test {prior_test}')
    
    if save_img:
        loss_fig_path = save_str.format(
            par=par,
            fold=fold_idx,
            klb=kl_beta,
            clb=classification_loss_beta,
            ld=latent_dim,
            mse_beta=mse_beta,
            arch= arch_str,
            file='loss.png'
        )
        plt.savefig(loss_fig_path)



