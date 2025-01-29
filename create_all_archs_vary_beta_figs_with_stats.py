import glob
import os
import sys
import pandas as pd
import numpy as np
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import torch
from torch import nn

from helpers import models, visualization_helper, pytorch_helpers

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



situation_num = int(sys.argv[1])
config_path = f"simulated_data/2d/situation_{situation_num}/config.yaml"
config = load_config(config_path)

table_path = f'results/tables/compiled_table_situation_{situation_num}_RSA.csv'
df = pd.read_csv(table_path)

data_info = config['data']
targ = data_info['target']
base_path = data_info["base_path"]
data_path = os.path.join(base_path, data_info['train_path'])
print(data_path)
par = data_info['par']  # can add this to the loop at some point
data_path = data_path.format(par=par)
save_str = os.path.join(base_path, data_info['save_path'], data_info['file_str'])

# arch params
arch_param_info = config['arch_params']
latent_dim = arch_param_info['latent_dim']
encoder_archs = arch_param_info['encoder_archs']
class_archs = arch_param_info['class_archs']
encoder_act = arch_param_info['encoder_act']
classifier_act = arch_param_info['classifier_act']
encoder_act_str = encoder_act[3:]
classifier_act_str = classifier_act[3:]

# training params
param_info = config['params']
epochs = param_info['epochs']
kl_beta = param_info['kl_beta']
classification_loss_betas = param_info['classification_loss_betas']
mse_beta = param_info['mse_beta']
learning_rate = param_info['learning_rate']
batch_size = param_info['batch_size']
seed_vals = param_info['seed_vals']
drop_out = param_info['drop_out']

# load data
input_dim = 100
n_classes = 2
parcellated_df = pd.read_csv(data_path, header=None, sep=' ')
x_cols = parcellated_df.columns

y = np.zeros(len(parcellated_df))
y[int(len(parcellated_df) / 2):] = 1
parcellated_df['labels'] = y

first_pass = True
classification_loss_betas = [10, 100, 500, -1]
model = models.VariationalEncoderSecondHead_modular_act_dropout
arch_string = 'classact-{classifier_act_str}_classarch-{classifier_arch}_targ-{targ}_encact-{encoder_act_str}_encarch-{encoder_arch}_bs-{batch_size}_seed-{seed_val}_dr-{drop_out}'
fig_save_path = 'results/paper_results/figures/situation-{situation}_mse-{mse}_clb-{clb}_ld-{ld}_scatter_{arch_str}.png'
fig, axs = plt.subplots(4, 
                        len(classification_loss_betas), 
                        figsize=(15, 12), 
                        sharex='row', sharey='row')

# Define column titles for each beta
col_titles = []
for clb in classification_loss_betas:
    if clb == -1:
        col_titles.append("Fully Supervised")
    else:
        col_titles.append(f"Classification Weight: {clb}")

# Put column titles above the top row
for i, ax in enumerate(axs[0]):
    ax.set_title(col_titles[i], pad=20, fontsize=12)



first_pass = True
for seed_val in seed_vals:  # loop across seed vals
    Kfolder = StratifiedKFold(shuffle=True, random_state=seed_val)
    subplot_idx_class = 0
    for encoder_arch in encoder_archs:  # loop across encoder archs
        global_z_min = np.inf
        global_z_max = -np.inf
        
        for classifier_arch in class_archs:
            subplot_idx_enc = 0
            enc_str = 'Linear' if encoder_arch == [] else 'Non-Linear'
            class_arch_str = 'Linear' if classifier_arch == [] else 'Non-Linear'
            row_label = f'{enc_str} Encoder \n{class_arch_str} Classifier'
            axs[subplot_idx_class][0].set_ylabel(row_label, rotation=0, labelpad=50)
    
            # hackey way to get global (row wise) global min and max of latent space
            for classification_loss_beta in classification_loss_betas:  
                classification_save_loss_beta = classification_loss_beta
                if classification_loss_beta < 0:
                    classification_loss_beta = 1
                    mse_beta = 0
                else:
                    mse_beta = 1
                splitter = Kfolder.split(parcellated_df.values, y)
                
                for fold_idx, (train, test) in enumerate(splitter): 
                    
                    #we don't do cross validation just train and test for one fold
                    if fold_idx != 0:
                        break

                    test_df = parcellated_df.iloc[test, :]
                    test_dataloader = pytorch_helpers.create_dataloader(
                        test_df,
                        x_cols,
                        targ,
                        batch_size=len(test_df),
                        conversion_function=torch.tensor
                    )

                    vae = model(
                        input_dim,
                        latent_dim,
                        encoder_arch,
                        classifier_arch,
                        eval(encoder_act),
                        eval(classifier_act),
                        n_classes,
                        dropout_rate=drop_out
                    )
                    
                    arch_str = arch_string.format(
                        classifier_act_str=classifier_act_str,
                        classifier_arch=classifier_arch,
                        targ=targ,
                        encoder_act_str=encoder_act_str,
                        encoder_arch=encoder_arch,
                        batch_size=batch_size,
                        seed_val=seed_val,
                        drop_out=drop_out
                    )
                    
                    model_save_path = save_str.format(
                        par=par,
                        fold=fold_idx,
                        klb=kl_beta,
                        clb=classification_loss_beta,
                        ld=latent_dim,
                        mse_beta=mse_beta,
                        arch=arch_str,
                        file='epoch-1500'
                    )

                    if first_pass:
                        X_test, y_test = next(iter(test_dataloader))
                        first_pass = False

                    z, y_pred, x_hat = visualization_helper.embed_data_from_model_path(
                        vae,
                        model_save_path,
                        X_test,
                        ''
                    )
                    _, y_pred = pytorch_helpers.get_labels_from_logits(y_pred)

                    global_z_min = min(global_z_min, z[:, 0].min(), z[:, 1].min())
                    global_z_max = max(global_z_max, z[:, 0].max(), z[:, 1].max())

            # Same logic as above just does actual plotting with min and max for z values for each row which hackey loop above got
            subplot_idx_enc = 0
            for classification_loss_beta in classification_loss_betas:  # loop across weights on classifier
                classification_save_loss_beta = classification_loss_beta
                if classification_loss_beta < 0:
                    classification_loss_beta = 1
                    mse_beta = 0
                else:
                    mse_beta = 1
                splitter = Kfolder.split(parcellated_df.values, y)
                
                for fold_idx, (train, test) in enumerate(splitter):  # loop across folds
                    if fold_idx != 0:
                        break

                    test_df = parcellated_df.iloc[test, :]
                    test_dataloader = pytorch_helpers.create_dataloader(
                        test_df,
                        x_cols,
                        targ,
                        batch_size=len(test_df),
                        conversion_function=torch.tensor
                    )

                    vae = model(
                        input_dim,
                        latent_dim,
                        encoder_arch,
                        classifier_arch,
                        eval(encoder_act),
                        eval(classifier_act),
                        n_classes,
                        dropout_rate=drop_out
                    )
                    
                    arch_str = arch_string.format(
                        classifier_act_str=classifier_act_str,
                        classifier_arch=classifier_arch,
                        targ=targ,
                        encoder_act_str=encoder_act_str,
                        encoder_arch=encoder_arch,
                        batch_size=batch_size,
                        seed_val=seed_val,
                        drop_out=drop_out
                    )
                    
                    model_save_path = save_str.format(
                        par=par,
                        fold=fold_idx,
                        klb=kl_beta,
                        clb=classification_loss_beta,
                        ld=latent_dim,
                        mse_beta=mse_beta,
                        arch=arch_str,
                        file='epoch-1500'
                    )

                    if first_pass:
                        X_test, y_test = next(iter(test_dataloader))
                        first_pass = False

                    z, y_pred, x_hat = visualization_helper.embed_data_from_model_path(
                        vae,
                        model_save_path,
                        X_test,
                        ''
                    )
                    _, y_pred = pytorch_helpers.get_labels_from_logits(y_pred)
                    
                    ax = axs[subplot_idx_class][subplot_idx_enc]
                    
                    visualization_helper.plot_latent_space_contours(
                        ax=ax,
                        z=z, 
                        model=vae,
                        z_min=global_z_min,
                        z_max=global_z_max,
                        resolution=0.1,
                        cmap='viridis', 
                        alpha=0.7,
                        num_levels=50,
                        vmin=0,
                        vmax=1
                    )

                    scatter1 = ax.scatter(
                        z[:, 0],
                        z[:, 1],
                        c=y_test, 
                        cmap='viridis', 
                        vmin=0,
                        vmax=1, 
                        edgecolors='k'
                    )
                    variant_row = df.loc[
                        (df['Encoder Architecture']==enc_str) &
                        (df['Classifier Architecture']==class_arch_str) &
                        (df['Classifier Weight']==classification_loss_beta)
                    ]

                    stats = np.round(
                        variant_row.iloc[-1][['R Squared','RSA','F1-Score']].astype(float).values,
                        2
                    )
                    if classification_loss_beta ==1:
                        title = f'r = {stats[1]:.2f}, F1 = {stats[2]:.2f}'     
                    else:
                        title = f'R2 = {stats[0]:.2f}, r = {stats[1]:.2f}, F1 = {stats[2]:.2f}'
                    
                    #ax.set_title(title)
                    ax.text(
                        0.5, 1.02,               # x,y in Axes coords (0..1)
                        title,
                        transform=ax.transAxes,  # so (0.5,1.02) is relative to subplot
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black'
                    )
                subplot_idx_enc += 1
            subplot_idx_class += 1
            
# Finally, adjust layout so top titles aren’t cut off
plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig(f'results/figures/vary_arch_betas_situation-{situation_num}.png')