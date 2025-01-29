import glob, os,sys,copy
import pandas as pd
import numpy as np
import yaml

from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import torch
from torch import nn

sys.path.append("../")
from helpers import models, visualization_helper, pytorch_helpers

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

situation_num = int(sys.argv[1])

config_path = f"simulated_data/2d/situation_{situation_num}/config.yaml"

print(config_path)
config = load_config(config_path)

data_info = config['data']
targ = data_info['target']
base_path = data_info["base_path"]
data_path = os.path.join(base_path,data_info['train_path'])
par = data_info['par']#can add this too the loop at some point
data_path = data_path.format(par=par)
save_str =  os.path.join(base_path,data_info['save_path'],data_info['file_str'])

#training params
param_info = config['params']
epochs = 12# param_info['epochs']
kl_beta= param_info['kl_beta']
classification_loss_betas=param_info['classification_loss_betas']
print(classification_loss_betas)

recon_class_ratio = 0
mse_beta=param_info['mse_beta']
learning_rate = param_info['learning_rate']
batch_size = param_info['batch_size']
seed_vals = param_info['seed_vals']
drop_out = param_info['drop_out']
    
#arch params
arch_param_info = config['arch_params']
latent_dim = arch_param_info['latent_dim']
encoder_archs = arch_param_info['encoder_archs']
class_archs = arch_param_info['class_archs']
encoder_act = arch_param_info['encoder_act']
classifier_act = arch_param_info['classifier_act']
encoder_act_str = encoder_act[3:]
classifier_act_str = classifier_act[3:]

#load data
input_dim = int(data_info['train_path'].split('_')[-1].split('d')[0])
n_classes = 2
parcellated_df = pd.read_csv(data_path, header=None,sep=' ')
x_cols = parcellated_df.columns

y = np.zeros(len(parcellated_df))
y[int(len(parcellated_df)/2):]=1
parcellated_df['labels']=y

model = models.VariationalEncoderSecondHead_modular_act_dropout
arch_string = 'classact-{classifier_act_str}_classarch-{classifier_arch}_targ-{targ}_encact-{encoder_act_str}_encarch-{encoder_arch}_bs1-{batch_size}_seed-{seed_val}_dr-{drop_out}'

classifier_burn_in = 50
for seed_val in seed_vals:#loop across seed vals
    Kfolder = StratifiedKFold(shuffle=True, random_state=seed_val)
    
    for encoder_arch in encoder_archs:#loop across encoder archs
        for classifier_arch in class_archs:#loop across classifier archs
            for classification_loss_beta in classification_loss_betas:#loop across weights on classifier
                torch.manual_seed(seed_val)
                splitter = Kfolder.split(parcellated_df.values,y)
                    
                if classification_loss_beta <0:
                    classification_loss_beta = 1
                    mse_beta = 0
                else:
                    mse_beta = 1
                ####### training "loop" ######
                for fold_idx, (train, test) in enumerate(splitter):#loop across folds
                    
                    ############ load data ################# 
                    train_df = parcellated_df.iloc[train,:]
                    test_df = parcellated_df.iloc[test,:]
                    train_dataloader = pytorch_helpers.create_dataloader(
                        train_df,
                        x_cols,
                        targ,
                        batch_size=batch_size,
                        conversion_function=torch.tensor
                    )
                    test_dataloader = pytorch_helpers.create_dataloader(
                        test_df,
                        x_cols,
                        targ,
                        batch_size=len(test_df),
                        conversion_function=torch.tensor
                    )

                    ########## initialize and train model#############
                    vae = model(
                        input_dim,
                        latent_dim,#config
                        encoder_arch,#config
                        classifier_arch,#config
                        eval(encoder_act),
                        eval(classifier_act),
                        n_classes,
                        dropout_rate=drop_out#config
                    )
                    
                    loss_df, mse_state_dict,acc_state_dict = pytorch_helpers.train_model(
                        vae,
                        train_dataloader,
                        test_dataloader,
                        epochs,
                        class_beta=classification_loss_beta,
                        burn_in_for_classifier=classifier_burn_in,
                        mse_beta=mse_beta,
                        kl_beta=kl_beta,
                        ratio_recon_to_class=recon_class_ratio,
                        learning_rate=eval(learning_rate),
                        burn_in_class=1
                    )
                    
                    #### SAVE INFO ########
                    arch_str= arch_string.format(
                        classifier_act_str=classifier_act_str,
                        classifier_arch=classifier_arch,
                        targ=targ,
                        encoder_act_str=encoder_act_str,
                        encoder_arch=encoder_arch,
                        batch_size=batch_size,
                        seed_val=seed_val,
                        drop_out=drop_out
                    )
                    print(arch_str)
                    loss_save_path = save_str.format(
                        par  = par,
                        fold = fold_idx,
                        klb  = kl_beta,
                        clb  = classification_loss_beta,
                        ld   = latent_dim,
                        mse_beta=mse_beta,
                        arch = arch_str,
                        file = 'loss_log1.csv'
                    )
                    loss_df.to_csv(loss_save_path)
                    model_state_dict = copy.deepcopy(vae.state_dict())
                    model_save_path = save_str.format(
                        par  = par,
                        fold = fold_idx,
                        klb  = kl_beta,
                        clb  = classification_loss_beta,
                        ld   = latent_dim,
                        mse_beta=mse_beta,
                        arch = arch_str,
                        file = f'epoch-{epochs}'
                    )
                    torch.save(model_state_dict,model_save_path)
                    ##### VISUALIZE ###########
                    visualization_helper.plot_train_loss_info(
                        loss_df,
                        save_str,
                        par,
                        fold_idx,
                        kl_beta,
                        classification_loss_beta,
                        latent_dim,
                        arch_str,
                        mse_beta
                    )
                    plt.close()
                    break


