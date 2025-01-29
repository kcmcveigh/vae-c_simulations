import glob, os,sys
import pandas as pd
import numpy as np
import yaml

from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, f1_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt

import torch
from torch import nn

from helpers import models, pytorch_helpers

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calc_and_flatten_dist_matrix(points):
    # Method 1: Using scipy.spatial.distance.cdist
    distance_matrix = cdist(points, points, metric='euclidean')
    
    row_indices, col_indices = np.tril_indices_from(distance_matrix, k=-1)
    
    return distance_matrix[row_indices, col_indices]

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

latent_path = data_info['train_path'].split('X_100d')[0] + 'latent_X_seed-7.csv'
latent_path = os.path.join(base_path,latent_path)
latent_2d_ground_truth = pd.read_csv(latent_path, header=None, sep=' ')

#training params
param_info = config['params']
epochs = param_info['epochs']
kl_beta= param_info['kl_beta']
classification_loss_betas=param_info['classification_loss_betas']
print(classification_loss_betas)
classifier_burn_in = 50
recon_class_ratio = 0
mse_beta=param_info['mse_beta']
learning_rate = param_info['learning_rate']
batch_size = param_info['batch_size']
seed_vals = param_info['seed_vals']
drop_out = param_info['drop_out']


if recon_class_ratio!=0:
    classification_loss_beta = 1/recon_class_ratio
    mse_beta = 1
    total_weights = classification_loss_beta + mse_beta
    percent_mse,percent_class = mse_beta/total_weights,classification_loss_beta/total_weights
    print(classification_loss_betas,percent_mse,percent_class)
    #assert len(classification_loss_betas)!=0,'recon class ratio not zero and list of class betas'
      
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

first_pass=True
model = models.VariationalEncoderSecondHead_modular_act_dropout
cross_entropy_loss_func = torch.nn.CrossEntropyLoss()

save_cols = ["Encoder Architecture","Classifier Architecture",
             "Classifier Weight","R Squared","F1-Score","Accuracy","RSA", "kl-divergence"
             ,"Cross Entropy Loss","MSE",'latent_cov','Seed']
save_arr = []
torch_mse = nn.MSELoss(reduction='sum')
arch_string = 'classact-{classifier_act_str}_classarch-{classifier_arch}_targ-{targ}_encact-{encoder_act_str}_encarch-{encoder_arch}_bs-{batch_size}_seed-{seed_val}_dr-{drop_out}'
for seed_val in seed_vals:#loop across seed vals
    Kfolder = StratifiedKFold(shuffle=True, random_state=seed_val)
    
    for encoder_arch in encoder_archs:#loop across encoder archs
        for classifier_arch in class_archs:#loop across classifier archs
            for classification_loss_beta in classification_loss_betas:
                classification_save_loss_beta = classification_loss_beta
                torch.manual_seed(seed_val)
                
                if classification_loss_beta <0:
                    classification_loss_beta = 1
                    mse_beta = 0
                else:
                    mse_beta = 1
                splitter = Kfolder.split(parcellated_df.values,y)
                
                for fold_idx, (train, test) in enumerate(splitter):#loop across folds
                    
                   
                    
                    ############ load data #################
                    test_df = parcellated_df.iloc[test,:]
                    test_latents = latent_2d_ground_truth.iloc[test,:]
                    
                    test_dataloader = pytorch_helpers.create_dataloader(
                        test_df,
                        x_cols,
                        targ,
                        batch_size=len(test_df),
                        conversion_function=torch.tensor,
                        shuffle=False
                    )

                    ########## initialize model#############
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
                    
                    model_save_path = save_str.format(
                        par  = par,
                        fold = fold_idx,
                        klb  = kl_beta,
                        clb  = classification_loss_beta,
                        ld   = latent_dim,
                        mse_beta=mse_beta,
                        arch = arch_str,
                        file = 'epoch-1500'
                    )

                    vae.load_state_dict(torch.load(model_save_path))
                    vae.eval()

                    if first_pass:
                        X_test,y_test = next(iter(test_dataloader))
                        first_pass=False

                    X_hat,latent_est,y_hat = vae(X_test)
                    X_test_np = X_test.detach().numpy()
                    X_hat_np=X_hat.detach().numpy()
                    latent_est_np = latent_est.detach().numpy()
                    
                    y_hat_probs, y_hat_labels = pytorch_helpers.get_labels_from_logits(y_hat.detach())
                    
                    ce_loss= float(cross_entropy_loss_func(y_hat_probs,y_test).numpy())
                    f1 = f1_score(y_test,y_hat_labels)
                    acc = accuracy_score(y_test,y_hat_labels)
                    r2 = r2_score(X_test_np,X_hat_np)
                    mse = torch_mse(X_test,X_hat)/len(X_test)
                    mse = float(mse.detach().numpy())
                    kl = vae.kl.detach().numpy()/len(X_test)
                    
                    est_dist = calc_and_flatten_dist_matrix(latent_est_np)
                    ground_truth_dist = calc_and_flatten_dist_matrix(test_latents)
                    rsa = np.corrcoef(est_dist,ground_truth_dist)[0][1]
                    
                    enc_str ='Linear' if encoder_arch == [] else 'Non-Linear'
                    arch_str_save ='Linear' if classifier_arch == [] else 'Non-Linear'
                    
                    cov = np.cov(latent_est_np.T)
                    save_arr.append([enc_str,arch_str_save,
                                     classification_loss_beta,r2,
                                     f1,acc, rsa, kl,ce_loss,mse,
                                     cov[0,:],seed_val])
                    break

df = pd.DataFrame(save_arr,columns=save_cols)
df.to_csv(f'results/tables/compiled_table_situation_{situation_num}_RSA1.csv')