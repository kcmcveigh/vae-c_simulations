import torch
import copy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, r2_score
from tqdm import tqdm
# Named tuple to represent validation loss information
ValidLossInfo = namedtuple('ValidLossInfo', ['valid_acc', 'valid_mse', 'valid_class_loss'])

############# training model helpers
def validate(valid_dataloader, model,n_samples=10):
    """
    Validate the given model on a validation dataset.

    :param valid_dataloader: DataLoader object containing the validation dataset
    :param model: PyTorch model to be validated
    :return: Named tuple containing validation accuracy, mean squared error, and class loss
    """
    epoch_tests = []
    for idx in range(n_samples):
        epoch_test_acc, test_preds, test_mse, test_class = calc_test_acc(valid_dataloader, model)
        epoch_tests.append([epoch_test_acc, test_mse.cpu().numpy(), test_class.cpu().numpy()])
    epoch_tests = np.array(epoch_tests)
    mean_test = np.mean(epoch_tests, axis=0)
    return ValidLossInfo(
        valid_acc=mean_test[0],
        valid_mse=mean_test[1],
        valid_class_loss=mean_test[2])

def create_save_df(losses,acc_list, test_dataloader, valid_losses):
    """
    Create a DataFrame containing training and validation losses.

    :param losses: List of training losses
    :param test_dataloader: DataLoader object containing the test dataset
    :param valid_losses: List of validation losses
    :return: DataFrame containing the training and validation losses
    """
    # Create train loss DataFrame
    loss_df = pd.DataFrame(losses,
                           columns=['total_loss', 
                                    'reconstruction_loss',
                                    'kl-divergence', 
                                    'classification_loss'])
    loss_df['train_acc']=acc_list
    
    # Get % of labels of the most common class (chance)
    X_test, y_test = next(iter(test_dataloader))
    mode = torch.mode(y_test.cpu())
    chance_of_priors_test = (list(y_test).count(mode.values) / len(y_test))
    loss_df['prior_test'] = chance_of_priors_test
    
    # Create validation DataFrame
    valid_loss_df = pd.DataFrame(valid_losses, columns=ValidLossInfo._fields)
    loss_df = loss_df.join(valid_loss_df)
    return loss_df

def compute_weights(loss1, loss2, R, total_weight=1.0):
    '''
    Computes the weights for the two losses given the desired ratio R of loss1 to loss2.
    loss1: float the first loss
    loss2: float the second loss
    R: float the desired ratio of loss1 to loss2
    total_weight: float the total weight of the two losses
    '''
    w1 = (R * loss2 / loss1) / (R * loss2 / loss1 + 1)
    w2 = 1 / (R * loss2 / loss1 + 1)
    return w1 * total_weight, w2 * total_weight

def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    epochs,
    class_beta=1,
    burn_in_for_classifier=0,
    mse_beta=1,
    kl_beta=1,
    learning_rate=1e-4,
    burn_in_class=0,
    ratio_recon_to_class = 0,
    classification_loss_func=torch.nn.CrossEntropyLoss(),
    recon_loss_func=torch.nn.MSELoss(reduction='sum'),
    opt_func=torch.optim.Adam,
    weight_decay = .0
):
    """
    Train the given model using the specified parameters.

    :param model: PyTorch model to be trained
    :param train_dataloader: DataLoader object containing the training dataset
    :param valid_dataloader: DataLoader object containing the validation dataset
    :param class_beta: Classification loss weighting factor (default: 1)
    :param mse_beta: Mean squared error weighting factor (default: 1)
    :param kl_beta: KL-divergence weighting factor (default: 1)
    :param learning_rate: Learning rate for the optimizer (default: 1e-4)
    :param classification_loss_func: Classification loss function (default: CrossEntropyLoss)
    :param mse_loss: Mean squared error loss function (default: MSELoss with sum reduction)
    :param opt_func: Optimizer function (default: Adam)
    :return: DataFrame with loss information, model parameters for accuracy, and model parameters for MSE
    """
    # Initialize training info lists
    acc_list, losses, valid_losses = [], [], []
    min_loss = np.inf
    best_mse = np.inf
    
    opt = opt_func(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    print(burn_in_for_classifier)
    for i in tqdm(range(epochs),disable=True):
        model.train()
        if i < burn_in_for_classifier:
            class_loss_beta=burn_in_class
            mse_b = 1
        if i == burn_in_for_classifier:
            class_loss_beta=class_beta
            mse_b=mse_beta
            print(class_loss_beta,mse_b)
            opt = opt_func(model.parameters(), lr=learning_rate, weight_decay=.01)
        if (ratio_recon_to_class > 0) and (i > burn_in_for_classifier) and (i>0):#need to test
            mse_loss_val = epoch_loss_list[1]
            class_loss = epoch_loss_list[3]
            mse_b, class_loss_beta = compute_weights(mse_loss_val,class_loss,ratio_recon_to_class)
            #print(mse_loss_val,mse_b,class_loss,class_loss_beta)
        
        epoch_loss_list, epoch_acc_list = train_epoch(
            train_dataloader,
            opt,
            model,
            recon_loss_func,
            classification_loss_func,
            mse_beta=mse_b,
            kl_beta=kl_beta,
            class_loss_beta=class_loss_beta
        )
        
        losses.append(epoch_loss_list)
        acc_list.append(epoch_acc_list)

        # Validate model performance
        model.eval()
        valid_loss_epoch = validate(valid_dataloader, model)
        valid_losses.append(valid_loss_epoch)
        
        # Save best models
        if i > 10:
            last_mse_loss = valid_loss_epoch.valid_mse
            last_class = valid_loss_epoch.valid_class_loss
            if last_class < min_loss:
                model_params_acc = copy.deepcopy(model.state_dict())
                min_loss = last_class
                # print(mean_epoch_tests[0],mean_epoch_tests[2])
            if last_mse_loss < best_mse:
                model_params_mse = copy.deepcopy(model.state_dict())
                best_mse = last_mse_loss

    # Save data
    loss_df = create_save_df(
        losses,
        acc_list,
        valid_dataloader, 
        valid_losses,
    )
    
    return loss_df,model_params_mse, model_params_acc


        


def create_dataloader(df,
                      x_cols,
                      y_col,
                      batch_size=32,
                      conversion_function = torch.tensor,
                      shuffle=True
                     ):
    X_numpy =df[x_cols].values
    y_list =df[y_col].values
    y_list = [conversion_function(label) for label in y_list]
    x_tensor = torch.tensor(X_numpy).float()
    y_tensor = torch.tensor(y_list).long()
    dataset = TensorDataset(x_tensor,y_tensor)
    dataloader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)
    return dataloader 
    



def train_epoch(dataloader,
                opt,
                model,
                recon_loss_func,
                class_loss_func,
                kl_beta=1,
                class_loss_beta=1,
                mse_beta = 1,
               ):
    epoch_loss_list =[]
    acc_list = []
    
    for X, y in dataloader:
        opt.zero_grad()
        X_hat, z_est,y_pred = model(X)

        probs = y_pred.softmax(dim=1)
        classification_loss = class_loss_func(probs,y)
        recon_loss = recon_loss_func(X,X_hat)/len(X)
        kl_loss = model.kl/len(X)

        loss =  mse_beta * recon_loss + kl_beta * kl_loss + class_loss_beta * classification_loss
        loss.backward()
        opt.step()

        _, pred_label = torch.max(probs.detach(), dim=1)
        acc = torch.sum(y == pred_label)/len(y)

        epoch_loss_list.append([
            loss.detach(),
            recon_loss.detach(),
            kl_loss.detach(),
            classification_loss.detach()
        ])
        
        acc_list.append(acc.detach().numpy())
        
    return np.mean(epoch_loss_list,axis=0), np.mean(acc_list)

def get_labels_from_logits(y_pred):
    test_probs = y_pred.softmax(dim=1)
    _, test_pred_label = torch.max(test_probs, dim=1)
    return test_probs, test_pred_label

def calc_test_acc(dataloader,model,class_loss_func=torch.nn.CrossEntropyLoss()):
    test_acc_list = []
    with torch.no_grad():
        for X, y in dataloader:
            X_hat, z_est,y_pred = model(X)
            # test_probs = y_pred.softmax(dim=1)
            # _, test_pred_label = torch.max(test_probs, dim=1)
            test_probs, test_pred_label = get_labels_from_logits(y_pred)
            test_acc = torch.sum(y == test_pred_label)/len(y)
            test_acc_list.append(test_acc.detach().numpy())
            mse_loss = torch.nn.functional.mse_loss(X,X_hat,reduction='sum')/len(X)
            class_loss = class_loss_func(test_probs,y)
    return np.mean(test_acc_list), test_pred_label.detach().numpy(), mse_loss,class_loss


