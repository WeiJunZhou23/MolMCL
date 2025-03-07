import os
import sys
import pickle
import copy
import csv
import random
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_batch
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from molmcl.finetune.loader import MoleculeDataset
##################################################
from molmcl.finetune.loader_customized import MoleculeDataset_cm
##################################################
from molmcl.finetune.model import GNNPredictor
from molmcl.finetune.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_
from molmcl.splitters import scaffold_split, moleculeace_split
from molmcl.utils.scheduler import PolynomialDecayLR

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')  # Ensure reduction='mean'

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))  # RMSE already reduces to a scalar

def get_optimizer(model, lr_params):
    assert isinstance(lr_params, dict)

    pretrain_name, prompt_name, finetune_name = [], [], []
    for name, param in model.named_parameters():
        if 'gnn' in name or 'aggr' in name:
            pretrain_name.append(name)
        elif 'graph_pred_linear' in name:
            finetune_name.append(name)
        else:
            prompt_name.append(name)


    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters()))))
    finetune_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in finetune_name, model.named_parameters()))))
    prompt_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in prompt_name, model.named_parameters()))))

    # Adam, (Adadelta), Adagrad, RAdam
    optimizer = torch.optim.Adam([
        {'params': finetune_params},
        {'params': pretrain_params, 'lr': float(lr_params['pretrain_lr'])},
        {'params': prompt_params, 'lr': float(lr_params['prompt_lr'])}
    ], lr=float(lr_params['finetune_lr']), weight_decay=float(lr_params['decay']))

    return optimizer


# def get_dataloader(config, seed=0):
#     # Setup dataset
#     dataset = MoleculeDataset(config['dataset']['data_dir'],
#                               config['dataset']['data_name'],
#                               config['dataset']['feat_type'])

#     num_task = dataset.num_task
#     print('Loading dataset {} of size {} with num_task={}'.format(config['dataset']['data_name'], len(dataset), num_task))

#     if 'CHEMBL' in config['dataset']['data_name']:  # MoleculeACE stratified random split
#         train_idx, val_idx, test_idx = moleculeace_split(dataset.smiles, dataset.labels, val_size=0.1, test_size=0.1)
#     else:  # MoleculeNet scaffold split
#         train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, frac_valid=0.1, frac_test=0.1, balanced=False)
    
#     train_dataset, val_dataset, test_dataset = \
#         Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

#     train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

#     return dataset, train_loader, val_loader, test_loader

def get_dataloader(config, seed=0, split_type="scaffold", save_split=True):

    split_type = config['dataset']['split_type']
    
    if split_type == 'customized':
        
        # Directly load pre-defined DataFrames
        df_train = pd.read_csv(config['dataset']['custom_train_path'])
        df_val = pd.read_csv(config['dataset']['custom_val_path'])
        df_test = pd.read_csv(config['dataset']['custom_test_path'])
        
        # Assume the first column is SMILES and the second column is labels
        smiles_col = df_train.columns[0]  # First column
        labels_col = df_train.columns[1]  # Second column

        dataset = MoleculeDataset_cm(df_train[smiles_col], df_train[[labels_col]], config['dataset']['feat_type'])
        train_dataset = MoleculeDataset_cm(df_train[smiles_col], df_train[[labels_col]], config['dataset']['feat_type'])
        val_dataset = MoleculeDataset_cm(df_val[smiles_col], df_val[[labels_col]], config['dataset']['feat_type'])
        test_dataset = MoleculeDataset_cm(df_test[smiles_col], df_test[[labels_col]], config['dataset']['feat_type'])

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
        return dataset, train_loader, val_loader, test_loader

    else:
        # Setup dataset
        dataset = MoleculeDataset(config['dataset']['data_dir'],
                                  config['dataset']['data_name'],
                                  config['dataset']['feat_type'])
        num_task = dataset.num_task
        print(f'Loading dataset {config["dataset"]["data_name"]} of size {len(dataset)} with num_task={num_task}')
        print(f'Split type: {config["dataset"]["split_type"]}; Seed {seed}')
    
        train_file = os.path.join(config['dataset']['data_dir'], f"{config['dataset']['data_name']}_train.csv")
        val_file = os.path.join(config['dataset']['data_dir'], f"{config['dataset']['data_name']}_val.csv")
        test_file = os.path.join(config['dataset']['data_dir'], f"{config['dataset']['data_name']}_test.csv")
    
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            train_idx = pd.read_csv(train_file)['index'].tolist()
            val_idx = pd.read_csv(val_file)['index'].tolist()
            test_idx = pd.read_csv(test_file)['index'].tolist()
        else:
            if split_type == "moleculeace" and "CHEMBL" in config['dataset']['data_name']:
                train_idx, val_idx, test_idx = moleculeace_split(dataset.smiles, dataset.labels, val_size=0.1, test_size=0.1)
            elif split_type == "random":
                indices = np.arange(len(dataset))
                train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=seed)
                val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=seed)
            else:  # Default to MoleculeNet scaffold split
                train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, frac_valid=0.1, frac_test=0.1, balanced=False)
        
            if save_split:
                pd.DataFrame({'index': train_idx}).to_csv(train_file, index=False)
                pd.DataFrame({'index': val_idx}).to_csv(val_file, index=False)
                pd.DataFrame({'index': test_idx}).to_csv(test_file, index=False)
    
        train_dataset, val_dataset, test_dataset = \
            Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
        return dataset, train_loader, val_loader, test_loader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval(model, val_loader, config, metric='rmse'):
    assert metric in ['rmse', 'r2']
    model.eval()
    y_true, y_scores = [], []
    for step, batch in enumerate(val_loader):
        batch = batch.to(config['device'])
        with torch.no_grad():
            predict = model(batch)['predict']

        y_true.append(batch.label.view(predict.shape))
        y_scores.append(predict)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if 'CHEMBL' in config['dataset']['data_name'] or config['dataset']['task'] == 'regression':
        if metric == 'rmse':
            score = np.sqrt(mean_squared_error(y_true, y_scores))
        else:
            score = r2_score(y_true, y_scores)
    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

        score = np.mean(roc_list)

    return score


def train(model, train_loader, criterion, optimizer, scheduler, config, channel_idx=-1):
    model.train()
    loss_history = []
    channel_weight = 0
    for idx, batch in enumerate(train_loader):
        batch.to(config['device'])
        output = model(batch, channel_idx=channel_idx)
        predict = output['predict']
        label = batch.label.view(predict.shape)

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            mask = label == 0  # nan entry
            loss = criterion(predict.double(), (label + 1) / 2) * (~mask)
            loss = loss.sum() / (~mask).sum()
        elif isinstance(criterion, nn.MSELoss):
            loss = criterion(predict, label)
            loss = loss.mean()
        elif isinstance(criterion, RMSELoss):  # Include RMSELoss here
            loss = criterion(predict, label)
            loss = loss.mean()
        elif isinstance(criterion, nn.L1Loss):  # Include MAE here
            loss = criterion(predict, label)
            loss = loss.mean()
        else:
            raise Exception("Unsupported loss function")

        optimizer.zero_grad()
        loss.backward()
        if config['optim']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
        optimizer.step()

        if config['optim']['scheduler'] == 'poly_decay':
            scheduler.step()

        loss_history.append(loss.item())

    channel_weight = channel_weight / len(train_loader)

    return np.mean(loss_history), channel_weight


def optimize_prompt_weight_ri(model, train_loader, val_loader, config, metric='euclidean', act='softmax', max_num=5000):
    temperature = config['model']['temperature']
    skip_bo = config['prompt_optim']['skip_bo']

    # Extract channel-wise embeddings for all training data
    num = 0
    model.eval()
    graph_rep_list, label_list = [], []
    for loader in [train_loader, val_loader]:
        if loader is None:
            continue
        for batch in loader:
            batch.to(config['device'])
            with torch.no_grad():
                graph_reps = []
                if model.backbone == 'gps':
                    h_g, node_repres = model.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    h_g, node_repres = model.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # map back to batched nodes for aggregation
                batch_x, batch_mask = to_dense_batch(node_repres, batch.batch)

                # conditional aggregation given the prompt_inds
                for i in range(len(model.prompt_token)):
                    h_g, h_x, _ = model.aggrs[i](batch_x, batch_mask)
                    if config['model']['normalize']:
                        h_g = F.normalize(h_g, dim=-1)
                    graph_reps.append(h_g)

            graph_reps_batch = torch.stack(graph_reps)
            labels_batch = batch.label.view(-1, model.num_tasks)

            is_valid = (labels_batch != 0).sum(-1) == labels_batch.size(1)
            graph_rep_list.append(graph_reps_batch[:, is_valid])
            label_list.append(labels_batch[is_valid])

            num += graph_rep_list[-1].size(1)
            if num > max_num:
                break

    graph_reps = torch.concat(graph_rep_list, dim=1).cpu()  # (num_prompt, N, emb_dim)
    labels = torch.concat(label_list, dim=0).cpu()  # (N, 1)

    return optimize_prompt_weight_ri_(graph_reps, labels, n_runs=50, n_inits=50, n_points=5, n_restarts=512,
                                      n_samples=512, temperature=temperature, metric=metric,
                                      skip_bo=skip_bo, verbose=config['verbose'])


def main(config):
    if not config['model']['checkpoint']:
        config['model']['use_prompt'] = False
    save_dir = config['save_dir']
    try:
        os.makedirs(save_dir, exist_ok=True)  # Creates all necessary directories
    except OSError as error:
        print(f"Error creating directory {save_dir}: {error}")
        
    runseeds = np.random.randint(100, size=config['num_run'])
    split_seed = config['split_seed']
    # Setup model
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError('Unrecognized feature type. Please choose from [basic/rich/super_rich].')

    # Main:
    #avg_auc_last, avg_auc_best = [], []

    best_initialization = None
    if config['prompt_optim']['inits']:
        best_initialization = torch.Tensor(config['prompt_optim']['inits'])

    for i in range(config['num_run']):
        # Setup dataset and dataloader
        if config['dataset']['split_type'] == 'customized':
            dataset, train_loader, val_loader, test_loader = get_dataloader(config, seed=(split_seed+i))
        else:
            dataset, train_loader, val_loader, test_loader = get_dataloader(config, seed=(split_seed+i))
        # Setup model
        model = GNNPredictor(num_layer=config['model']['num_layer'],
                             emb_dim=config['model']['emb_dim'],
                             num_tasks=dataset.num_task,
                             normalize=config['model']['normalize'],
                             atom_feat_dim=atom_feat_dim,
                             bond_feat_dim=bond_feat_dim,
                             drop_ratio=config['model']['dropout_ratio'],
                             attn_drop_ratio=config['model']['attn_dropout_ratio'],
                             temperature=config['model']['temperature'],
                             use_prompt=config['model']['use_prompt'],
                             model_head=config['model']['heads'],
                             layer_norm_out=config['model']['layernorm'], 
                             backbone=config['model']['backbone'])

        if config['model']['checkpoint']:
            print('Loading checkpoint from {}'.format(config['model']['checkpoint']))
            model.load_state_dict(torch.load(config['model']['checkpoint'])['wrapper'], strict=False)
        model.to(config['device'])

        # Train prompt:
        if config['model']['use_prompt']:
            if best_initialization is None:
                best_initialization = optimize_prompt_weight_ri(model, train_loader, val_loader, config)

            model.set_prompt_weight(best_initialization.to(config['device']))

            # if args.verbose and use_prompt:
            initial_prompt_probs = model.get_prompt_weight('softmax').data.cpu()
            initial_prompt_weights = model.get_prompt_weight('none').data.cpu()
            print('Initial prompt weight:', initial_prompt_weights)
            print('Initial prompt prob:  ', initial_prompt_probs)

        # Setup optimizer
        optimizer = get_optimizer(model, config['optim'])
        scheduler = None
        if config['optim']['scheduler'] == 'cos_anneal':
            scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0.0001)
        elif config['optim']['scheduler'] == 'poly_decay':
            scheduler = PolynomialDecayLR(optimizer, warmup_updates=config['epochs'] * len(train_loader) // 10,
                                          tot_updates=config['epochs'] * len(train_loader),
                                          lr=config['optim']['finetune_lr'], end_lr=1e-9, power=1)
        
        best_score, best_checkpoint = None, None  # Initialize here
        # Setup loss function
        if config['dataset']['task'] == 'regression':
            print('Initialize best score and best checkpoint for regression tasks.')
            best_score = float('inf')
            if config['dataset']['loss_func'] == 'MSE':
                criterion = nn.MSELoss(reduction='none')
            elif config['dataset']['loss_func'] == 'RMSE':
                criterion = RMSELoss()
            elif config['dataset']['loss_func'] == 'MAE':
                criterion = nn.L1Loss(reduction='none')
            
        elif config['dataset']['task'] == 'classification':
            print('Initialize best score and best checkpoint for classification tasks.')
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            best_score = -float('inf')
        else:
            raise NotImplementedError

        # Setup learnable parameters:
        model.freeze_aggr_module()

        # Setup random seed
        print("Seed:", runseeds[i])
        set_seed(runseeds[i])
        model_path = os.path.join(save_dir, f'model_{i}.pt')

        csv_filename = os.path.join(save_dir, f'scores_run_{i}.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_score", "val_score", "test_score"])  # Header row
            
        for epoch in tqdm(range(1, config['epochs'] + 1)):
            # train one epoch
            train(model, train_loader, criterion, optimizer, scheduler, config)

            # evaluate validation
            score = eval(model, val_loader, config)
            test_score = eval(model, test_loader, config)
            train_score = eval(model, train_loader, config)
            
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch, train_score, score, test_score])
                
            if config['optim']['scheduler'] == 'cos_anneal':
                scheduler.step()

            if config['verbose'] and config['model']['use_prompt']:
                weight = model.get_prompt_weight('softmax').data.cpu().numpy()
                cur_lr = optimizer.param_groups[-1]['lr']
                tqdm.write(
                    f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr} [{weight[0]:>4.3f} {weight[1]:>4.3f} {weight[2]:>4.3f}]")
            elif config['verbose']:
                cur_lr = optimizer.param_groups[-1]['lr']
                tqdm.write(f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr}")
            
            if config['dataset']['task'] == 'regression':
                if score < best_score:
                    best_score = score
                    #best_checkpoint = copy.deepcopy(model.state_dict())
                    #torch.save(best_checkpoint, model_path)
                    torch.save({'wrapper': model.state_dict()}, model_path)
                    print(f"Best model saved at epoch {epoch} with score {score:.4f}")
            elif config['dataset']['task'] == 'classification':
                if score > best_score:
                    best_score = score
                    #best_checkpoint = copy.deepcopy(model.state_dict())
                    torch.save({'wrapper': model.state_dict()}, model_path)
                    print(f"Best model saved at epoch {epoch} with score {score:.4f}")

        score_last_checkpoint = eval(model, test_loader, config)
        #avg_auc_last.append(score_last_checkpoint)
        if config['model']['use_prompt']:
            print('Prompt weight of last checkpoint:', model.get_prompt_weight('softmax').data.cpu())

        #model.load_state_dict(best_checkpoint)
        score_best_checkpoint = eval(model, test_loader, config)
        #avg_auc_best.append(score_best_checkpoint)
        if config['model']['use_prompt']:
            print('Prompt weight of best checkpoint:', model.get_prompt_weight('softmax').data.cpu())

        if 'CHEMBL' in config['dataset']['data_name']:
            print('[Best R2]: {:.4f} {:.4f} {:.4f}'.format(best_score, score_last_checkpoint, score_best_checkpoint))
        else:
            print('[Best score]: {:.4f}'.format(score_best_checkpoint))

    #print(avg_auc_last)
    #print('[Last] {} {}'.format(np.mean(avg_auc_last), np.std(avg_auc_last)))
    #print(avg_auc_best)
    #print('[Best] {} {}'.format(np.mean(avg_auc_best), np.std(avg_auc_best)))
    
    #res_path = os.path.join(save_dir, f"{config['dataset']['data_name']}_{config['dataset']['feat_type']}.txt")
    
    #with open(res_path, 'w') as f:
        #for i in range(len(avg_auc_best)):
            #f.write('Run #{} (seed={}): best={} last={}\n'.format(i + 1, runseeds[i], avg_auc_best[i], avg_auc_last[i]))
        #f.write('Average last score: {}\n'.format(np.mean(avg_auc_last)))
        #f.write('Average best score: {}\n'.format(np.mean(avg_auc_best)))


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        raise Exception('Number of arguments is wrong.')

    if 'CHEMBL' in sys.argv[1]:
        with open('./config/moleculeace/chembl.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['dataset']['data_name'] = sys.argv[1].split('/')[1]
    else:
        with open('./config/{}.yaml'.format(sys.argv[1]), 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv) == 3:
        config['dataset']['feat_type'] = sys.argv[2]
    print(config)
    main(config)
