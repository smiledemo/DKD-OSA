import time
import torch
import os
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import uniform, normal
import pandas as pd
from utils import *
from evaluation import *

cuda = True if torch.cuda.is_available() else False  # 修改为根据是否有GPU来设置
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else LongTensor


def train_step3(args, dataloaders, toTrModels, nonTrModels, lossFunctions, optimizers, centers):
    att_cents = centers['att_cents']

    tr_src_iter = enumerate(dataloaders['tr_loader_src'])
    tr_tgt_iter = enumerate(dataloaders['tr_loader_tgt'])
    larger_data, larger_batches = count_epoch_on_large_dataset(
        train_loader_source=dataloaders['tr_loader_src'],
        train_loader_target=dataloaders['tr_loader_tgt'])
    n_epochs = args.epochs  

    for epoch in range(n_epochs):
        print("Training Epoch {}/{}...".format(epoch, n_epochs))
        if (epoch + 1) < 2:  


            centers['zt_clu_cents'], centers['at_clu_cents'] = get_clu_centers(args, toTrModels, dataloaders, centers, label='init')  
        else:
            centers['zt_clu_cents'], centers['at_clu_cents'] = get_clu_centers(args, toTrModels, dataloaders, centers, label='clf')  
        lam = 2 / (1 + math.exp(-1 * 10 * epoch / n_epochs)) - 1  


        for key in toTrModels:
            toTrModels[key].train()

        for key in optimizers:
            adjust_learning_rate(optimizers[key], epoch, args)


        for itern in range(larger_batches):
            try:
                (feats_tgt, _, _, clu_tgt) = tr_tgt_iter.__next__()[1]
            except StopIteration:
                tr_tgt_iter = enumerate(dataloaders['tr_loader_tgt'])
                (feats_tgt, _, _, clu_tgt) = tr_tgt_iter.__next__()[1]
            try:
                (feats_src, lbls_src, atts_src, clu_src) = tr_src_iter.__next__()[1]
            except StopIteration:
                tr_src_iter = enumerate(dataloaders['tr_loader_src'])
                (feats_src, lbls_src, atts_src, clu_src) = tr_src_iter.__next__()[1]

            feats_src_var = Variable(feats_src.type(FloatTensor))
            lbls_src_var = Variable(lbls_src.type(LongTensor))
            atts_src_var = Variable(atts_src.type(FloatTensor))
            feats_tgt_var = Variable(feats_tgt.type(FloatTensor))
            clu_tgt_var = Variable(clu_tgt.type(LongTensor))

            ''' -----------------------source-------------------------------------'''

            s_z = toTrModels['GenZ'](feats_src_var)
            if args.GenA_type == 'GCN':
                s_vertices = feats_src_var
                s_adj = get_graph(a=s_vertices, b=s_vertices, dist='euclidean', alpha=0.2, graph_type='adjacency')
                s_att_pred = toTrModels['GenA'](s_z, s_adj)
            else:
                s_att_pred = toTrModels['GenA'](s_z)
                propagator = get_graph(s_z, s_z, dist='euclidean', alpha=0.2)
                propagator = F.normalize(propagator, p=1, dim=1)
                s_att_pred = torch.mm(propagator, s_att_pred)

            if args.combine_za:
                s_f = combine_ZA(s_z, atts_src_var)
                s_f_ = combine_ZA(s_z, s_att_pred)
                s_prob, s_y_pred, _ = toTrModels['Clf'](s_f)  # (NxC)(Nx1)(Nx1)
                s_prob_, s_y_pred_, _ = toTrModels['Clf'](s_f_)  # (NxC)(Nx1)(Nx1)
            else:
                s_f = s_z
                s_prob, s_y_pred, _ = toTrModels['Clf'](s_f)  # (NxC)(Nx1)(Nx1)


            loss_s_clf = lossFunctions['CELoss'](s_prob, lbls_src_var)

            loss_s_att = lossFunctions['BCELoss'](s_att_pred, atts_src_var)

            if args.combine_za:
                loss_s_clf_ = lossFunctions['CELoss'](s_prob_, lbls_src_var)
            else:
                pass

            ''' ---------------target------------------------------------------'''

            t_z = toTrModels['GenZ'](feats_tgt_var)
            if args.GenA_type == 'GCN':
                t_vertices = feats_tgt_var  
                t_adj = get_graph(a=t_vertices, b=t_vertices, dist='euclidean', alpha=0.2, graph_type='adjacency')
                t_att_pred = toTrModels['GenA'](t_z, t_adj)
            else:
                t_att_pred = toTrModels['GenA'](t_z)
                propagator = get_graph(t_z, t_z, dist='euclidean', alpha=0.2)
                propagator = F.normalize(propagator, p=1, dim=1)
                t_att_pred = torch.mm(propagator, t_att_pred)

            if args.combine_za:
                t_f = combine_ZA(t_z, t_att_pred)
            else:
                t_f = t_z
            t_prob, t_y_pred, _ = toTrModels['Clf'](t_f)  
            t_su_prob, t_su_pred = toTrModels['ClfSU'](t_f)  


            t_conf_shr_inds = clu_tgt_var < 10  
            t_conf_unk_inds = clu_tgt_var >= 10  

            yt_conf_shr = clu_tgt_var[t_conf_shr_inds]  
            yt_conf_unk = clu_tgt_var[t_conf_unk_inds] * 0 + 10

            at_conf_shr = att_cents[yt_conf_shr]

            bit_conf_shr = Variable(LongTensor(t_conf_shr_inds.sum().item()).fill_(0.0), requires_grad=False)
            bit_conf_unk = Variable(LongTensor(t_conf_unk_inds.sum().item()).fill_(1.0), requires_grad=False)


            t_z_conf_shr = t_z[t_conf_shr_inds]
            if args.combine_za:
                t_f_conf_shr = combine_ZA(t_z_conf_shr, at_conf_shr)
                t_prob_conf_shr, t_y_pred_conf_shr, _ = toTrModels['Clf'](t_f_conf_shr)  # (NxC)(Nx1)(Nx1)
                t_su_prob_conf_shr, t_su_pred_conf_shr = toTrModels['ClfSU'](t_f_conf_shr)
            else:
                pass


            loss_t_att_conf = lossFunctions['BCELoss'](t_att_pred[t_conf_shr_inds], at_conf_shr)

            loss_t_clf_conf_shr = lossFunctions['CELoss'](t_prob[t_conf_shr_inds], yt_conf_shr)
            if args.combine_za:
                loss_t_clf_conf_shr_ = lossFunctions['CELoss'](t_prob_conf_shr, yt_conf_shr)
            else:
                pass  
            loss_t_clf_conf_unk = lossFunctions['CELoss'](t_prob[t_conf_unk_inds], yt_conf_unk)


            loss_t_ClfSU = lossFunctions['CELoss'](t_su_prob[t_conf_shr_inds], bit_conf_shr) \
                           + 1 * lossFunctions['CELoss'](t_su_prob[t_conf_unk_inds], bit_conf_unk)

            if args.combine_za:
                loss_t_ClfSU_ = lossFunctions['CELoss'](t_su_prob_conf_shr, bit_conf_shr)
            else:
                pass

            ''' --------------- Domain alignment ------------------------------------------'''

            z_clu_cents = centers['zt_clu_cents'].detach()
            s_z_dist = get_dist_map(a=s_z, b=z_clu_cents, dist='euclidean')
            s_mask = get_oneHot(lbls_src_var, args.tgt_nc)
            loss_s_center = torch.sum(s_z_dist * s_mask) / torch.sum(s_mask) - 0.1 * torch.sum(s_z_dist * (1 - s_mask)) / torch.sum((1 - s_mask))

            t_z_dist = get_dist_map(a=t_z, b=z_clu_cents, dist='euclidean')
            t_mask = get_oneHot(clu_tgt_var, args.tgt_nc)
            loss_t_center = torch.sum(t_z_dist * t_mask) / torch.sum(t_mask) - 0.1 * torch.sum(t_z_dist * (1 - t_mask)) / torch.sum((1 - t_mask))

            loss_center = loss_s_center + loss_t_center

            
            loss_weights = dynamic_weight_adjustment(args, epoch, loss_s_clf, loss_s_att, loss_t_clf_conf_shr, loss_t_clf_conf_unk, loss_center)

            training_opts = [optimizers['opt_Clf'], optimizers['opt_GenZ'], optimizers['opt_GenA'], optimizers['opt_ClfSU']]
            for opt in training_opts:
                opt.zero_grad()
            if args.combine_za:
                loss = (loss_weights['loss_s_clf'] * loss_s_clf +
                        loss_weights['loss_s_clf_'] * loss_s_clf_ +
                        loss_weights['loss_s_att'] * loss_s_att +
                        loss_weights['loss_t_clf_conf_shr'] * loss_t_clf_conf_shr +
                        loss_weights['loss_t_clf_conf_shr_'] * loss_t_clf_conf_shr_ +
                        loss_weights['loss_t_clf_conf_unk'] * loss_t_clf_conf_unk +
                        loss_weights['loss_t_att_conf'] * loss_t_att_conf +
                        loss_weights['loss_t_ClfSU'] * loss_t_ClfSU +
                        loss_weights['loss_t_ClfSU_'] * loss_t_ClfSU_ +
                        loss_weights['loss_center'] * loss_center)
            else:
                loss = (loss_weights['loss_s_clf'] * loss_s_clf +
                        loss_weights['loss_s_att'] * loss_s_att +
                        loss_weights['loss_t_clf_conf_shr'] * loss_t_clf_conf_shr +
                        loss_weights['loss_t_clf_conf_unk'] * loss_t_clf_conf_unk +
                        loss_weights['loss_t_att_conf'] * loss_t_att_conf +
                        loss_weights['loss_t_ClfSU'] * loss_t_ClfSU)
            loss.backward()
            for opt in training_opts:
                opt.step()

        print("Evaluate epoch = ", epoch)
        eval(args, epoch, dataloaders['te_loader_tgt'], toTrModels, nonTrModels, centers)

    for m in toTrModels:
        save_path = './saved_weights/' + args.att_type + '/step3/N2AwA/'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        file_name = save_path + args.src + '2' + args.tgt + '_' + m + '.pth'
        torch.save(toTrModels[m].state_dict(), file_name)
    return

def dynamic_weight_adjustment(args, epoch, loss_s_clf, loss_s_att, loss_t_clf_conf_shr, loss_t_clf_conf_unk, loss_center):
    total_loss = loss_s_clf + loss_s_att + loss_t_clf_conf_shr + loss_t_clf_conf_unk + loss_center
    loss_weights = {
        'loss_s_clf': 1.0 * (loss_s_clf / total_loss).item(),
        'loss_s_att': 1.0 * (loss_s_att / total_loss).item(),
        'loss_t_clf_conf_shr': 1.0 * (loss_t_clf_conf_shr / total_loss).item(),
        'loss_t_clf_conf_unk': 1.0 * (loss_t_clf_conf_unk / total_loss).item(),
        'loss_center': 1.0 * (loss_center / total_loss).item()
    }
    if args.combine_za:
        loss_weights['loss_s_clf_'] = 1.0 * (loss_s_clf / total_loss).item()
        loss_weights['loss_t_clf_conf_shr_'] = 1.0 * (loss_t_clf_conf_shr / total_loss).item()
        loss_weights['loss_t_att_conf'] = 1.0 * (loss_t_att_conf / total_loss).item()
        loss_weights['loss_t_ClfSU'] = 1.0 * (loss_t_ClfSU / total_loss).item()
        loss_weights['loss_t_ClfSU_'] = 1.0 * (loss_t_ClfSU_ / total_loss).item()
    else:
        loss_weights['loss_t_att_conf'] = 1.0 * (loss_t_att_conf / total_loss).item()
        loss_weights['loss_t_ClfSU'] = 1.0 * (loss_t_ClfSU / total_loss).item()
    return loss_weights

def self_supervised_training(network, dataloader, device):
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    for data in dataloader:
        inputs = data[0].to(device)
        perturbation_strength = 0.1
        perturbed_inputs = inputs + perturbation_strength * torch.randn_like(inputs).to(device)
        features, outputs = network(perturbed_inputs)
        loss = F.mse_loss(outputs, network(inputs)[1])  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
