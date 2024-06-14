import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_graph, combine_ZA
from networks import CrossDomainNetwork, TargetDomainNetwork


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def eval(args, epoch, dataloader, toTrModels, nonTrModels, centers):

    bi_cm = np.zeros((17, 2))
    open_cm = np.zeros((17, 11))
    sr_cm = np.zeros((17, 17))


    att_cents = centers['att_cents']
    te_tgt_iter = enumerate(dataloader)
    n_batches = len(dataloader)


    cross_domain_net = CrossDomainNetwork().to(device)
    target_domain_net = TargetDomainNetwork().to(device)

    for itern in range(n_batches):
        (feats_tgt, lbls_tgt, atts_tgt, clu_tgt) = te_tgt_iter.__next__()[1]

      
        feats_tgt_var = Variable(feats_tgt.type(FloatTensor))

     
        t_z = cross_domain_net(feats_tgt_var)[0] 

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

      
        t_su_prob, t_su_pred = target_domain_net(t_f)  
        t_pred_shr_inds = t_su_pred == 0
        t_pred_unk_inds = t_su_pred == 1

       
        t_prob, t_y_pred, t_mul_dis = toTrModels['Clf'](t_f)

        if args.binary:
            threshold = 0.5
            t_att_pred[t_att_pred >= threshold] = 1
            t_att_pred[t_att_pred < threshold] = 0

        yt_pred_att = Variable(FloatTensor(t_att_pred.size(0)).fill_(-1), requires_grad=False)
        yt_pred_att_shr = nonTrModels['ProtClf'](t_att_pred.detach(), att_cents[:10, :], dist='cosine', T=0.1)[1]
        yt_pred_att_unk = nonTrModels['ProtClf'](t_att_pred.detach(), att_cents[10:, :], dist='cosine', T=0.1)[1] + 10
        yt_pred_att_shr = yt_pred_att_shr.type(FloatTensor)
        yt_pred_att_unk = yt_pred_att_unk.type(FloatTensor)
        yt_pred_att[t_pred_shr_inds] = yt_pred_att_shr[t_pred_shr_inds]
        yt_pred_att[t_pred_unk_inds] = yt_pred_att_unk[t_pred_unk_inds]

        for i in range(len(lbls_tgt)):
            yi_true = lbls_tgt[i].item()
            yi_bi_pred = t_su_pred[i].item()
            bi_cm[yi_true, yi_bi_pred] += 1.0

            yi_open_pred = t_y_pred[i].item()
            open_cm[yi_true, yi_open_pred] += 1.0

            yi_att_pred = int(yt_pred_att[i].item())
            sr_cm[yi_true, yi_att_pred] += 1.0

   
    bi_cm = bi_cm.astype(np.float) / np.sum(bi_cm, axis=1, keepdims=True)
    shr_Bi_Acc = np.sum(bi_cm[:10, 0]) / 10
    unk_Bi_Acc = np.sum(bi_cm[10:, 1]) / 7

    open_cm = open_cm.astype(np.float) / np.sum(open_cm, axis=1, keepdims=True)
    shr_Open_Acc = np.sum(open_cm[:10, :10].diagonal()) / 10
    unk_Open_Acc = np.sum(open_cm[10:, 10]) / 7

    sr_cm = sr_cm.astype(np.float) / np.sum(sr_cm, axis=1, keepdims=True)
    shr_SR_Acc = np.sum(sr_cm[:10, :10].diagonal()) / 10
    unk_SR_Acc = np.sum(sr_cm[10:, 10:].diagonal()) / 7

    Open_Acc = (shr_Open_Acc * 10 + unk_Open_Acc) / 11 * 100
    SR_Acc = (2 * shr_SR_Acc * unk_SR_Acc) / (shr_SR_Acc + unk_SR_Acc) * 100

    result = f"Ep {epoch}(s/u): {shr_Bi_Acc * 100:.2f}/{unk_Bi_Acc * 100:.2f} Open: " \
             f"OS*={shr_Open_Acc * 100:.2f}/OS^={unk_Open_Acc * 100:.2f}/OS={Open_Acc:.2f} " \
             f"SR: S={shr_SR_Acc * 100:.2f}/U={unk_SR_Acc * 100:.2f}/H={SR_Acc:.2f}"
    print(result)

    results_path = './results/' + args.att_type + '/step3/N2AwA/' + args.src + '2' + args.tgt
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    with open(results_path + '/eval_record.txt', 'a') as f:
        f.write(result + '\n')
