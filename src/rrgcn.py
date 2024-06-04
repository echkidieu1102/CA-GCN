import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE

import sys
import scipy.sparse as sp
sys.path.append("..")

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),
                                    nn.Dropout(0.4),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)
    
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers): # n_layers = 2 UnionRGCNLayer
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', use_static=False,
                 entity_prediction=False, use_cuda=False, gpu = 0, analysis=False, alpha=0.2, linear_classifier_mode="soft"):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name # convtranse
        self.encoder_name = encoder_name # uvrgcn
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.use_static = use_static
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.alpha = alpha
        self.linear_classifier_mode = linear_classifier_mode

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        # # Contrastive Learning
        self.linear_frequency = nn.Linear(self.num_ents, self.h_dim)
        self.contrastive_hidden_layer = nn.Linear(2 * self.h_dim, self.h_dim)
        self.linear_classifier_layer = LinearClassifier(2 * self.h_dim, 1)
        self.linear_classifier_layer.apply(self.weights_init)
        self.weights_init(self.linear_frequency)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_entities=num_ents, embedding_dim=h_dim, gpu=gpu, layer_norm=layer_norm,
                                         input_dropout=input_dropout, hidden_dropout=hidden_dropout, feature_map_dropout=feat_dropout)
        else:
            raise NotImplementedError 
        
        self.tanh = nn.Tanh()
        self.crossEntropy = nn.BCELoss()
    
    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
    
    def linear_classifier_loss(self, actor1, r, combine_emb, history_label, frequency_hidden):
        # history_label_pred = F.sigmoid(
        #     self.linear_classifier_layer(torch.cat((self.dynamic_emb[actor1],self.emb_rel[r], frequency_hidden), dim=1)))
        history_label_pred = F.sigmoid(
            self.linear_classifier_layer(
                torch.cat((combine_emb, frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        if history_label_pred.size(0)==1:
          tmp_label = torch.unsqueeze(tmp_label, 0)
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label)) + 1e-5
        return ce_loss, history_label_pred, ce_accuracy

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.linear_classifier_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param
    
    def freeze_parameter(self):
        self.w1.requires_grad_(False)
        self.w2.requires_grad_(False)
        self.emb_rel.requires_grad_(False)
        self.dynamic_emb.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.rgcn.requires_grad_(False)
        self.relation_cell_1.requires_grad_(False)
        if self.decoder_name == "convtranse":
            self.decoder_ob.requires_grad_(False)

    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x

    def calculate_spc_loss(self, actor1, r, combine_emb,  targets, frequency_hidden):
        # projections = self.contrastive_layer(
        #     torch.cat((self.dynamic_emb[actor1], self.emb_rel[r], frequency_hidden), dim=1))
        projections = self.contrastive_layer(
            torch.cat((combine_emb, frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.gpu)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.gpu)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        static_emb = None

        history_embs = []
        rel_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            # self.h: (num_ents, h_dim); g.r_to_e: (node)relation to edge
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            # g.r_len: [(0, 4), (4, 9), ...]
            # g.uniq_r: [r0, r1, ..., r0', r1', ...]
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            # emb_rel: (num_rels*2, h_dim)
            # x_input: (num_rels*2, h_dim)represents the aggregated edge embeddings related to the nodes connected by the edges.

            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
                rel_embs.append(self.h_0)
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1) # (num_rels*2, h_dim*2)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0 # self.h_0: (num_rels*2, h_dim)
                rel_embs.append(self.h_0)

            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            # self.h = time_weight * current_h + (1-time_weight) * self.h 
            self.h = current_h # self.h: (num_ents, h_dim)
            history_embs.append(self.h)
        return history_embs, static_emb, rel_embs, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, test_triplets, use_cuda, mode):
        '''
        :param test_graph:
        :param num_rels:
        :param static_graph:
        :param history_tail_seq:
        :param one_hot_tail_seq
        :param test_triplets: (num_triples_time, 3)
        :param use_cuda:
        :return:
        '''
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets)) # (batch_size, 3)
            s = all_triples[:,0]
            r = all_triples[:,1]
            o = all_triples[:,2]
            classifier_loss = None
            
            evolve_embs, _, r_embs, _, _ = self.forward(test_graph, static_graph, use_cuda)
            score, combine_info = self.decoder_ob.forward(evolve_embs, r_embs, all_triples, history_tail_seq, one_hot_tail_seq, self.layer_norm, use_cuda, mode="test")
            
            if mode =="test":
                labels = []
                for i, col in enumerate(all_triples[:, 2]):
                    labels.append(one_hot_label_seq[i,col].item())
                labels = torch.Tensor(labels).to(self.gpu).unsqueeze(1)

                history_id = []
                for i in range(len(one_hot_label_seq)):
                    col_idx = np.where(one_hot_label_seq[i].cpu() == 1)[0]
                    history_id.append(col_idx)

                softmax_frequency = F.softmax(history_tail_seq, dim=1)
                frequency_hidden = self.tanh(self.linear_frequency(softmax_frequency))

                classifier_loss, pred, acc = self.linear_classifier_loss(s, r, combine_info, labels, frequency_hidden)
                mask = (torch.zeros(all_triples.shape[0], self.num_ents)).to(self.gpu)
                for i in range(all_triples.shape[0]):
                    if pred[i].item() > 0.5:
                        mask[i, history_id[i]] = 1
                    else:
                        mask[i, :] = 1
                        mask[i, history_id[i]] = 0
#                mask = one_hot_label_seq
                mask = torch.tensor(np.array(one_hot_label_seq.cpu() == 0, dtype=float)).to(self.gpu)
                # mask = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float)).to(self.gpu)
                if self.linear_classifier_mode == 'soft':
                    mask = F.softmax(mask, dim=1)
                score = torch.mul(score, mask)
            
            return all_triples, score, classifier_loss # (batch_size, 3) (batch_size, num_ents)


    def get_loss(self, glist, triples, static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph:
        :param history_tail_seq:
        :param one_hot_tail_seq
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        s = all_triples[:,0]
        r = all_triples[:,1]

        evolve_embs, static_emb, r_embs, _, _ = self.forward(glist, static_graph, use_cuda)
        if self.entity_prediction:
            scores_ob, combine_info = self.decoder_ob.forward(evolve_embs, r_embs, all_triples, history_tail_seq=None, one_hot_tail_seq=None, layer_norm=self.layer_norm, use_cuda=use_cuda)
            scores_ob = scores_ob.view(-1, self.num_ents)
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        # Contrastive learning
        labels = []
        for i, col in enumerate(all_triples[:, 2]):
            labels.append(one_hot_label_seq[i,col].item())
        labels = torch.Tensor(labels).to(self.gpu).unsqueeze(1)
        softmax_frequency = F.softmax(history_tail_seq, dim=1)
        frequency_hidden = self.tanh(self.linear_frequency(softmax_frequency))
        spc_loss = self.calculate_spc_loss(s, r, combine_info, labels, frequency_hidden)
        
        loss = loss_ent*self.alpha + spc_loss*(1-self.alpha)
        return loss
        
    def get_loss_classifier(self, glist, triples, static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, use_cuda):
        loss = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)
        s = all_triples[:,0]
        r = all_triples[:,1]

        evolve_embs, static_emb, r_embs, _, _ = self.forward(glist, static_graph, use_cuda)
        if self.entity_prediction:
            scores_ob, combine_emb = self.decoder_ob.forward(evolve_embs, r_embs, all_triples, history_tail_seq=None, one_hot_tail_seq=None, layer_norm=self.layer_norm, use_cuda=use_cuda)

        labels = []
        for i, col in enumerate(all_triples[:, 2]):
            labels.append(one_hot_label_seq[i,col].item())
        labels = torch.Tensor(labels).to(self.gpu).unsqueeze(1)
        softmax_frequency = F.softmax(history_tail_seq, dim=1)
        frequency_hidden = self.tanh(self.linear_frequency(softmax_frequency))

        classifier_loss, _, _ = self.linear_classifier_loss(s, r, combine_emb, history_label=labels, frequency_hidden=frequency_hidden)
        loss+=classifier_loss
        return loss + self.oracle_l1(0.01)
