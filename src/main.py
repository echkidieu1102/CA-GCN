import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.sparse as sp
# sys.path.append("") # DEBUG
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph, add_inverse_rel
from src.rrgcn import RecurrentRGCN
import torch.nn.modules.rnn
from rgcn.knowledge_graph import _read_triplets_as_list
from collections import defaultdict

import warnings
warnings.filterwarnings(action='ignore')


def test(model, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, model_name, static_graph, static_len, all_tail_seq_dict, all_label_seq_dict, mode):
    '''
    :param model: model used to test
    :param history_list: valid: train; test: train+valid
    :param test_list: valid: valid; test: test
    :param num_rels: number of relations
    :param num_nodes: number of nodes
    :param use_cuda:
    :param all_ans_list: dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph:
    :param static_len:
    :param mode:
    :return:
    '''
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    history_time_last = len(history_list) - 1
    test_time_list = [idx for idx in range(history_time_last+1, history_time_last+1+len(test_list))]
    num_rel_2 = num_rels * 2

    # Use a sparse matrix to represent the frequency of a static graph.
    if args.add_static_graph:
        all_static_tail_seq = all_tail_seq_dict[static_len-1]
        

    # With snapshot, create sparse matrix (s,r;r) 
    acc_mask_all = []
    for time_idx, test_snap in enumerate(tqdm(test_list)): 
        triple_with_inverse = add_inverse_rel(test_snap, num_rels)
        seq_idx = triple_with_inverse[:, 0] * num_rel_2 + triple_with_inverse[:, 1] # (s,r) have r
        seq_label_idx = triple_with_inverse[:, 0]
        current_timestamp_idx = test_time_list[time_idx]

        if time_idx - args.test_history_len < 0:
            input_list = [snap for snap in history_list[time_idx - args.test_history_len:]] + [snap for snap in test_list[0: time_idx]]
        else:
            input_list = [snap for snap in test_list[time_idx - args.test_history_len: time_idx]]

        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        
        # Get the frequency from 0 to the current timestamp and create a mask for appearances in history.
        history_tail_seq, one_hot_tail_seq, one_hot_label_seq, one_hot_trust_label = None, None, None, None

        # # START
        if mode == "test":
        # # Create frequency
            all_tail_seq = all_tail_seq_dict[current_timestamp_idx-1]
            history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())    # convert to dense matrix
            if args.add_static_graph:
                static_tail_seq = torch.Tensor(all_static_tail_seq[seq_idx].todense())
                one_hot_tail_seq = static_tail_seq.masked_fill(static_tail_seq != 0, 1)
            else:
                one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)

            # Trust label
            trust_label = all_tail_seq_dict[current_timestamp_idx]
            trust_label_seq = torch.Tensor(trust_label[seq_idx].todense())
            one_hot_trust_label = trust_label_seq.masked_fill(trust_label_seq != 0, 1)

            # Label
            all_label_seq = all_label_seq_dict[current_timestamp_idx-1]
            history_label_seq = torch.Tensor(all_label_seq[seq_label_idx].todense()) # FIX
            # history_label_seq = torch.Tensor(all_label_seq.todense())
            one_hot_label_seq = history_label_seq.masked_fill(history_label_seq != 0, 1)

            if use_cuda:
                history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                one_hot_label_seq = one_hot_label_seq.to(args.gpu)
                one_hot_trust_label = one_hot_trust_label.to(args.gpu)

        # # END
        # (tensor)all_triples: (batch_size, 3); (tensor)score: (batch_size, num_ents)
        test_triples, final_score, acc_mask = model.predict(history_glist, num_rels, static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, one_hot_trust_label, test_triples_input, use_cuda, mode)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000)

        if acc_mask is not None:
            acc_mask_all.append(acc_mask)
        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    if mode == 'test':
        print("Acc mask: {:.4f}".format(np.mean(acc_mask_all)))
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    return mrr_raw, mrr_filter


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset) # DEBUG
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    data_list = train_list + valid_list + test_list
    timestamp_len = len(data_list)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_rel_2 = num_rels * 2

    # for time-aware filtered evaluation
    all_ans_list_test_time_filter = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes) # data.test: np.array([[s, r, o, time], []...])
    all_ans_list_valid_time_filter = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes)

    model_name = "{}-{}-{}-ly{}-dilate{}-his{}-dp{}|{}|{}|{}-gpu{}"\
        .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len,
                args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    # model_dir = './models_debug/' # DEBUG
    model_dir = '../models/'
    model_state_file = model_dir + model_name
    # model_state_file = './models/' + model_name
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        num_static_rels,
                        num_words,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        aggregation=args.aggregation,
                        use_static=args.add_static_graph,
                        entity_prediction=args.entity_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        analysis = args.run_analysis,
                        alpha = args.alpha,
                        linear_classifier_mode = args.linear_classifier_mode)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # CREATE 
    all_tail_seq_dict = {}
    all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes * num_rel_2, num_nodes))
    for time_idx in tqdm(range(0, timestamp_len), desc="Load tail seq"):
        tim_tail_seq = sp.load_npz('../data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, time_idx))
        # tim_tail_seq = sp.load_npz('./data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, time_idx)) # DEBUG
        all_tail_seq = all_tail_seq + tim_tail_seq
        all_tail_seq_dict[time_idx] = all_tail_seq
    
    # CREATE LABELS:
    all_label_seq_dict = {}
    all_label_seq = sp.csr_matrix(([], ([], [])), shape=(num_nodes, num_nodes))
    for time_idx in tqdm(range(0, timestamp_len), desc="Load label seq"):
        tim_label_seq = sp.load_npz('../data/{}/history_seq/h_t_label_history_seq_{}.npz'.format(args.dataset, time_idx))
        # tim_label_seq = sp.load_npz('./data/{}/history_seq/h_t_label_history_seq_{}.npz'.format(args.dataset, time_idx)) # DEBUG
        all_label_seq = all_label_seq + tim_label_seq
        all_label_seq_dict[time_idx] = all_label_seq

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter = test(model,
                                   train_list+valid_list,
                                   test_list,
                                   num_rels,
                                   num_nodes,
                                   use_cuda,
                                   all_ans_list_test_time_filter,
                                   model_state_file,
                                   static_graph,
                                   timestamp_len,
                                   all_tail_seq_dict,
                                   all_label_seq_dict,
                                   mode="test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if args.load_pretrain and os.path.exists(model_state_file):
            if use_cuda:
                checkpoint = torch.load(model_state_file, map_location=torch.device(args.gpu))
            else:
                checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
            print("Load Model name: {}. Using best epoch : {}".format(model_state_file, checkpoint['epoch']))  # use best stat checkpoint
            print("\n"+"-"*10+"load checkpoint"+"-"*10+"\n")
            model.load_state_dict(checkpoint['state_dict'])

        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx, desc="Timestamp"):
                if train_sample_num == 0: continue

                output = train_list[train_sample_num:train_sample_num+1]

                triple_with_inverse = add_inverse_rel(output[0], num_rels)
                seq_idx = triple_with_inverse[:, 0] * num_rel_2 + triple_with_inverse[:, 1]
                seq_label_idx = triple_with_inverse[:, 0] # FIX

                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                all_tail_seq = all_tail_seq_dict[train_sample_num - 1]
                history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                # history_tail_seq, one_hot_tail_seq = None, None

                # Trust label
                trust_label = all_tail_seq_dict[train_sample_num]
                trust_label_seq = torch.Tensor(trust_label[seq_idx].todense())
                one_hot_trust_label = trust_label_seq.masked_fill(trust_label_seq != 0, 1)

                # Label
                all_label_seq = all_label_seq_dict[train_sample_num - 1]
                history_label_seq = torch.Tensor(all_label_seq[seq_label_idx].todense()) # FIX
                # history_label_seq = torch.Tensor(all_label_seq.todense())
                one_hot_label_seq = history_label_seq.masked_fill(history_label_seq != 0, 1)

                if use_cuda:
                    history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                    one_hot_label_seq = one_hot_label_seq.to(args.gpu)
                    one_hot_trust_label = one_hot_trust_label.to(args.gpu)

                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                loss_e = model.get_loss(history_glist, output[0], static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, one_hot_trust_label, use_cuda)
                loss = loss_e

                losses.append(loss.item())
                losses_e.append(loss_e.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity:{:.4f} Best MRR {:.4f} | Model {} ".format(epoch, np.mean(losses), np.mean(losses_e), best_mrr, model_name))

            # validation
            if epoch and (epoch+1) % args.evaluate_every == 0:
                mrr_raw, mrr_filter = test(model,
                                           train_list,
                                           valid_list,
                                           num_rels,
                                           num_nodes,
                                           use_cuda,
                                           all_ans_list_valid_time_filter,
                                           model_state_file,
                                           static_graph,
                                           timestamp_len,
                                           all_tail_seq_dict,
                                           all_label_seq_dict,
                                           mode="train")
                # entity prediction evalution
                if mrr_raw < best_mrr:
                    if epoch >= args.n_epochs:
                        break
                else:
                    best_mrr = mrr_raw
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        if args.linear_classifier:
            if use_cuda:
                checkpoint = torch.load(model_state_file, map_location=torch.device(args.gpu))
            else:
                checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))
            print("Load Model name: {}. Using best epoch : {}".format(model_state_file, checkpoint['epoch']))  # use best stat checkpoint
            print("\n"+"-"*10+"start linear classifier"+"-"*10+"\n")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_linear = torch.optim.Adam(model.parameters(), lr=args.linear_lr, weight_decay=1e-5)
            model.freeze_parameter()  # freeze parameter except linear

            for epoch in range(args.n_epochs_classifier):
                total_linear_loss = 0
                model.train()
                losses = []
                # losses_e = []

                idx = [_ for _ in range(len(train_list))]
                random.shuffle(idx)

                for train_sample_num in tqdm(idx, desc="Linear Classifier"):
                    if train_sample_num == 0: continue

                    output = train_list[train_sample_num:train_sample_num+1]

                    triple_with_inverse = add_inverse_rel(output[0], num_rels)
                    seq_idx = triple_with_inverse[:, 0] * num_rel_2 + triple_with_inverse[:, 1]
                    seq_label_idx = triple_with_inverse[:, 0] # FIX

                    if train_sample_num - args.train_history_len<0:
                        input_list = train_list[0: train_sample_num]
                    else:
                        input_list = train_list[train_sample_num - args.train_history_len: train_sample_num]

                    all_tail_seq = all_tail_seq_dict[train_sample_num - 1]
                    history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                    # history_tail_seq, one_hot_tail_seq = None, None

                    # Trust label
                    trust_label = all_tail_seq_dict[train_sample_num]
                    trust_label_seq = torch.Tensor(trust_label[seq_idx].todense())
                    one_hot_trust_label = trust_label_seq.masked_fill(trust_label_seq != 0, 1)

                    # Label
                    all_label_seq = all_label_seq_dict[train_sample_num - 1]
                    history_label_seq = torch.Tensor(all_label_seq[seq_label_idx].todense()) # FIX
                    # history_label_seq = torch.Tensor(all_label_seq.todense())
                    one_hot_label_seq = history_label_seq.masked_fill(history_label_seq != 0, 1)

                    if use_cuda:
                        history_tail_seq, one_hot_tail_seq = history_tail_seq.to(args.gpu), one_hot_tail_seq.to(args.gpu)
                        one_hot_label_seq = one_hot_label_seq.to(args.gpu)
                        one_hot_trust_label = one_hot_trust_label.to(args.gpu)
                    
                    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                    output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                    loss_linear = model.get_loss_classifier(history_glist, output[0], static_graph, history_tail_seq, one_hot_tail_seq, one_hot_label_seq, one_hot_trust_label, use_cuda)
                    if loss_linear is not None:
                        error = loss_linear
                        losses.append(loss_linear.item())
                    else:
                        continue
                    error.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                    optimizer_linear.step()
                    optimizer_linear.zero_grad()
                    total_linear_loss += error.item()
                print("Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} ".format(epoch, np.mean(losses), best_mrr, model_name))
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

        mrr_raw, mrr_filter = test(model,
                                   train_list+valid_list,
                                   test_list,
                                   num_rels,
                                   num_nodes,
                                   use_cuda,
                                   all_ans_list_test_time_filter,
                                   model_state_file,
                                   static_graph,
                                   timestamp_len,
                                   all_tail_seq_dict,
                                   all_label_seq_dict,
                                   mode="test")
    return mrr_raw, mrr_filter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="alpha for nceloss")
    
    #
    parser.add_argument("--n-epochs-classifier", type=int, default=20,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--linear-classifier", action='store_true', default=True,
                        help="train classifier to create mask")
    parser.add_argument("--linear-lr", type=float, default=0.001,
                        help="learning rate linear classifier")
    parser.add_argument("--linear_classifier_mode", type=str, default='divergent', help="convergent and divergent mode for sLinear Classifier")
    parser.add_argument("--load-pretrain", action='store_true', default=False,
                        help="loading checkpoint, continuing training.")

    # configuration for encoder RGCN stat
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")


    args = parser.parse_args()
    print(args)
    run_experiment(args)
    sys.exit()