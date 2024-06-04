import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import argparse
# python get_history_record.py --d YAGO
# python get_history_record.py --d WIKI
# python get_history_record.py --d ICEWS18
# python get_history_record.py --d ICEWS14
# python get_history_record.py --d GDELT
parser = argparse.ArgumentParser(description='Config')
parser.add_argument("-d", "--dataset",default="ICEWS14", type=str, required=True,
                        help="dataset to use")
args = parser.parse_args()
print(args)


def load_quadruples(inPath, fileName, num_r):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([tail, rel + num_r, head, time])
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)
num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')
train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt', num_r)
dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt', num_r)
test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt', num_r)
all_data = np.concatenate((train_data, dev_data, test_data), axis=0)
all_times = np.concatenate((train_times, dev_times, test_times))

save_dir_obj = './data/{}/history_seq/'.format(args.dataset)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirs(save_dir_obj)

# With (s,r,o,t): create sparse matrix (s,r;e) of timestamp
num_r_2 = num_r * 2
row = all_data[:, 0] * num_r_2 + all_data[:, 1] # row:= s * 2|R| + r
col_rel = all_data[:, 1]    # r
d_ = np.ones(len(row))
tail_rel = sp.csr_matrix((d_, (row, col_rel)), shape=(num_e * num_r_2, num_r_2))    # -> (s,r) have r will assign 1
sp.save_npz('./data/{}/history_seq/h_r_seq_rel.npz'.format(args.dataset), tail_rel)

for idx, tim in tqdm(enumerate(all_times)):
    test_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] == tim]) # split snapshots
    # Get object_entities
    # With (s, r, o, t)
    row = test_new_data[:, 0] * num_r_2 + test_new_data[:, 1]   # row = s * 2|R| + r
    col = test_new_data[:, 2]   # o
    d = np.ones(len(row))
    tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r_2, num_e))   # -> (s,r) have o will assign 1
    sp.save_npz('./data/{}/history_seq/h_r_history_seq_{}.npz'.format(args.dataset, idx), tail_seq)

# With (s,r,o,t): create sparse matrix (s,o) of timestamp
for idx, tim in tqdm(enumerate(all_times)):
    test_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] == tim]) # split snapshots
    # Get object_entities
    row = test_new_data[:, 0]   # s
    col = test_new_data[:, 2]   # o
    d = np.ones(len(row))
    sub_obj = sp.csr_matrix((d, (row, col)), shape=(num_e, num_e))
    sp.save_npz('./data/{}/history_seq/h_t_label_history_seq_{}.npz'.format(args.dataset, idx), sub_obj)