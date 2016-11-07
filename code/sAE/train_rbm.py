import numpy as np
import random
import math
import sys
import pickle
import sim_snn as sim
import mnist_utils as mu

train_x, train_y = mu.get_train_data()
train_x /= 255.


result_dir = sys.argv[1]
conf_file = '%sconf.pickle'%result_dir
with open(conf_file, 'rb') as handle:
    neuron_conf = pickle.load(handle)
    train_conf = pickle.load(handle)
    test_conf = pickle.load(handle)

label_size = train_conf['net_size'][-1]
train_label = np.zeros((train_y.size, 10))
train_label[range(train_y.size), train_y.astype(int)] = 1.
train_label = np.tile(train_label, label_size/10)

#np.random.seed(train_conf['seed'])
train_num = train_conf['train_num']
dbnet = sim.init_train(train_conf, model='label')

start_e = int(sys.argv[2])
start_o = int(sys.argv[3])
start_l = int(sys.argv[4])
print 'Start training from epoch:%d, offset:%d, layer:%d'%(start_e, start_o, start_l)
dbnet = sim.train_dbn_greedy(neuron_conf, train_conf, train_x, train_y=train_y, w_dir=result_dir, start_e=start_e, start_o=start_o, curr_l=start_l) #train_x[:1000], train_y=train_y[:1000]
