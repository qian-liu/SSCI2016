import numpy as np
import random
import math
import sys
import pickle
import snn_simulator as sim
import mnist_utils as mu
import matplotlib.pyplot as plt

train_x, train_y = mu.get_train_data()
train_x /= 255.
train_label = np.zeros((train_y.size,10))
for i in range(train_y.size):
    train_label[i,train_y[i]] = 1.
train_data = np.append(train_x, train_label, axis=1)
test_x, test_y = mu.get_test_data()
test_x /= 255.
label_size = 10
test_label = np.zeros((test_y.size,label_size))


conf_file = sys.argv[1]
dbn_dir = sys.argv[2]
with open(conf_file, 'rb') as handle:
    neuron_conf = pickle.load(handle)
    train_conf = pickle.load(handle)
    test_conf = pickle.load(handle)

np.random.seed(train_conf['seed'])
train_num = train_conf['train_num']
pop_size = train_conf['net_size']#[train_x.shape[1], 500, 500, 2000, 10]
dbnet = sim.init_train(pop_size, train_conf, model='label')
result_list = []
for offset in range(0, 20000, train_num):
    dbnet = sim.train_dbn_greedy(dbnet, neuron_conf, train_conf, train_x[offset:offset+train_num], train_y[offset:offset+train_num])
    print offset+train_num
    if np.mod(offset+train_num, 1000)==0:
        dbn_file = '%s/%d.npy'%(dbn_dir, offset+train_num)
        np.save(dbn_file, dbnet)

        np.random.seed(test_conf['seed'])
        dbnet_test = sim.init_test(test_conf, dbnet)
        result = sim.dbn_test(test_x[:test_conf['batch_size']], test_y[:test_conf['batch_size']], test_conf, neuron_conf, dbnet_test)
        print len(result), result[-1]
        result_list.append( result[-1])
        plt.plot(result)
        plt.savefig('%s/ca_%d.pdf'%(dbn_dir,offset+train_num))
        dbnet = np.load(dbn_file)
        
        np.save('%s/result.npy'%dbn_dir, result_list)
    

