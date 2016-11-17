#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys
import pickle
import sim_snn as sim
import mnist_utils as mu


test_x, test_y = mu.get_test_data()
test_x /= 255.
label_size = 10
test_label = np.zeros((test_y.size,label_size))

result_dir = sys.argv[1]
conf_file = '%sconf.pickle'%result_dir
log_file = '%slog.txt'%result_dir
logf = open(log_file, 'a+')
with open(conf_file, 'rb') as handle:
    neuron_conf = pickle.load(handle)
    train_conf = pickle.load(handle)
    test_conf = pickle.load(handle)

#test_conf['batch_size'] = 10
train_all = 60000
#step = 1000
train_list =  [1000, 10000, 20000, 30000, 40000, 50000, 60000]
dur = test_conf['duration']
np.random.seed(test_conf['seed'])
train_l = 0
for epoch in range(0,train_conf['epoch']):
    for train_num in train_list[:]:
        dbn_file = '%s%d_%d_%d.npy'%(result_dir, epoch, train_num, train_l)
        print epoch, train_num
        dbnet = np.load(dbn_file)
        dbnet_test = sim.init_test(test_conf, dbnet)
        result = sim.dbn_test(dbnet_test, neuron_conf, test_x[:test_conf['batch_size']], test_conf, model='label', test_y=test_y[:test_conf['batch_size']])
        str = 'epoch:%d, offset:%d, CA = %.2f\n'%(epoch, train_num, result[-1])
        logf.write(str) 
        #plt.clf()
        #plt.plot(result[-dur:])
        #plt.savefig('%s%d.pdf'%(result_dir,train_num+step ))
   
    result = np.array(result)
    result = np.reshape(result, (len(train_list), dur))
    np.save('%sresult_%d_%d.npy'%(result_dir,epoch, train_list[-1]), result)
    test_conf['result'] = []
logf.close()
