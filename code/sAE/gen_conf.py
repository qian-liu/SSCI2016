import numpy as np
import pickle
import sys

conf_dir = sys.argv[1]
neuron_conf = {
    'v_reset' : 0., #mV
    'v_thresh' : 1., #MV
    'tau_refr' : 0. #ms
}


eta = 0.0001
tau_stdp = 10
delta_w =  eta*np.ones(tau_stdp)#np.linspace(1,0,tau_stdp)
delta_w[-1] = 0
rate = 100.
method = 'max'
train_conf = {
    'net_size' : [784, 500, 10],
    'eta' :  eta,
    'tau_stdp' : tau_stdp,
    'delta_w' : delta_w,
    'batch_size' : 1,
    'epoch' : 3,
    'duration' : 100, #500
    'rate' : rate,
    'method' : method,
    'random' : 'normal',
    'mu' : 0.,
    'sigma' : 0.01,
    'seed' : 0,
    'train_num' : 100
}

test_conf = {
    'batch_size' : 10000, #10000
    'duration' : 1000,
    'rate' : rate,
    'method' : method,
    'seed' : 0,
    'result' : []
}

conf_file = '%sconf.pickle'%conf_dir
with open(conf_file, 'wb') as handle:
    pickle.dump(neuron_conf, handle)
    pickle.dump(train_conf, handle)
    pickle.dump(test_conf, handle)
