import pickle
import sys
conf_file = sys.argv[1]

def print_dict(conf, name):
    print name, ' = {'
    for x in conf:
        print '\t', x, ':', conf[x]
    print '}'

with open(conf_file, 'rb') as handle:
    neuron_conf = pickle.load(handle)
    train_conf = pickle.load(handle)
    test_conf = pickle.load(handle)
    
print_dict(neuron_conf, 'neuron_conf')
print_dict(train_conf, 'train_conf')
print_dict(test_conf, 'test_conf')
