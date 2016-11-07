import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import random
import math

def init_layer(layer_class, v_size, h_size, b_size, w_down):
    layer = {}
    layer['class'] = layer_class #'l':normal layers; 'top': top layers with labels
    layer['vis_size'] = v_size
    layer['hid_size'] = h_size
    if layer_class == 'top':
        vall_size =  sum(layer['vis_size'])
    else:
        vall_size = layer['vis_size']
    w = np.reshape(w_down, (layer['hid_size'],vall_size))
    layer['w_down'] = w
    layer['w_up'] = np.transpose(w)
    layer['vis_v'] = np.zeros((b_size, vall_size))
    layer['hid_v'] = np.zeros((b_size, h_size))
    layer['vis_spikes'] = np.zeros((b_size, vall_size),dtype='bool')
    layer['hid_spikes'] = np.zeros((b_size, h_size),dtype='bool')
    layer['vis_sum_spikes'] = np.zeros((b_size, vall_size))
    layer['hid_sum_spikes'] = np.zeros((b_size, h_size))
    layer['vis_refr'] = np.zeros((b_size, vall_size))
    layer['hid_refr'] = np.zeros((b_size, h_size))
    
    
    return layer
    
def spike_gens(image_list, rate=100., method='max'):
    p = np.zeros(image_list.shape)
    if method == 'sum':
        sum_value = image_list.sum(axis=1)
    elif method == 'max':
        sum_value = image_list.max(axis=1)
    
    for i in range(image_list.shape[0]):
        p[i,:] = image_list[i,:]/sum_value[i] * (rate/1000.)
        
    spikes = (np.random.uniform(0, 1, image_list.shape) <= p) * 1.
    return spikes
    
def update_up(layer, input_spikes, neuron_conf):
    # delete the spikes generated in the last ms
    layer['hid_spikes'][:,:] = 0
    # summed input
    sum_input = np.dot(input_spikes,layer['w_up'])
    # update membrane potential if neurons are not in refractory period
    index = layer['hid_refr']==0
    layer['hid_v'][index] += sum_input[index]
    
    # check threshold to generate spikes
    index = layer['hid_v']>=neuron_conf['v_thresh']
    layer['hid_spikes'][index] = 1
    #layer['hid_v'][layer['hid_v']<neuron_conf['v_thresh']] -= 0.0001*(neuron_conf['v_thresh'] - neuron_conf['v_reset'])
    # reset membrane potential to the resting voltage
    layer['hid_v'][index] = neuron_conf['v_reset']
    
    # membrane potential must be higher than v_reset
    layer['hid_v'][layer['hid_v']<0] = neuron_conf['v_reset']
    
    # set the refractory period to the spiking neurons
    layer['hid_refr'][index] = neuron_conf['tau_refr']
    layer['hid_refr'][layer['hid_refr']>0] -= 1.
    # update accumulated spikes
    layer['hid_sum_spikes'] += layer['hid_spikes'].astype(int)
    return layer
    
def teach_up(layer, input_spikes, neuron_conf):
    spikes = np.zeros(layer['hid_spikes'].shape).astype(int)
    # summed input
    sum_input = np.dot(input_spikes,layer['w_up'])
    # update membrane potential if neurons are not in refractory period
    index = layer['hid_refr']==0
    hid_v = np.copy(layer['hid_v'][index])
    hid_v += sum_input[index]
    
    # check threshold to generate spikes
    index = hid_v>=neuron_conf['v_thresh']
    spikes[index] = 1
    
    return spikes
    
def update_down(layer, input_spikes, neuron_conf):
    # delete the spikes generated in the last ms
    layer['vis_spikes'][:,:] = 0
    # summed input
    sum_input = np.dot(input_spikes,layer['w_down'])
    # update membrane potential if neurons are not in refractory period
    index = layer['vis_refr']==0
    layer['vis_v'][index] += sum_input[index]
    
    # check threshold to generate spikes
    index = layer['vis_v']>=neuron_conf['v_thresh']
    layer['vis_spikes'][index] = 1
    #layer['vis_v'][layer['vis_v']<neuron_conf['v_thresh']] -= 0.001*(neuron_conf['v_thresh'] - neuron_conf['v_reset'])
    # reset membrane potential to the resting voltage
    layer['vis_v'][index] = neuron_conf['v_reset']
    
    # membrane potential must be higher than v_reset
    layer['vis_v'][layer['vis_v']<0] = neuron_conf['v_reset']
    
    # set the refractory period to the spiking neurons
    layer['vis_refr'][index] = neuron_conf['tau_refr']
    layer['vis_refr'][layer['vis_refr']>0] -= 1.
    # update accumulated spikes
    layer['vis_sum_spikes'] += layer['vis_spikes'].astype(int)
    return layer

def update_top(layer, input_spikes, neuron_conf, batch_size):
    zero_spikes = np.zeros((batch_size, layer['vis_size'][1]))
    input_spikes = np.append(input_spikes,zero_spikes,axis=1)
    layer = update_up(layer, input_spikes, neuron_conf)
    input_spikes = layer['hid_spikes']
    layer = update_down(layer, input_spikes, neuron_conf)
    return layer

def greedy(layer, input_spikes, teach_spikes, neuron_conf, train_conf):
    # get h
    layer = update_up(layer, input_spikes, neuron_conf)
    layer['hid_last_spikes'].pop(-1)
    layer['hid_last_spikes'].insert(0, layer['hid_spikes'].astype('int')) # newly generated spikes, the time diff = 0
    
    # w+ when teaching_spikes (input_spikes) arrives
    for i in range(train_conf['tau_stdp']):
        w_plus = np.einsum('ij,ik->ijk', teach_spikes, train_conf['delta_w'][i]*layer['hid_last_spikes'][i]).mean(axis=0)
        layer['w_up'] += w_plus
    
    # w- when generating spikes
    layer = update_down(layer, layer['hid_spikes'], neuron_conf)
    for i in range(train_conf['tau_stdp']):
        w_minus = np.einsum('ij,ik->ijk', layer['vis_spikes'], train_conf['delta_w'][i]*layer['hid_last_spikes'][i]).mean(axis=0)
        layer['w_up'] -= w_minus
    
    layer['w_down'] = np.transpose(layer['w_up'])
    return layer
    
    
def init_weights(mode, size, conf):
    if mode == 'normal':
        w = np.random.normal(conf[0], conf[1], size) 
    elif mode == 'uniform':
        w = np.random.uniform(conf[0], conf[1], size)
    return w #w_down
    
def init_train_layer(layer_class, v_size, h_size, train_conf):
    b_size = train_conf['batch_size']
    if layer_class == 'l':
        vall_size = v_size
    elif layer_class == 'top':
        vall_size = sum(v_size)
    w = init_weights(train_conf['random'], vall_size*h_size, [train_conf['mu'], train_conf['sigma']]) 
    layer = init_layer(layer_class, v_size, h_size, b_size, w)
    layer['hid_last_spikes'] = list()
    for window in range(train_conf['tau_stdp']):
        layer['hid_last_spikes'].append(np.zeros((b_size, h_size), dtype='int'))        
    return layer
    
def init_train(train_conf, model='label'): #model: 'label' for classification or 'recon' for reconstruction 
    pop_size = train_conf['net_size']
    dbnet=[]
    np.random.seed(train_conf['seed'])
    if model == 'label':
        num_normal_layers = len(pop_size)-3
    else:
        num_normal_layers = len(pop_size)-1
    
    # build normal layers
    for l in range(num_normal_layers):
        vis_size = pop_size[l]     # num of visible units
        hid_size = pop_size[l+1]   # num of hidden units
        layer = init_train_layer('l', vis_size, hid_size, train_conf)     
        dbnet.append(layer)
    
    if model == 'label': # add the 'top' layer    
        vis_size = [pop_size[-3], pop_size[-1]]
        hid_size = pop_size[-2]
        layer = init_train_layer('top', vis_size, hid_size, train_conf)
        dbnet.append(layer)

    return dbnet
    
def init_test(test_conf, dbnet):
    dbn_test=[]
    np.random.seed(train_conf['seed'])
    for l in range(len(dbnet)):
        vis_size = dbnet[l]['vis_size']
        hid_size = dbnet[l]['hid_size']
        w_down = dbnet[l]['w_down']
        layer = init_layer(dbnet[l]['class'], vis_size, hid_size, test_conf['batch_size'], w_down)
        dbn_test.append(layer)
    return dbn_test
    
    
def train_layer(dbnet, curr_l, neuron_conf, train_conf, train_x, train_y=[], record=False):
    rate = train_conf['rate']
    method = train_conf['method']
    b_size = train_conf['batch_size']
    
    if record:
        w_list = list()
    
    for offset in range(b_size, train_x.shape[0]+1, b_size):
        image_list = train_x[offset-b_size:offset, :]
        for dt in range(train_conf['duration']): #train_conf['duration']
            input_spikes = spike_gens(image_list, rate=rate, method=method) #generate input spikes for 1ms
            teach_spikes = spike_gens(image_list, rate=rate, method=method) #generate teaching spikes for 1ms
            
            # real input_spikes for the training layer
            for l in range(curr_l):
                teach_spikes = teach_up(dbnet[l], teach_spikes, neuron_conf)
                dbnet[l] = update_up(dbnet[l], input_spikes, neuron_conf)
                input_spikes = dbnet[l]['hid_spikes']
                    
            layer = dbnet[curr_l]
            if layer['class'] == 'top':
                # add label values to the training data
                labels = np.zeros((b_size,layer['vis_size'][1]))
                labels[np.arange(b_size).astype(int),train_y[offset-b_size:offset].astype(int)]=1.
                label_spikes = spike_gens(labels, rate=rate, method=method)
                label_teach_spikes = spike_gens(labels, rate=rate, method=method)
                input_spikes = np.append(input_spikes,label_spikes,axis=1)
                teach_spikes = np.append(teach_spikes,label_teach_spikes,axis=1)
            layer = greedy(layer, input_spikes,  teach_spikes, neuron_conf, train_conf)
            if record:
                w_list.append(np.copy(layer['w_up']))
    
    if record:
        return dbnet, w_list
    else:
        return dbnet
        
def train_dbn_greedy(neuron_conf, train_conf, train_x, train_y=[], w_dir='temp/', start_e=0, start_o=0, curr_l=0):
    dbnet = init_train(train_conf, model='label')
    epoch = start_e
    offset = start_o
    np.random.seed(train_conf['seed'])
    train_num = train_conf['train_num']
    if start_e > 0 or start_o > 0:
        dbnet = np.load('%s%d_%d_%d.npy'%(w_dir,start_e,start_o,curr_l))
    
    while curr_l < len(dbnet):
        while epoch < train_conf['epoch']:
            while offset < train_x.shape[0]:    
                dbnet = train_layer(dbnet, curr_l, neuron_conf, train_conf, train_x[offset:offset+train_num], train_y[offset:offset+train_num])
                print 'epoch: %d, offset:%d, layer: %d'%(epoch, offset+train_num, curr_l)
                if np.mod(offset+train_num, 100)==0:
                    dbn_file = '%s%d_%d_%d.npy'%(w_dir, epoch, offset+train_num, curr_l)
                    np.save(dbn_file, dbnet)
                offset += train_num
            offset = 0
            epoch+= 1
        epoch = 0
        curr_l += 1 
    return dbnet       
