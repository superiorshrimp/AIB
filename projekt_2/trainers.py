import numpy as np


def calc_wg_of_neuron(neuron_idx, input_models):
    wg = np.zeros(len(input_models[0]))
    
    for j in range(len(input_models[0])):
        if neuron_idx == j: continue
        sum = 0.0
        for m in range(len(input_models)):
            sum += input_models[m][neuron_idx] * input_models[m][j]
        
        wg[j] = (1.0 / float(len(input_models))) * sum

    return wg

def train(network, input_models):
    weights = np.zeros((len(input_models[0]), len(input_models[0])))

    for i in range(len(input_models[0])):
        weights[i] = calc_wg_of_neuron(i, input_models)

    network.set_weights(weights)
