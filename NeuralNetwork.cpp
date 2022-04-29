
#include "NeuralNetwork.h"

void NeuralNetwork::getFinalResults(vector<double> &resultVals) const {
    resultVals.clear();
    for(unsigned n=0; n<network.back().size()-1; ++n){
        resultVals.push_back(network.back()[n].setOutput());
    }
}

void NeuralNetwork::backwardPropagation(const vector<double> &targetVals){
    //Error calculation - RMS
    Layer &outputLayer = network.back();
    networkError = 0.0;
    for(unsigned n=0; n<outputLayer.size()-1; ++n){
        double delta = targetVals[n] - outputLayer[n].setOutput();
        networkError += delta * delta;
    }
    networkError /= outputLayer.size()-1;
    networkError = sqrt(networkError);

    //Calculate Output Gradients
    for(unsigned n=0; n<outputLayer.size()-1; ++n) {
        outputLayer[n].calcOutputLayerGradients(targetVals[n]);
    }

    // Calculate Gradients for all hidden layers
    for(unsigned layerNum = network.size()-2; layerNum>0; --layerNum){
        Layer &hiddenLayer = network[layerNum];
        Layer &nextLayer = network[layerNum+1];
        for(unsigned n = 0; n<hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenLayerGradients(nextLayer);
        }
    }

    //update weights
    for(unsigned layerNum = network.size()-1; layerNum>0; --layerNum) {
        Layer &layer = network[layerNum];
        Layer &prevLayer = network[layerNum-1];
        for(unsigned n=0; n<layer.size()-1; ++n){
            layer[n].updateWeights(prevLayer);
        }

    }

}

void NeuralNetwork::fowardPropagation(const vector<double> &inputVals) {
    assert(inputVals.size() == network[0].size() - 1);
    for(unsigned i=0; i<inputVals.size(); i++){
        network[0][i].setOutput(inputVals[i]);
    }

    for(unsigned layerNum = 1; layerNum<network.size(); ++layerNum){
        Layer &prevLayer = network[layerNum-1];
        for(unsigned n = 0; n<network[layerNum].size()-1; ++n){
            network[layerNum][n].fowardPropagation(prevLayer);
        }
    }

}

NeuralNetwork::NeuralNetwork(const vector<unsigned> &topology) {
    unsigned numLayers = topology.size();
    for(unsigned layer = 0; layer < numLayers; ++layer){
        network.push_back(Layer());
        unsigned outputNums = layer == topology.size()-1 ? 0 : topology[layer+1];
        for(unsigned neuronNum = 0; neuronNum <= topology[layer]; ++neuronNum){
            network.back().push_back(Neuron(outputNums, neuronNum));
        }
        network.back().back().setOutput(1.0);
    }
}
