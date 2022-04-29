#include "Neuron.h"

double Neuron::learningRate = 0.15;
double Neuron::momentum = 0.5;

void Neuron::updateWeights(Layer &prevLayer) {
    for(unsigned n=0; n<prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[id].deltaWeight;
        double newDeltaWeight = (learningRate * neuron.setOutput() * gradient)
                                + (momentum * oldDeltaWeight);
        neuron.outputWeights[id].deltaWeight = newDeltaWeight;
        neuron.outputWeights[id].weight += newDeltaWeight;
    }
}

double Neuron::sumOfDerivativesOfWeights(const Layer &nextLayer) const {
    double sum = 0.0;
    for(unsigned n=0; n<nextLayer.size()-1; ++n){
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }
    return sum;
}
void Neuron::calcHiddenLayerGradients(const Layer &nextLayer){
    double dow = sumOfDerivativesOfWeights(nextLayer);
    gradient = dow * Neuron::activationFunctionDerivative(output);
}

void Neuron::calcOutputLayerGradients(double targetVal){
    double delta = targetVal - output;
    gradient = delta * Neuron::activationFunctionDerivative(output);
}

double Neuron::activationFunction(double x) {
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    return 1.0-x*x;
}

void Neuron::fowardPropagation(const Layer &prevLayer){
    double sum = 0.0;
    for(unsigned n = 0; n<prevLayer.size(); ++n){
        sum += prevLayer[n].setOutput() * 
                prevLayer[n].outputWeights[id].weight;
    }

    output = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned outputNums, unsigned myIndex) {
    for(unsigned c = 0; c < outputNums; ++c){
        outputWeights.push_back(Edge());
        outputWeights.back().weight = getRandomWeight();
    }

    id = myIndex;
}