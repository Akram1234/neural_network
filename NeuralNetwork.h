
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <cassert>
#include "Neuron.h"

using namespace std;

class NeuralNetwork { 

public:

    NeuralNetwork(const vector<unsigned> &topology);

    void fowardPropagation(const vector<double> &inputVals);
    void backwardPropagation(const vector<double> &targetVals);
    void getFinalResults(vector<double> &resultVals) const;

private:

    vector<Layer> network;
    double networkError;

};

#endif