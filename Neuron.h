#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cmath>

using namespace std;

struct Edge {
    double weight;
    double deltaWeight;
};

class Neuron;
typedef vector<Neuron> Layer;

class Neuron {

public:

    Neuron(unsigned outputNums, unsigned myIndex);
    
    void fowardPropagation(const Layer &prevLayer);
    void calcOutputLayerGradients(double targetVal);
    void calcHiddenLayerGradients(const Layer &nextLayer);
    void updateWeights(Layer &prevLayer);

    void setOutput(double val) { output = val; }
    double setOutput(void) const { return output; }

private:

    static double learningRate;
    static double momentum;
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double getRandomWeight(void) { return rand()/double(RAND_MAX); }
    
    double sumOfDerivativesOfWeights(const Layer &nextLayer) const;

    vector<Edge> outputWeights;
    double output;
    unsigned id;
    double gradient;
};



#endif