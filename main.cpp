#include <iostream>
#include "TrainingData.h"
#include "Neuron.h"
#include "NeuralNetwork.h"

#define DATA_LOCATION "TrainingData.txt"

void showVectorVals(string label, vector<double> &v); 

int main() {

    TrainingData data(DATA_LOCATION);
    
    vector<unsigned> topology;
    data.getTopology(topology);

    NeuralNetwork nn(topology);

    vector<double> input, target, result;
    int trainingPass = 0;

    while (!data.isEndOfFile()) {

        ++trainingPass;
        if (data.getNextInputs(input) != topology[0]) {
            break;
        }

        cout << endl << "Sample " << trainingPass << endl;

        showVectorVals("Input:", input);
        nn.fowardPropagation(input);

        nn.getFinalResults(result);
        showVectorVals("Output:", result);

        data.getTargetOutputs(target);
        showVectorVals("Target:", target);
        assert(target.size() == topology.back());

        nn.backwardPropagation(target);
    }

    cout << endl << "Completed" << endl;
    return 0;
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}