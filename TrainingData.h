
#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData {

public:

    TrainingData(const string filename);

    bool isEndOfFile(void) { return trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:

    ifstream trainingDataFile;
};

#endif