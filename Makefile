
output: main.o TrainingData.o Neuron.o NeuralNetwork.o
	g++ main.o TrainingData.o Neuron.o NeuralNetwork.o -o output

main.o:	 main.cpp
	g++ -c main.cpp 

TrainingData.o: TrainingData.cpp
	g++ -c TrainingData.cpp 

Neuron.o: Neuron.cpp
	g++ -c Neuron.cpp

NeuralNetwork.o:	NeuralNetwork.cpp
	g++ -c NeuralNetwork.cpp

clean:
	rm *.o output
	
