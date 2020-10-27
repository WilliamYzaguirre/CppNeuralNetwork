#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


#include <vector>
#include "vectoroperations.h"
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>


class NeuralNetwork
{
public:
    NeuralNetwork(int layerCount, int neuronCount, int targetCount, int inputCount);

    ~NeuralNetwork();

    std::vector<double> getRandomDoubleVector(int count, double high);

    void train(std::vector<std::vector<double>> trainInput, std::vector<double> trainLabel, int eta, int batchSize=32, int epochNumber=32);

    void test(std::vector<std::vector<double>> testInput, std::vector<double> testLabel);

    //void printNetwork();

    //std::vector<std::vector<double>> getRandomWeightMatrix(int inputs, int neurons, double low, double high);

    std::vector<double> sigmoid(std::vector<double> v);

    std::vector<double> sigmoidPrime(std::vector<double> v);

    std::vector<double> getDoubleVector(int count, double value);

    std::vector<double> costFunction(std::vector<double> output, std::vector<double> y);

    std::vector<double> costDerivative(std::vector<double> output, std::vector<double> y);

    //std::vector<double> forwardPass(Layer layer, std::vector<double> input);

private:
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;

    //std::vector<NeuralLayer*> layers;
    int hiddenLayerCount;
    int neuronCount;
    int targetCount;
    int inputCount;
    int batchSize;
    int epochNumber;

    //std::uniform_real_distribution<> dis{-1, 1};

    //std::default_random_engine engine;

    int totalLayerCount;
};

#endif // NEURALNETWORK_H
