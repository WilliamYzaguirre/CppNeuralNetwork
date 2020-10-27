#include "neuralnetwork.h"


/*
 * Constructor builds the layers and neurons, and assigns random weights (# of inputs for first layer, and # of neurons for subsequent),
 * and biases
 * LayerCount is number os interior layers. Input layer and target layer do not count towards that number
*/
NeuralNetwork::NeuralNetwork(int hiddenLayerCount, int neuronCount, int targetCount, int inputCount)
    :hiddenLayerCount{hiddenLayerCount}, neuronCount{neuronCount}, targetCount{targetCount}, inputCount{inputCount}
{
    totalLayerCount = hiddenLayerCount + 1;
    // Initialize bias. This is a vector of layerCount + 1 vectors
    // We use layerCount + 1 because we need to make the target layer as well
    // I don't need to make an input layer because it has no weights or bias. It's just data
    for (int i = 0; i < totalLayerCount; ++i)
    {
        // Start with target layer since it's special. We want to catch it before the others.
        if (i == totalLayerCount - 1)
        {
            // Target layer gets targetCount biases
            //biases.push_back(getRandomDoubleVector(targetCount, 5));
            biases.push_back(getDoubleVector(targetCount, 0));

        }
        // Otherwise, we make a hidden layer
        else
        {
            // Hidden layers have neuronCount biases
            //biases.push_back(getRandomDoubleVector(neuronCount, 5));
            biases.push_back(getDoubleVector(neuronCount, 1));

        }
    }

    // Initialize the weights. This is a vector of layer + 1 vectors of neurons
    for (int i = 0; i < totalLayerCount; ++i)
    {
        // Start with first layer since it has unique weight counts (inputCount)
        if (i == 0)
        {
            // First, make the layer vector
            std::vector<std::vector<double>> layer;

            // We need to give the target layer targetCount vectors of neuronCount size
            for (int j = 0; j < neuronCount; ++j)
            {
                //layer.push_back(getRandomDoubleVector(inputCount, 5));
                layer.push_back(getDoubleVector(inputCount, 0));

            }

            weights.push_back(layer);
        }

        //  Then target layer
        else if (i == totalLayerCount - 1)
        {
            // First, make the layer vector
            std::vector<std::vector<double>> layer;

            // We need to give the target layer targetCount vectors of neuronCount size
            for (int j = 0; j < targetCount; ++j)
            {
                //layer.push_back(getRandomDoubleVector(neuronCount, 5));
                layer.push_back(getDoubleVector(neuronCount, 1));

            }

            weights.push_back(layer);
        }
        // Otherwise, we give the layer neuronCount vectors of neuronCount size
        else
        {
            // Make the layer
            std::vector<std::vector<double>> layer;

            for (int j = 0; j < neuronCount; ++j)
            {
                //layer.push_back(getRandomDoubleVector(neuronCount, 5));
                layer.push_back(getRandomDoubleVector(neuronCount, 0));

            }

            weights.push_back(layer);
        }
    }

    // Make sure everything got created properly
    std::cout << "Bias layer count: " << biases.size() << std::endl;
    std::cout << "Bias layer Sizes: ";
    for (auto layer : biases)
    {
        std::cout << layer.size() << " ";
    }
    std::cout << "\nWeight layer count: " << weights.size() << std::endl;
    std::cout << "Weight layer sizes: ";
    for (auto layer : weights)
    {
        std::cout << layer.size() << " ";
    }
    std::cout << "\nWeight layer weights count: ";
    for (auto layer : weights)
    {
        for (auto weights : layer)
        {
            std::cout << weights.size() << " ";
        }

        std::cout << " | ";
    }

}

NeuralNetwork::~NeuralNetwork()
{

}

std::vector<double> NeuralNetwork::getRandomDoubleVector(int count, double high)
{
    std::srand((unsigned) time(0));
    std::vector<double> ret;
    for (int i = 0; i < count; ++i)
    {
        double random = (((double) rand() / RAND_MAX)*(high*2) - high);
        ret.push_back(random);
    }
    return ret;
}


void NeuralNetwork::train(std::vector<std::vector<double>> trainInput, std::vector<double> trainLabel, int eta, int batchSize, int epochNumber)
{
    for (int epoch = 0; epoch < epochNumber; ++epoch)
    {
        double totalPoints = trainInput.size();
        int batchCount = ceil(totalPoints / batchSize);

        for (int iteration = 0; iteration < batchCount; ++iteration)
        {
            //fix later
            int currentBatchCount = batchSize;

            std::vector<std::vector<std::vector<double>>> totalWeightGradient;
            for (auto layer : weights)
            {
                std::vector<std::vector<double>> row;
                for (auto neuron : layer)
                {
                    row.push_back(getDoubleVector(neuron.size(), 0));
                }
                totalWeightGradient.push_back(row);
            }

            std::vector<std::vector<double>> totalBiasGradient;
            for (auto layer : biases)
            {
                totalBiasGradient.push_back(getDoubleVector(layer.size(), 0));
            }

            for (int batch = iteration * batchSize; batch < batchSize && batch < totalPoints; ++batch)
            {

                std::vector<std::vector<double>> costs;

                std::vector<double> activation = trainInput[batch];

                std::vector<std::vector<double>> zs; // 2D matrix to store all z's. z = weight . activation + b
                std::vector<std::vector<double>> activations; // 2D matrix to store all activations. activation = acticationFunction(z)

                std::vector<std::vector<std::vector<double>>> gradientw;
                std::vector<std::vector<double>> gradientb;

                activations.push_back(activation);

                // Forward pass
                for (int i = 0; i < totalLayerCount; ++i)
                {
                    std::vector<double> z = vectorAdd(vectorMatrixMult(weights[i], activation), biases[i]);
                    zs.push_back(z);
                    std::vector<double> newActivation = sigmoid(z);
                    activations.push_back(newActivation);
                    activation.clear();
                    activation = newActivation;
                }

                // Back pass
                // Using cost function |a(L) - y|^2, so cost derive is 2(a(L) - y)
                // Get correct vector

                std::vector<double> y;
                for (int i = 0; i < targetCount; ++i)
                {
                    if (i == trainLabel[batch])
                    {
                        y.push_back(1);
                    }
                    else
                    {
                        y.push_back(0);
                    }
                }

                // delta = costDerivative(a(L) - y)*activationDerivative(z(L))
                // dC/db = costDerivative(a(L) - y)*activationDerivative(z(L))
                // dC/dw = a(L-1)*costDerivative(a(L) - y)*activationDerivative(z(L))

                std::vector<double> costVector = costFunction(activations[activations.size() - 1], y);
                costs.push_back(costVector);

                std::vector<double> costDerivativeDebug = costDerivative(activations[activations.size() - 1], y);
                std::vector<double> sigmoidPrimeDebug = sigmoidPrime(zs[zs.size() - 1]);

                std::vector<double> delta = hadamardVector(costDerivative(activations[activations.size() - 1], y), sigmoidPrime(zs[zs.size() - 1]));
                gradientb.push_back(delta);

                std::vector<std::vector<double>> gradientWDebug = vectorTransposeMult(delta, activations[activations.size() - 2]);
                gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - 2]));

                for (int i = 2; i < totalLayerCount + 1; ++i)
                {
                    std::vector<double> z = zs[zs.size() - i];
                    std::vector<double> sp = sigmoidPrime(z);
                    std::vector<std::vector<double>> weightsDebug = weights[weights.size() - i + 1];
                    std::vector<double> vectorMatrixMultDebug1 = vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta);
                    delta = hadamardVector(vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta), sp);
                    std::vector<std::vector<double>> matrixTransposeDebug = matrixTranspose(weights[weights.size() - i + 1]);
                    std::vector<double> vectorMatrixMultDebug2 = vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta);
                    gradientb.push_back(delta);
                    gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - i - 1]));
                }

                std::reverse(gradientb.begin(), gradientb.end());
                std::reverse(gradientw.begin(), gradientw.end());

                for (int i = 0; i < int(totalBiasGradient.size()); ++i)
                {
                    for (int j = 0; j < int(totalBiasGradient[i].size()); ++j)
                    {
                        totalBiasGradient[i][j] += gradientb[i][j];
                    }
                }

                for (int i = 0; i < int(totalWeightGradient.size()); ++i)
                {
                    for (int j = 0; j < int(totalWeightGradient[i].size()); ++j)
                    {
                        for (int h = 0; h < int(totalWeightGradient[i][j].size()); ++h)
                        {
                            totalWeightGradient[i][j][h] += gradientw[i][j][h];
                        }
                    }
                }

                double averageCost = vectorSum(averageVectors(costs));

                std::cout << "Batch " << iteration << " Complete. Average cost: " << averageCost << std::endl;
            }

            // Normalize the total gradient vectors

            for (auto b : totalBiasGradient)
            {
                normalizeVector(b);
            }

            for (auto layer : totalWeightGradient)
            {
                for (auto w : layer)
                {
                    normalizeVector(w);
                }
            }

            for (int i = 0; i < int(biases.size()); ++i)
            {
                for (int j = 0; j < int(biases[i].size()); ++j)
                {
                    biases.at(i).at(j) -= (eta/currentBatchCount)*totalBiasGradient[i][j];
                }
            }

            for (int i = 0; i < int(weights.size()); ++i)
            {
                for (int j = 0; j < int(weights[i].size()); ++j)
                {
                    for (int h = 0; h < int(weights[i][j].size()); ++h)
                    {
                        weights[i][j][h] = weights[i][j][h] - (eta/currentBatchCount)*totalWeightGradient[i][j][h];
                    }
                }
            }
        }
        std::cout << "Epoch " << epoch << " Complete." << std::endl;
    }

}

void NeuralNetwork::test(std::vector<std::vector<double> > testInput, std::vector<double> testLabel)
{
    int correct = 0;
    int incorrect = 0;
    for (int input = 0; input < int(testInput.size()); ++input)
    {
        std::vector<double> activation = testInput[input];
        for (int i = 0; i < totalLayerCount; ++i)
        {
            std::vector<double> z = vectorAdd(vectorMatrixMult(weights[i], activation), biases[i]);
            activation = sigmoid(z);
        }
        int index = 0;
        double max = activation[0];
        for (int i = 1; i < int(activation.size()); ++i)
        {
            if (activation[i] > max)
            {
                index = i;
                max = activation[i];
            }
        }
        if (index == testLabel[input])
        {
            correct++;
        }
        else
        {
            incorrect++;
        }
    }
    std::cout << "Correct: " << correct << ". Incorrect: " << incorrect << std::endl;
}

std::vector<double> NeuralNetwork::sigmoid(std::vector<double> v)
{
    std::vector<double> ret;
    for (auto item : v)
    {
         ret.push_back(1 / (1 + std::exp(-item)));
    }
    return ret;
}

std::vector<double> NeuralNetwork::sigmoidPrime(std::vector<double> v)
{
    std::vector<double> ret = hadamardVector(sigmoid(v), vectorSubtract(getDoubleVector(v.size(), 1), sigmoid(v))); //sig(v) * (1 - sig(v)
    return ret;
}

std::vector<double> NeuralNetwork::getDoubleVector(int count, double value)
{
    std::vector<double> ret;
    for (int i = 0; i < count; ++i)
    {
         ret.push_back(value);
    }
    return ret;
}

std::vector<double> NeuralNetwork::costFunction(std::vector<double> output, std::vector<double> y)
{
    std::vector<double> diff = vectorSubtract(output, y);
    return (hadamardVector(diff, diff));
}

std::vector<double> NeuralNetwork::costDerivative(std::vector<double> output, std::vector<double> y)
{
    std::vector<double> diff = vectorSubtract(output, y);
    return (hadamardVector(diff, getDoubleVector(diff.size(), 2)));
}




