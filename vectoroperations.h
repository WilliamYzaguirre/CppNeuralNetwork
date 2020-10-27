#ifndef VECTOROPERATIONS_H
#define VECTOROPERATIONS_H

#include <vector>
#include <iostream>
#include <math.h>

double vectorDotProduct(std::vector<double> v1, std::vector<double> v2);

std::vector<double> vectorMatrixMult(std::vector<std::vector<double>> m, std::vector<double> v);

std::vector<double> vectorAdd(std::vector<double> v1, std::vector<double> v2);

std::vector<double> vectorSubtract(std::vector<double> v1, std::vector<double> v2);

std::vector<double> hadamardVector(std::vector<double> v1, std::vector<double> v2);

double vectorSum(std::vector<double> v1);

std::vector<std::vector<double>> vectorTransposeMult(std::vector<double> v1, std::vector<double> v2);

std::vector<std::vector<double>> matrixTranspose(std::vector<std::vector<double>> m);

void normalizeVector(std::vector<double>& v);

std::vector<double> averageVectors(std::vector<std::vector<double>> vectors);

#endif // VECTOROPERATIONS_H
