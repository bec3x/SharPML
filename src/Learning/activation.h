#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <vector>
#include "tensor.h"

enum class activation_t
{
	Sigmoid,
	Tanh,
	Relu,
	LRelu,
	Softmax
};

static tensor<float> net_tanh(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = tanh(x[i]);
	}

	return a;
}

static tensor<float> tanh_dev(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = (1 - (x[i] * x[i]));
	}

	return a;
}

static tensor<float> sig(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = (1 / (1 + (exp(-x[i]))));
	}

	return a;
}
static tensor<float> sigmoid_dev(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = ((exp(-x[i])) / pow(1 + exp(-x[i]), 2));
	}

	return a;
}

static tensor<float> relu(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = x[i] > 0 ? x[i] : 0;
	}

	return a;
}

static tensor<float> relu_dev(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = x[i] > 0 ? 1.0f : 0.0f;
	}

	return a;
}

static tensor<float> lRelu(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = x[i] > 0 ? x[i] : 0.01f * x[i];
	}

	return a;
}
static tensor<float> lRelu_dev(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	for (int i = 0; i < x.size(); i++) {
		a(i, 0, 0) = x[i] > 0 ? 1.0f : 0.01f;
	}

	return a;
}

static tensor<float> softmax(std::vector<float> x)
{
	tensor<float> a(x.size(), 1, 1);

	float exp_sum = 0.0f;
	float output_max = -FLT_MAX;

	std::vector<float> shift_output = x;
	for (int i = 0; i < shift_output.size(); i++) {
		if (output_max < shift_output[i]) output_max = shift_output[i];
	}

	for (int i = 0; i < shift_output.size(); i++) {
		shift_output[i] = exp((shift_output[i] - output_max));
		exp_sum += shift_output[i];
	}

	for (int n = 0; n < x.size(); n++) {
		a(n, 0, 0) = shift_output[n] / exp_sum;
	}

	return a;
}

static tensor<float> softmax_dev(std::vector<float> x)
{
	tensor<float> s = softmax(x);

	tensor<float> a(x.size(), x.size(), 1);

	for (int i = 0; i < x.size(); i++) {
		for (int j = 0; j < x.size(); j++) {
			a(i, j, 0) = s(i, 0, 0) * ((i == j) - s(j, 0, 0));
		}
	}

	return a;
}

#endif // !ACTIVATION_H
