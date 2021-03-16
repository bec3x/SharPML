#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

enum class activation_t
{
	Sigmoid,
	Tanh,
	Relu,
	LRelu
};

static float net_tanh(float x) { return tanh(x); }
static float tanh_dev(float x) { return 1 - (x * x); }

static float sig(float x) { return 1 / (1 + (exp(-x))); }
static float sigmoid_dev(float x) { return (exp(-x)) / pow(1 + exp(-x), 2); }

static float relu(float x) { return x > 0 ? x : 0; }
static float relu_dev(float x) { return x > 0 ? 1.0f : 0.0f; }

static float lRelu(float x) { return x > 0 ? x : 0.01f * x; }
static float lRelu_dev(float x) { return x > 0 ? 1.0f : 0.01f; }

#endif // !ACTIVATION_H
