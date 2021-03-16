#include "Neuron.h"
#include <functional>

Neuron::Neuron(unsigned int nr_outputs, unsigned int index)
{
	_index = index;

	for (int i = 0; i < nr_outputs; i++) {
		_output_weights.push_back(Edge());
		_output_weights.back().weight = random_weight();
	}
}

void Neuron::feed_forward(const Layer& prevLayer, std::function<float(float)> activation)
{
	float sum = 0.0f;

	for (unsigned int i = 0; i < prevLayer.size(); i++) {
		sum += prevLayer[i]._output_val * prevLayer[i]._output_weights[_index].weight;
	}

	_output_val = activation(sum);
}

void Neuron::calculate_output_gradient(float target, std::function<float(float)> derivitive)
{
	float delta = target - _output_val;
	_gradient = delta * derivitive(_output_val);
}

void Neuron::calculate_hidden_gradient(const Layer& nextLayer, std::function<float(float)> derivitive)
{
	float dow = sum_gradients_by_weights(nextLayer);
	_gradient = dow * derivitive(_output_val);
}

void Neuron::update_input_weights(Layer& prevLayer)
{
	for (unsigned int i = 0; i < prevLayer.size(); i++) {
		Neuron& neuron = prevLayer[i];

		float oldDelta = neuron._output_weights[_index].delta_wt;
		float newDelta = (_eta * neuron.get_output_val() * _gradient) + (_alpha * oldDelta);

		neuron._output_weights[_index].delta_wt = newDelta;
		neuron._output_weights[_index].weight += newDelta;
	}
}

float Neuron::sum_gradients_by_weights(const Layer& nextLayer) const
{
	float sum = 0.0;

	for (unsigned int i = 0; i < nextLayer.size() - 1; i++) {
		sum += _output_weights[i].weight * nextLayer[i]._gradient;
	}

	return sum;
}