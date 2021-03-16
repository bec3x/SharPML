#include "SharPNet.h"
#include <string>
#include <cassert>
#include <functional>
#include <algorithm>

typedef std::vector<int>::iterator iterator;

SharPNet::SharPNet(std::vector<int>& topology, activation_t activation = activation_t::Relu)
{
	// Selected the activation function for the neural net
	switch (activation) {
	case activation_t::Sigmoid:
		_activation_function = sig;
		_activation_derviative = sigmoid_dev;
		break;
	case activation_t::Tanh:
		_activation_function = net_tanh;
		_activation_derviative = tanh_dev;
		break;
	case activation_t::Relu:
		_activation_function = relu;
		_activation_derviative = relu_dev;
		break;
	case activation_t::LRelu:
		_activation_function = lRelu;
		_activation_derviative = lRelu_dev;
		break;
	}

	// creating the layers specified by the user
	for (unsigned int i = 0; i < topology.size(); i++) {
		unsigned int nr_outputs = i == topology.size() - 1 ? 0 : topology[i + 1];

		_layers.push_back(Layer());
		
		for (int j = 0; j <= topology[i]; j++) {
			_layers.back().push_back(Neuron(nr_outputs, j));
		}

		_layers.back().back().set_output_val(1.0); // Bias node
	}

	_training_accuracy = 0.0;
	_accuracy = 0.0;
	_smoothing_factor = 0.0;
}

double SharPNet::train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int nr_epochs)
{
	assert(inputs.size() == outputs.size());
	_smoothing_factor = inputs.size() * .05f;

	// create a vector of indicies that will allow us to "shuffle" the two vectors passed
	// to the train function, while keeping the same mapping to inputs and outputs and
	// it will ensure the integrity of the two vectors
	std::vector<int> indicies;
	for (unsigned int i = 0; i < inputs.size(); i++) {
		indicies.push_back(i);
	}

	for (int pass = 0; pass < nr_epochs; pass++) {
		std::random_shuffle(indicies.begin(), indicies.end());

		for (iterator it = indicies.begin(); it != indicies.end(); it++) {
			feed_forward(inputs[*it]);
			back_propagation(outputs[*it]);
		}
	}

	_training_accuracy = (1 - _training_accuracy) * 100;
	return _training_accuracy;
}

void SharPNet::feed_forward(std::vector<float>& inputs)
{
	assert(inputs.size() == _layers[0].size() - 1);

	for (unsigned int i = 0; i < inputs.size(); i++) {
		_layers[0][i].set_output_val(inputs[i]);
	}

	for (unsigned int i = 1; i < _layers.size(); i++) {
		Layer& prevLayer = _layers[i - 1];
		Layer& currLayer = _layers[i];

		for (unsigned int j = 0; j < _layers[i].size() - 1; j++) {
			_layers[i][j].feed_forward(prevLayer, _activation_function);
		}
	}
}

void SharPNet::back_propagation(std::vector<float>& outputs)
{
	// Calculate overall network error with root mean squared error
	Layer& outputLayer = _layers.back();
	float error = 0.0;

	for (unsigned int i = 0; i < outputLayer.size() - 1; i++) {
		float delta = outputs[i] - outputLayer[i].get_output_val();
		error += delta * delta;
	}

	error /= outputLayer.size() - 1;
	error = sqrt(error);

	_training_accuracy = (float)((_training_accuracy * _smoothing_factor) + error) / (_smoothing_factor + 1.0);

	for (unsigned int i = 0; i < outputLayer.size() - 1; i++) {
		outputLayer[i].calculate_output_gradient(outputs[i], _activation_derviative);
	}

	for (unsigned int i = _layers.size() - 2; i > 0; i--) {
		Layer& hiddenLayer = _layers[i];
		Layer& nextLayer = _layers[i + 1];

		for (unsigned int j = 0; j < hiddenLayer.size(); j++) {
			hiddenLayer[j].calculate_hidden_gradient(nextLayer, _activation_derviative);
		}
	}

	for (unsigned int i = _layers.size() - 1; i > 0; i--) {
		Layer& layer = _layers[i];
		Layer& prevLayer = _layers[i - 1];

		for (unsigned int j = 0; j < layer.size() - 1; j++) {
			layer[j].update_input_weights(prevLayer);
		}
	}
}

double SharPNet::evaluate(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
{
	assert(inputs.size() == outputs.size());
	double model_accuracy = 0.0;
	double smoothing_factor = inputs.size() * .05;
	double sum = 0.0;
	std::vector<double> accuracy_vector;

	for (unsigned int i = 0; i < inputs.size(); i++) {
		feed_forward(inputs[i]);

		Layer& outputLayer = _layers.back();
		float error = 0.0f;

		for (unsigned int j = 0; j < outputLayer.size() - 1; j++) {
			float delta = (float)((double)outputs[i][j] - outputLayer[j].get_output_val());
			error = delta * delta;
		}

		error /= outputLayer.size() - 1;
		error = sqrt(error);

		model_accuracy = ((model_accuracy * smoothing_factor) + error) / (smoothing_factor + 1.0);
		accuracy_vector.push_back(model_accuracy);
	}

	for (auto ele : accuracy_vector) {
		sum += ele;
	}

	_accuracy = sum / accuracy_vector.size();
	_accuracy = (1 - _accuracy) * 100;
	return _accuracy;
}

void SharPNet::get_results(std::vector<float>& inputs, std::vector<float>& results)
{
	results.clear();
	feed_forward(inputs);

	for (unsigned int i = 0; i < _layers.back().size() - 1; i++) {
		results.push_back(_layers.back()[i].get_output_val());
	}
}