#include "SharPNetConv.h"
#include <fstream>
#include <iostream>

SharPNetConv::SharPNetConv(std::vector<layer*> topology, loss_t loss_function, float learning_rate)
{
	_layers = std::move(topology);
	_loss_function = loss_function;
	_learning_rate = learning_rate;

	_training_accuracy = 0.0f;
	_accuracy = 0.0f;
	_smoothing_factor = 0.0f;
}

std::vector<std::pair<float, float>> SharPNetConv::train(std::vector<image_sample> samples, int nr_epochs)
{
	_smoothing_factor = samples.size() * .05f;

	for (int pass = 0; pass < nr_epochs; pass++) {
		std::vector<tensor<float>> predictions;
		std::vector<tensor<float>> actual;

		for (unsigned int i = 0; i < samples.size(); i++) {
			tensor<float> data = convert_to_tensor(samples[i].data);
			tensor<float> expected = convert_to_tensor(samples[i].expected);

			feed_forword(data);
			back_propagation(expected);	

			predictions.emplace_back(_layers.back()->get_output());
			actual.emplace_back(expected);
		}

		float loss = calculate_loss(predictions, actual);
		_training_accuracy = (1 - _training_accuracy) * 100;

		_history.emplace_back(std::make_pair(loss, _training_accuracy));
	}

	return _history;
}

void SharPNetConv::feed_forword(tensor<float>& input)
{
	for (unsigned int layer = 0; layer < _layers.size(); layer++) {
		if (layer == 0) {
			_layers[layer]->activate(input);
		}
		else {
			tensor<float> last_output = _layers[layer - 1]->get_output();
			_layers[layer]->activate(last_output);
		}
	}
}

void SharPNetConv::back_propagation(tensor<float>& expected)
{
	int network_output_size = _layers.back()->get_output()._size._x *
		_layers.back()->get_output()._size._y *
		_layers.back()->get_output()._size._z;

	int expected_size = expected._size._x * expected._size._y * expected._size._z;

	assert(network_output_size == expected_size);

	tensor<float> output_gradients = _layers.back()->get_output() - expected;

	for (int layer = _layers.size() - 1; layer >= 0; layer--) {
		if (layer == _layers.size() - 1) {
			_layers[layer]->calc_grads(output_gradients);
		}
		else {
			tensor<float> last_gradients = _layers[layer + 1]->get_gradients();
			_layers[layer]->calc_grads(last_gradients);
		}
	}

	for (unsigned int layer = 0; layer < _layers.size(); layer++) {
		_layers[layer]->fix_weights(_learning_rate);
	}

	float error = 0.0;
	for (int j = 0; j < network_output_size; j++) {
		float delta = _layers.back()->get_output()._data[j] - expected._data[j];
		error += delta * delta;
	}

	error /= network_output_size;
	error = sqrt(error);

	_training_accuracy = ((_training_accuracy * _smoothing_factor + error) / (_smoothing_factor + 1.0));
}

float SharPNetConv::evaluate(std::vector<image_sample> samples)
{
	float model_accuracy = 0.0;
	float smoothing_factor = samples.size() * .05f;
	float sum = 0.0;
	std::vector<float> accuracy_vector;

	for (unsigned int i = 0; i < samples.size(); i++) {
		tensor<float> input = convert_to_tensor(samples[i].data);
		tensor<float> expected = convert_to_tensor(samples[i].expected);

		int network_output_size = _layers.back()->get_output()._size._x *
			_layers.back()->get_output()._size._y *
			_layers.back()->get_output()._size._z;

		int expected_size = expected._size._x * expected._size._y * expected._size._z;

		feed_forword(input);

		assert(network_output_size == expected_size);

		float error = 0.0;
		for (int j = 0; j < network_output_size; j++) {
			float delta = _layers.back()->get_output()._data[j] - expected._data[j];
			error = delta * delta;
		}

		error /= network_output_size;
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

float SharPNetConv::calculate_loss(std::vector<tensor<float>> predictions, std::vector<tensor<float>> actual)
{
	float loss = 0.0f;
	
	if (_loss_function == loss_t::MeanSquaredError) {
		loss = MSE(predictions, actual);
	}
	else if (_loss_function == loss_t::BinaryCrossentropy) {
		loss = BinaryCrossentropy(predictions, actual);
	}
	else if (_loss_function == loss_t::CategoricalCrossentropy) {
		loss = CategoricalCrossentropy(predictions, actual);
	}

	return loss;
}

bool SharPNetConv::save(std::string filepath)
{
	std::ofstream outfile(filepath);

	if (!outfile.is_open()) { return false; }

	for (auto& layer : _layers) {
		outfile << layer->to_string();
	}

	outfile.close();
	return true;	
}

bool SharPNetConv::load(std::string filepath)
{
	std::vector<layer*> layers;
	std::ifstream infile(filepath);
	std::string line;

	if (!infile.is_open()) { return false; }

	while (getline(infile, line)) {
		if (line.length() != 0) {
			if (line == "fullconnected") {
				getline(infile, line);
				tensor<float> tensor_input = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_output = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_weight = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_gradients = string_to_tensor(line);

				getline(infile, line);
				activation_t function;
				if (line == "Tanh") {
					function = activation_t::Tanh;
				}
				else if (line == "Sigmoid") {
					function = activation_t::Sigmoid;
				}
				else if (line == "Relu") {
					function = activation_t::Relu;
				}
				else if (line == "LRelu") {
					function = activation_t::LRelu;
				}

				layers.push_back(new FullConnected(tensor_input, tensor_output,
					tensor_weight, tensor_gradients, function));
			}

			if (line == "convolutional") {
				getline(infile, line);
				tensor<float> tensor_input = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_output = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_gradients = string_to_tensor(line);

				std::vector<tensor<float>> filters;

				for (;;) {
					getline(infile, line);
					if (line == "end") { break; }
					filters.push_back(string_to_tensor(line));
				}

				getline(infile, line);
				int filter_dem = stoi(line);

				getline(infile, line);
				int stride = stoi(line);

				layers.push_back(new ConvLayer(tensor_input, tensor_output, tensor_gradients, 
					filters, stride, filter_dem));
			}

			if (line == "relu") {
				getline(infile, line);
				tensor<float> tensor_input = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_output = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_gradients = string_to_tensor(line);

				layers.push_back(new ReluLayer(tensor_input, tensor_output, tensor_gradients));
			}

			if (line == "pooling") {
				getline(infile, line);
				tensor<float> tensor_input = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_output = string_to_tensor(line);

				getline(infile, line);
				tensor<float> tensor_gradients = string_to_tensor(line);

				getline(infile, line);
				int filter_dem = stoi(line);

				getline(infile, line);
				int stride = stoi(line);

				layers.push_back(new PoolingLayer(tensor_input, tensor_output, tensor_gradients, 
					filter_dem, stride));
			}
		}
	}

	infile.close();

	_layers = std::move(layers);
}
