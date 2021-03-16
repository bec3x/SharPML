#ifndef FULLCONNECTED_H
#define FULLCONNECTED_H

#include <functional>
#include "activation.h"

class FullConnected : public layer
{
private:
	std::vector<float> _output_val;
	tensor<float> _weights;
	std::vector<gradient> _grads;
	std::function<double(double)> _activation_function;
	std::function<double(double)> _activation_derivative;

	activation_t _act_fcn;

	int map(point d);
	void activate();
public:

	FullConnected(td_size in_size, int output_size, activation_t act_fcn = activation_t::Tanh);
	FullConnected(const tensor<float>& in, const tensor<float>& out, const tensor<float>& weights, const tensor<float>& gradsIn, activation_t act_fcn = activation_t::Tanh);

	void activate(tensor<float>& in) {
		_input = in;
		activate();
	}

	void fix_weights(float learning_rate);
	void calc_grads(tensor<float>& grad_next_layer);
	std::string to_string();
};

inline FullConnected::FullConnected(td_size in_size, int output_size, activation_t act_fcn)
{
	_act_fcn = act_fcn;

	switch (act_fcn) {
	case activation_t::Tanh:
		_activation_function = net_tanh;
		_activation_derivative = tanh_dev;
		break;
	case activation_t::Sigmoid:
		_activation_function = sig;
		_activation_derivative = sigmoid_dev;
		break;
	case activation_t::Relu:
		_activation_function = relu;
		_activation_derivative = relu_dev;
		break;
	case activation_t::LRelu:
		_activation_function = lRelu;
		_activation_derivative = lRelu_dev;
		break;
	}

	_input = tensor<float>(in_size._x, in_size._y, in_size._z);
	_output = tensor<float>(output_size, 1, 1);
	_gradients = tensor<float>(in_size._x, in_size._y, in_size._z);

	_output_val = std::vector<float>(output_size);
	_grads = std::vector<gradient>(output_size);
	_weights = tensor<float>(in_size._x * in_size._y * in_size._z, output_size, 1);

	int max_index = in_size._x * in_size._y * in_size._z;

	for (int i = 0; i < output_size; i++) {
		for (int j = 0; j < max_index; j++) {
			_weights(j, i, 0) = ((rand() / float(RAND_MAX)) * 2) - 1;
		}
	}
}

inline FullConnected::FullConnected(const tensor<float>& in, const tensor<float>& out, const tensor<float>& weights,
	const tensor<float>& grads, activation_t act_fcn)
{
	_act_fcn = act_fcn;

	switch (act_fcn) {
	case activation_t::Tanh:
		_activation_function = net_tanh;
		_activation_derivative = tanh_dev;
		break;
	case activation_t::Sigmoid:
		_activation_function = sig;
		_activation_derivative = sigmoid_dev;
		break;
	case activation_t::Relu:
		_activation_function = relu;
		_activation_derivative = relu_dev;
		break;
	case activation_t::LRelu:
		_activation_function = lRelu;
		_activation_derivative = lRelu_dev;
		break;
	}

	_input = in;
	_output = out;
	_weights = weights;
	_gradients = grads;
}

inline int FullConnected::map(point d)
{
	return d._z * (_input._size._x * _input._size._y) +
		d._y * (_input._size._x) +
		d._x;
}

inline void FullConnected::activate()
{
	for (int n = 0; n < _output._size._x; n++) {
		float sum = 0;

		for (int i = 0; i < _input._size._x; i++) {
			for (int j = 0; j < _input._size._y; j++) {
				for (int k = 0; k < _input._size._z; k++) {
					int neuron = map({ i, j, k });
					sum += _input(i, j, k) * _weights(neuron, n, 0);
				}
			}
		}

		_output_val[n] = sum;
		_output(n, 0, 0) = _activation_function(sum);
	}
}

inline void FullConnected::fix_weights(float learning_rate)
{
	for (int n = 0; n < _output._size._x; n++) {
		gradient& grad = _grads[n];

		for (int i = 0; i < _input._size._x; i++) {
			for (int j = 0; j < _input._size._y; j++) {
				for (int k = 0; k < _input._size._z; k++) {
					int neuron = map({ i, j, k });
					float& w = _weights(neuron, n, 0);
					w = update_weight(w, grad, learning_rate, _input(i, j, k));
				}
			}
		}

		update_gradient(grad);
	}
}

inline void FullConnected::calc_grads(tensor<float>& grad_next_layer)
{
	int input_grad_size = _gradients._size._x * _gradients._size._y * _gradients._size._z;
	memset(_gradients._data, 0, input_grad_size * sizeof(float));

	for (unsigned int n = 0; n < _output._size._x; n++) {
		gradient& grad = _grads[n];
		grad.grad = grad_next_layer(n, 0, 0) * _activation_derivative(_output_val[n]);

		for (int i = 0; i < _input._size._x; i++) {
			for (int j = 0; j < _input._size._y; j++) {
				for (int k = 0; k < _input._size._z; k++) {
					int m = map({ i, j, k });
					_gradients(i, j, k) += grad.grad * _weights(m, n, 0);
				}				
			}
		}
	}
}

inline std::string FullConnected::to_string()
{
	std::stringstream ss;

	ss << "fullconnected" << std::endl;
	ss << tensor_to_string(_input) << std::endl;
	ss << tensor_to_string(_output) << std::endl;
	ss << tensor_to_string(_weights) << std::endl;
	ss << tensor_to_string(_gradients) << std::endl;
	
	if (_act_fcn == activation_t::Tanh) {
		ss << "Tanh" << std::endl;
	}
	else if (_act_fcn == activation_t::Sigmoid) {
		ss << "Sigmoid" << std::endl;
	}
	else if (_act_fcn == activation_t::Relu) {
		ss << "Relu" << std::endl;
	}
	else if (_act_fcn == activation_t::LRelu) {
		ss << "LRelu" << std::endl;
	}

	return ss.str();
}

#endif
