#ifndef CONVLAYER_H
#define CONVLAYER_H

#include "layer.h"
#include "learning.h"
#include "tensor.h"

class ConvLayer : public layer
{
private:

	std::vector<tensor<float>> _filters;
	std::vector<tensor<gradient>> _filter_gradients;

	unsigned short _stride;
	unsigned short _filter_dem;

	point map_to_input(point output, int z);
	range map_to_output(int x, int y);
	int normalize_range(float f, int max, bool lim_min);

	void activate();
public:

	ConvLayer(unsigned short stride, unsigned short filter_dim, unsigned short nr_filters, td_size in_size);
	ConvLayer(const tensor<float>& input, const tensor<float>& output, const tensor<float>& input_gradients, std::vector<tensor<float>> filters, unsigned short stride, unsigned short filter_dim);

	void activate(tensor<float>& in) {
		this->_input = in;
		activate();
	}

	void fix_weights(float learning_rate);
	void calc_grads(tensor<float>& grad_next_layer);
	std::string to_string();
};

inline ConvLayer::ConvLayer(unsigned short stride, unsigned short filter_dem, unsigned short nr_filters, td_size in_size)
{
	_gradients = tensor<float>(in_size._x, in_size._y, in_size._z);
	_input = tensor<float>(in_size._x, in_size._y, in_size._z);
	_output = tensor<float>((in_size._x - filter_dem) / stride + 1,
		(in_size._y - filter_dem) / stride + 1,
		nr_filters);

	_stride = stride;
	_filter_dem = filter_dem;

	// Add Padding?
	// https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
	assert((float(in_size._x - filter_dem) / stride + 1)
		==
		((in_size._x - filter_dem) / stride + 1));

	assert((float(in_size._y - filter_dem) / stride + 1)
		==
		((in_size._y - filter_dem) / stride + 1));

	for (int i = 0; i < nr_filters; i++) {
		tensor<float> tensor(filter_dem, filter_dem, in_size._z);

		for (int i = 0; i < filter_dem; i++) {
			for (int j = 0; j < filter_dem; j++) {
				for (int k = 0; k < in_size._z; k++) {
					tensor(i, j, k) = ((rand() / float(RAND_MAX)) * 2) - 1;
				}
			}
		}

		_filters.push_back(tensor);
	}

	for (unsigned int i = 0; i < nr_filters; i++) {
		tensor<gradient> tensor(filter_dem, filter_dem, in_size._z);
		_filter_gradients.push_back(tensor);
	}
}

inline ConvLayer::ConvLayer(const tensor<float>& input, const tensor<float>& output, const tensor<float>& input_gradients,
	std::vector<tensor<float>> filters, unsigned short stride, unsigned short filter_dem)
{
	_input = input;
	_output = output;
	_gradients = input_gradients;
	_filters = std::move(filters);
	_stride = stride;
	_filter_dem = filter_dem;
}

inline point ConvLayer::map_to_input(point output, int z)
{
	output._x *= _stride;
	output._y *= _stride;
	output._z = z;
	return output;
}

inline range ConvLayer::map_to_output(int x, int y)
{
	float a = x;
	float b = y;

	return
	{
		normalize_range((a - _filter_dem + 1) / _stride, _output._size._x, true),
		normalize_range((b - _filter_dem + 1) / _stride, _output._size._y, true),
		0,
		normalize_range(a / _stride, _output._size._x, false),
		normalize_range(b / _stride, _output._size._y, false),
		(int)_filters.size() - 1
	};
}

inline int ConvLayer::normalize_range(float f, int max, bool lim_min)
{
	if (f <= 0)
		return 0;

	max -= 1;
	if (f >= max)
		return max;

	if (lim_min)
		return ceil(f);
	else
		return floor(f);
}

inline void ConvLayer::activate()
{
	for (int filter = 0; filter < _filters.size(); filter++) {
		tensor<float>& filter_data = _filters[filter];

		for (int x = 0; x < _output._size._x; x++) {
			for (int y = 0; y < _output._size._y; y++) {
				point mapped = map_to_input({ (uint16_t)x, (uint16_t)y, 0 }, 0);
				float sum = 0;

				for (int i = 0; i < _filter_dem; i++) {
					for (int j = 0; j < _filter_dem; j++) {
						for (int z = 0; z < _input._size._z; z++) {
							float f = filter_data(i, j, z);
							float v = _input(mapped._x + i, mapped._y + j, z);
							sum += f * v;
						}
					}
				}

				_output(x, y, filter) = sum;
			}
		}
	}
}

inline void ConvLayer::fix_weights(float learning_rate)
{
	for (int a = 0; a < _filters.size(); a++) {
		for (int i = 0; i < _filter_dem; i++) {
			for (int j = 0; j < _filter_dem; j++) {
				for (int z = 0; z < _input._size._z; z++) {
					float& weight = _filters[a].get(i, j, z);
					gradient& grad = _filter_gradients[a].get(i, j, z);
					weight = update_weight(weight, grad, learning_rate);
					update_gradient(grad);
				}
			}
		}
	}
}

inline void ConvLayer::calc_grads(tensor<float>& next_layer_grad)
{
	for (unsigned int k = 0; k < _filter_gradients.size(); k++) {
		for (unsigned int i = 0; i < _filter_dem; i++) {
			for (unsigned int j = 0; j < _filter_dem; j++) {
				for (unsigned int z = 0; z < _input._size._z; z++) {
					_filter_gradients[k].get(i, j, z).grad = 0;
				}
			}
		}
	}

	for (int x = 0; x < _input._size._x; x++) {
		for (unsigned int y = 0; y < _input._size._y; y++) {
			range rn = map_to_output(x, y);

			for (unsigned int z = 0; z < _input._size._z; z++) {
				float sum_error = 0;
				for (unsigned int i = rn.min_x; i <= rn.max_x; i++) {
					int minx = i * _stride;

					for (unsigned int j = rn.min_y; j <= rn.max_y; j++) {
						int miny = j * _stride;

						for (unsigned int k = rn.min_z; k <= rn.max_z; k++) {
							int w_applied = _filters[k].get(x - minx, y - miny, z);
							sum_error += w_applied * next_layer_grad(i, j, k);
							_filter_gradients[k].get(x - minx, y - miny, z).grad += _input(x, y, z) * next_layer_grad(i, j, k);
						}
					}
				}

				_gradients(x, y, z) = sum_error;
			}
		}
	}
}

inline std::string ConvLayer::to_string()
{
	std::stringstream ss;
	ss << "convolutional" << std::endl;
	ss << tensor_to_string(_input) << std::endl;
	ss << tensor_to_string(_output) << std::endl;
	ss << tensor_to_string(_gradients) << std::endl;

	for (tensor<float> t : _filters) {
		ss << tensor_to_string(t) << std::endl;
	}

	ss << "end" << std::endl;
	ss << _filter_dem << std::endl;
	ss << _stride << std::endl;
	return ss.str();
}

#endif