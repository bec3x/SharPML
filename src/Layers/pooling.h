#ifndef POOLING_H
#define POOLING_H

class PoolingLayer : public layer
{
private:
	unsigned short _stride;
	unsigned short _filter_dem;

	int normalize_range(float f, int max, bool lim_min);
	point map_to_input(point out, int z);
	range map_to_output(int x, int y);
	void activate();

public:

	PoolingLayer(unsigned short stride, unsigned short filter_dem, td_size in_size);
	PoolingLayer(const tensor<float>& in, const tensor<float>& out, const tensor<float>& gradsIn, unsigned short extend_filter, unsigned short stride);

	void activate(tensor<float>& in) {
		this->_input = in;
		activate();
	}
		
	void fix_weights(float learning_rate) { }
	void calc_grads(tensor<float>& grad_next_layer);
	std::string to_string();
};

inline PoolingLayer::PoolingLayer(unsigned short stride, unsigned short filter_dem, td_size in_size)
{
	_input = tensor<float>(in_size._x, in_size._y, in_size._z);
	_gradients = tensor<float>(in_size._x, in_size._y, in_size._z);
	_output = tensor<float>((in_size._x - filter_dem) / stride + 1, (in_size._y - filter_dem) / stride + 1, in_size._z);

	_stride = stride;
	_filter_dem = filter_dem;

	assert((float(in_size._x - filter_dem) / stride + 1) == ((in_size._x - filter_dem) / stride + 1));
	assert((float(in_size._y - filter_dem) / stride + 1) == ((in_size._y - filter_dem) / stride + 1));
}

inline PoolingLayer::PoolingLayer(const tensor<float>& in, const tensor<float>& out, const tensor<float>& gradsIn, unsigned short filter_dem, unsigned short stride)
{
	_input = in;
	_output = out;
	_gradients = gradsIn;
	_filter_dem = filter_dem;
	_stride = stride;
}

inline point PoolingLayer::map_to_input(point out, int z)
{
	out._x *= _stride;
	out._y *= _stride;
	out._z = z;
	return out;
}

inline int PoolingLayer::normalize_range(float f, int max, bool lim_min)
{
	if (f <= 0) {
		return 0;
	}

	max -= 1;
	if (f >= max) {
		return max;
	}

	if (lim_min) {
		return (int)ceil(f);
	}
	else {
		return (int)floor(f);
	}
}

inline range PoolingLayer::map_to_output(int x, int y)
{
	float a = (float)x;
	float b = (float)y;

	return
	{
		normalize_range((a - _filter_dem + 1) / _stride, _output._size._x, true),
		normalize_range((b - _filter_dem + 1) / _stride, _output._size._y, true),
		0,
		normalize_range(a / _stride, _output._size._x, false),
		normalize_range(b / _stride, _output._size._y, false),
		(int)_output._size._z - 1,
	};
}

inline void PoolingLayer::activate()
{
	for (int x = 0; x < _output._size._x; x++) {
		for (int y = 0; y < _output._size._y; y++) {
			for (int z = 0; z < _output._size._z; z++) {
				point mapped = map_to_input({ (uint16_t)x, (uint16_t)y, 0 }, 0);

				float mval = -FLT_MAX;
				for (int i = 0; i < _filter_dem; i++) {
					for (int j = 0; j < _filter_dem; j++) {
						float v = _input(mapped._x + i, mapped._y + j, z);
						if (v > mval) {
							mval = v;
						}
					}
				}

				_output(x, y, z) = mval;
			}
		}
	}
}

inline void PoolingLayer::calc_grads(tensor<float>& grad_next_layer)
{
	for (int x = 0; x < _input._size._x; x++) {
		for (int y = 0; y < _input._size._y; y++) {
			range rn = map_to_output(x, y);

			for (int z = 0; z < _input._size._z; z++) {
				float sum_error = 0;
				for (int i = rn.min_x; i <= rn.max_x; i++) {
					int minx = i * _stride;

					for (int j = rn.min_y; j <= rn.max_y; j++) {
						int miny = j * _stride;

						int is_max = _input(x, y, z) == _output(i, j, z) ? 1 : 0;
						sum_error += is_max * grad_next_layer(i, j, z);
					}
				}

				_gradients(x, y, z) = sum_error;
			}
		}
	}
}

inline std::string PoolingLayer::to_string()
{
	std::stringstream ss;
	ss << "pooling" << std::endl;
	ss << tensor_to_string(_input) << std::endl;
	ss << tensor_to_string(_output) << std::endl;
	ss << tensor_to_string(_gradients) << std::endl;
	ss << _filter_dem << std::endl;
	ss << _stride << std::endl;

	return ss.str();
}
#endif