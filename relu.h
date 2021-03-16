#ifndef RELU_H
#define RELU_H

class ReluLayer : public layer
{

private:
	void activate();

public:

	explicit ReluLayer(td_size in_size);
	inline ReluLayer(const tensor<float>& in, const tensor<float>& out, const tensor<float>& grads)
	{
		_input = in;
		_output = out;
		_gradients = grads;
	}

	void activate(tensor<float>& in) {
		this->_input = in;
		activate();
	}

	
	void fix_weights(float learning_rate) { };
	void calc_grads(tensor<float>& grad_next_layer);
	std::string to_string();
};

inline ReluLayer::ReluLayer(td_size in_size)
{
	_input = tensor<float>(in_size._x, in_size._y, in_size._z);
	_output = tensor<float>(in_size._x, in_size._y, in_size._z);
	_gradients = tensor<float>(in_size._x, in_size._y, in_size._z);
}

inline void ReluLayer::activate()
{
	for (int i = 0; i < _input._size._x; i++) {
		for (int j = 0; j < _input._size._y; j++) {
			for (int k = 0; k < _input._size._z; k++) {
				float v = _input(i, j, k);
				if (v < 0) {
					v = 0;
				}

				_output(i, j, k) = v;
			}
		}
	}
}

inline void ReluLayer::calc_grads(tensor<float>& grad_next_layer)
{
	for (int i = 0; i < _input._size._x; i++) {
		for (int j = 0; j < _input._size._y; j++) {
			for (int k = 0; k < _input._size._z; k++) {
				_gradients(i, j, k) = (_input(i, j, k) < 0) ? 0 : 1 * grad_next_layer(i, j, k);
			}
		}
	}
}

inline std::string ReluLayer::to_string()
{
	std::stringstream ss;
	ss << "relu" << std::endl;
	ss << tensor_to_string(_input) << std::endl;
	ss << tensor_to_string(_output) << std::endl;
	ss << tensor_to_string(_gradients) << std::endl;
	return ss.str();
}
#endif // RELU_H
