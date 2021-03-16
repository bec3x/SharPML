#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

struct gradient
{
	float grad;
	float prev_grad;
	gradient()
	{
		grad = 0;
		prev_grad = 0;
	}
};

struct range
{
	int min_x, min_y, min_z;
	int max_x, max_y, max_z;
};

class layer
{
public:
	virtual void activate(tensor<float>& input) = 0;
	virtual void activate() = 0;

	virtual void fix_weights(float learning_rate) = 0;
	virtual void calc_grads(tensor<float>& grad_next_layer) = 0;

	virtual std::string to_string() = 0;

	td_size get_output_size() const { return _output._size; }

	tensor<float> get_input() const { return _input; }
	tensor<float> get_output() const { return _output; }
	tensor<float> get_gradients() const { return _gradients; }

protected:
	tensor<float> _gradients;
	tensor<float> _input;
	tensor<float> _output;
};

#endif // !LAYER_H
