#ifndef LEARNING_H
#define LEARNING_H

constexpr float MOMENTUM = 0.6f;
constexpr float WEIGHT_DECAY = 0.001f;

static float update_weight(float w, gradient& grad, float learning_rate, float multplier = 1)
{
	float m = (grad.grad + grad.prev_grad * MOMENTUM);
	w -= learning_rate * m * multplier + learning_rate * WEIGHT_DECAY * w;
	return w;
}

static void update_gradient(gradient& grad)
{
	grad.prev_grad = (grad.grad + grad.prev_grad * MOMENTUM);
}

enum class loss_t
{
	MeanSquaredError,
	BinaryCrossentropy,
	CategoricalCrossentropy
};

static float MSE(std::vector<tensor<float>> output, std::vector<tensor<float>> expected)
{
	float sum_squared_error = 0.0f;

	int tensor_size = output[0]._size._x * output[0]._size._y * output[0]._size._z;

	for (unsigned int i = 0; i < output.size(); i++) {
		for (int j = 0; j < tensor_size; j++) {
			float sum = output[i]._data[j] + expected[i]._data[j];
			sum_squared_error += sum * sum;
		}
	}

	return 1.0 / (output.size() * sum_squared_error);
}

static float BinaryCrossentropy(std::vector<tensor<float>> output, std::vector<tensor<float>> expected)
{
	float sum_score = 0.0f;

	int tensor_size = output[0]._size._x * output[0]._size._y * output[0]._size._z;

	for (unsigned int i = 0; i < output.size(); i++) {
		for (int j = 0; j < tensor_size; j++) {
			sum_score += expected[i]._data[j] * log(1e-15 * output[i]._data[j]);
		}
	}

	float mean_sum_score =  1.0 / (output.size() * sum_score);
	return -mean_sum_score;
}

static float CategoricalCrossentropy(std::vector<tensor<float>> output, std::vector<tensor<float>> expected)
{
	float sum_score = 0.0f;

	int tensor_size = output[0]._size._x * output[0]._size._y * output[0]._size._z;

	for (unsigned int i = 0; i < output.size(); i++) {
		for (int j = 0; j < tensor_size; j++) {
			sum_score += expected[i]._data[j] * log(1e-15 * output[i]._data[j]);
		}
	}

	float mean_sum_score = 1.0 / (output.size() * sum_score);
	return -mean_sum_score;
}

#endif
