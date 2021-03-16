#ifndef SHARPNETCONV_H
#define SHARPNETCONV_H

#include "tensor.h"
#include "layer.h"
#include "convolutional.h"
#include "fullconnected.h"
#include "relu.h"
#include "pooling.h"
#include "learning.h"

struct image_sample
{
	td_size image_shape;
	std::vector<std::vector<std::vector<float>>> data;
	std::vector<float> expected;
};

class SharPNetConv
{
private:

	float _training_accuracy;
	float _accuracy;
	float _smoothing_factor;
	loss_t _loss_function;
	float _learning_rate;

	std::vector<std::pair<float, float>> _history;
	std::vector<layer*> _layers;

	float calculate_loss(std::vector<tensor<float>> predictions, std::vector<tensor<float>> acutal);

	void feed_forword(tensor<float>& input);
	void back_propagation(tensor<float>& expected);

public:
	SharPNetConv(std::vector<layer*> topology, loss_t loss, float learning_rate = 0.01f);
	SharPNetConv(loss_t loss = loss_t::MeanSquaredError, float learning_rate = 0.01f) {
		_training_accuracy = 0.0f;
		_accuracy = 0.0f;
		_smoothing_factor = 0.0f;

		_loss_function = loss;
		this->_learning_rate = learning_rate;
	}

	std::vector<std::pair<float, float>> train(std::vector<image_sample> samples, int nr_epochs);
	float evaluate(std::vector<image_sample> samples);

	bool save(std::string filepath);
	bool load(std::string filepath);
};


#endif