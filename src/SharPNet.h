#ifndef SHARPNET_H
#define SHARPNET_H

#include <vector>
#include <functional>
#include <fstream>
#include "Neuron.h"
#include "activation.h"


class SharPNet
{
private:
	std::function<float(float)> _activation_function;
	std::function<float(float)> _activation_derviative;

	float _accuracy;
	float _training_accuracy;
	float _smoothing_factor;

	std::vector<Layer> _layers;

	void feed_forward(std::vector<float>& inputs);
	void back_propagation(std::vector<float>& outputs);

public:

	SharPNet(std::vector<int>& topology, activation_t activation);

	double train(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int nr_epochs);
	double evaluate(std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);
	void get_results(std::vector<float>& inputs, std::vector<float>& results);

	double get_accuracy() { return _accuracy; }

	bool save(std::ofstream& filename);
	bool load(std::ifstream& filename);
};


#endif