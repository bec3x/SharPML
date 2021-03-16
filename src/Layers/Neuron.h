#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <functional>

struct Edge
{
	float weight;
	float delta_wt;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned int nr_outputs, unsigned int index);

	void feed_forward(const Layer& prevLayer, std::function<float(float)> activation);
	void calculate_output_gradient(float targetVal, std::function<float(float)> derivitive);
	void calculate_hidden_gradient(const Layer& nextLayer, std::function<float(float)> derivitive);
	void update_input_weights(Layer& prevLayer);

	void set_output_val(float val) { _output_val = val; }
	float get_output_val(void) const { return _output_val; }

	void set_gradient(float val) { _gradient = val; }
	float get_gradient(void) const { return _gradient; }

private:
	std::vector<Edge> _output_weights;
	static float random_weight(void) { return rand() / static_cast<float>(RAND_MAX); }
	
	float _output_val;
	float _gradient;
	int _index;


	float _eta = .15f;
	float _alpha = .5f;

	float sum_gradients_by_weights(const Layer& nextLayer) const;
};

#endif // !NEURON_H
