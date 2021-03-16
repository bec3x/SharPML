#pragma once

#include <cassert>
#include <sstream>
#include <vector>
#include <iostream>

struct point
{
	int _x, _y, _z;
}; typedef point td_size;

template<typename T>
struct tensor
{
	T* _data;
	td_size _size;

	tensor() 
	{
		_data = nullptr;
		_size._x = 0;
		_size._y = 0;
		_size._z = 0;
	}

	tensor(int x, int y, int z) 
	{
		_data = new T[x * y * z];
		_size._x = x;
		_size._y = y;
		_size._z = z;
	}

	tensor(const tensor& other)
	{
		_data = new T[other._size._x * other._size._y * other._size._y];
		memcpy(this->_data, 
			   other._data, 
			   other._size._x * other._size._y * other._size._z * sizeof(T));
		this->_size = other._size;
	}

	tensor<T>& operator=(const tensor<T>& rhs)
	{
		if (&rhs == this) {
			return *this;
		}

		this->_data = new T[rhs._size._x * rhs._size._y * rhs._size._y];
		memcpy(this->_data, rhs._data, rhs._size._x * rhs._size._y * rhs._size._y * sizeof(T));
		this->_size = rhs._size;

		return *this;
	}

	tensor<T>& operator=(tensor<T>&& rhs) noexcept
	{
		if (&rhs == this) {
			return *this;
		}

		delete[] this->_data;

		this->_data = rhs._data;
		this->_size = rhs._size;
		rhs._data = nullptr;

		return *this;
	}

	tensor<T> operator+(tensor<T>& rhs)
	{
		tensor<T> copy(*this);
		int dimension_size = rhs._size._x * rhs._size._y * rhs._size._z;

		for (int i = 0; i < dimension_size; i++) {
			copy._data[i] += rhs._data[i];
		}

		return copy;
	}

	tensor<T> operator-(tensor<T>& rhs)
	{
		tensor<T> copy(*this);
		int dimension_size = rhs._size._x * rhs._size._y * rhs._size._z;

		for (int i = 0; i < dimension_size; i++) {
			copy._data[i] -= rhs._data[i];
		}

		return copy;
	}

	T& get(int x, int y, int z)
	{
		assert(x >= 0 && y >= 0 && z >= 0);
		assert(_size._x > x && _size._y > y && _size._z > z);

		return _data[z * (_size._x * _size._y) +
					y * (_size._x) +
					x];
	}

	T& operator()(int x, int y, int z)
	{
		return this->get(x, y, z);
	}

	void copy_from_vector(std::vector<std::vector<std::vector<T>>> data)
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				for (int k = 0; k < z; k++) {
					get(i, j, k) = data[k][j][i];
				}
			}
		}
	}

	~tensor() { 
		delete[] _data;
		this->_data = nullptr;
	}
};

static void print_tensor(tensor<float>& data)
{
	int x = data._size._x;
	int y = data._size._y;
	int z = data._size._z;

	for (int i = 0; i < z; i++) {
		std::cout << "[Dim] " << i << std::endl;

		for (int j = 0; j < y; j++) {
			for (int k = 0; k < x; k++) {
				std::cout << (float)data.get(k, j, i) << "\t";
			}
			std::cout << std::endl;
		}
	}
}

static void print_tensor(tensor<float>&& data) { print_tensor(data); }

static std::vector<std::vector<std::vector<float>>> tensor_to_vector(tensor<float> tensor)
{
	std::vector<std::vector<std::vector<float>>> a;

	for (int i = 0; i < tensor._size._x; i++) {
		std::vector<std::vector<float>> b;
		for (int j = 0; j < tensor._size._y; j++) {
			std::vector<float> c;
			for (int k = 0; k < tensor._size._z; k++) {
				c.push_back(tensor(i, j, k));
			}
			b.push_back(c);
		}
		a.push_back(b);
	}

	return a;
}

static tensor<float> convert_to_tensor(std::vector<std::vector<std::vector<float>>> data)
{
	int x = data.size();
	int y = data[0].size();
	int z = data[0][0].size();

	tensor<float> tensor(x, y, z);

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			for (int k = 0; k < z; k++) {
				tensor(i, j, k) = data[i][j][k];
			}
		}
	}

	return tensor;
}
static tensor<float> convert_to_tensor(std::vector<std::vector<float>> data)
{
	int x = data.size();
	int y = data[0].size();
	int z = 1;

	tensor<float> tensor(x, y, z);

	for (int i = 0; i < x; y++) {
		for (int j = 0; j < y; j++) {
			tensor(i, j, 0) = data[i][j];
		}
	}

	return tensor;
}

static tensor<float> convert_to_tensor(std::vector<float> data)
{
	int x = data.size();
	int y = 1;
	int z = 1;

	tensor<float> tensor(x, y, z);

	for (int i = 0; i < x; i++) {
		tensor(i, 0, 0) = data[i];
	}

	return tensor;
}

static std::string tensor_to_string(tensor<float>& data)
{
	std::stringstream ss;
	ss << data._size._x << " " << data._size._y << " " << data._size._z << " ";

	for (int z = 0; z < data._size._z; ++z) {
		for (int y = 0; y < data._size._y; ++y) {
			for (int x = 0; x < data._size._x; ++x) {
				ss << data.get(x, y, z) << " ";
			}
		}
	}

	return ss.str();
}

static tensor<float> string_to_tensor(const std::string& s)
{
	std::istringstream input_string_stream(s);
	std::vector<float> v((std::istream_iterator<float>(input_string_stream)),
		std::istream_iterator<float>());

	int x = (int)v[0];
	int y = (int)v[1];
	int z = (int)v[2];

	tensor<float> input_tensor = tensor<float>(x, y, z);

	for (int i = 0; i < z; i++) {
		for (int j = 0; j < y; j++) {
			for (int k = 0; k < x; k++) {
				input_tensor(k, j, i) = v[x * y * i + y * j + k + 3];
			}
		}
	}

	return input_tensor;
}
