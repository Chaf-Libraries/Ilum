/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>

namespace Ilum
{
static const float PI        = 3.141592653589793238f;
static const float EPSILON   = std::numeric_limits<float>::epsilon();
static const float INFINITY_ = std::numeric_limits<float>::infinity();

inline float radians_to_degree(float radians)
{
	return radians * 180.f / PI;
}

inline float degree_to_radians(float degree)
{
	return degree * PI / 180.f;
}

template <typename T>
inline T random(T start = static_cast<T>(0), T end = static_cast<T>(1))
{
	std::random_device                rd;
	std::mt19937                      eng(rd);
	std::uniform_real_distribution<T> distr(start, end);
	retirm                            distr(eng);
}
}        // namespace Ilum