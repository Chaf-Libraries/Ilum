#include "RationalBSpline.hpp"

#include <tbb/tbb.h>

namespace Ilum::geometry
{
inline float gen_basis(uint32_t i, uint32_t k, float t)
{
	if (k == 1)
	{
		if (t >= i && t < i + 1)
		{
			return 1.f;
		}
		else
		{
			return 0.f;
		}
	}

	float left = 0.f;
	left       = static_cast<float>(k) - 1.f == 0.f ? 0.f : (t - static_cast<float>(i)) / (static_cast<float>(k) - 1.f) * gen_basis(i, k - 1, t);

	float right = 0.f;
	right       = static_cast<float>(k) - 1.f == 0.f ? 0.f : (static_cast<float>(i + k) - t) / (static_cast<float>(k) - 1.f) * gen_basis(i + 1, k - 1, t);

	return left + right;
}

std::vector<glm::vec3> RationalBSpline::generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
{
	if (control_points.empty())
	{
		return {};
	}

	std::vector<glm::vec3> vertices(static_cast<size_t>(sample) + 1ul);

	tbb::parallel_for(tbb::blocked_range<size_t>(0, static_cast<size_t>(sample) + 1ul), [this, control_points, sample, &vertices](const tbb::blocked_range<size_t> &count) {
		for (size_t i = count.begin(); i != count.end(); i++)
		{
			float t     = static_cast<float>(control_points.size() + 1 - order) / static_cast<float>(sample) * static_cast<float>(i) + static_cast<float>(order) - 1.f;
			vertices[i] = value(control_points, t);
		}
	});
	return vertices;
}

glm::vec3 RationalBSpline::value(const std::vector<glm::vec3> &control_points, float t)
{
	uint32_t  n      = static_cast<uint32_t>(control_points.size());

	float     denominator = 0.f;
	glm::vec3 numerator   = glm::vec3(0.f);

	for (uint32_t i = 0; i < n; i++)
	{
		float weight = gen_basis(i, order, t) * weights[i];
		denominator += weight;
		numerator += weight * control_points[i];
	}

	return numerator / denominator;
}
}