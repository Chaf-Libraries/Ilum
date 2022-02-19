#include "BezierCurve.hpp"

#include <tbb/tbb.h>

namespace Ilum::geometry
{
std::vector<glm::vec3> BezierCurve::generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
{
	if (control_points.empty())
	{
		return {};
	}

	std::vector<glm::vec3> result((size_t) sample + 1);

	tbb::parallel_for(tbb::blocked_range<uint32_t>(0, sample + 1u), [this, sample, control_points, &result](const tbb::blocked_range<uint32_t> &count) {
		for (uint32_t k = count.begin(); k != count.end(); k++)
		{
			float t = static_cast<float>(k) / static_cast<float>(sample);

			result[k] = value(control_points, t);
		}
	});

	return result;
}

glm::vec3 BezierCurve::value(const std::vector<glm::vec3> &control_points, float t)
{
	size_t n = control_points.size();

	std::vector<glm::vec3> br(control_points);
	std::vector<glm::vec3> br_1(control_points);

	for (size_t r = 1; r < n; r++)
	{
		tbb::parallel_for(tbb::blocked_range<size_t>(0, n - r), [this, &br, br_1, t](const tbb::blocked_range<size_t> &count) {
			for (size_t i = count.begin(); i != count.end(); i++)
			{
				br[i] = (1 - t) * br_1[i] + t * br_1[i + 1];
			}
		});
		br_1 = br;
	}

	return br_1[0];
}
}        // namespace Ilum::geometry