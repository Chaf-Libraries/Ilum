#include "BezierSpline.hpp"
#include "BezierCurve.hpp"

__pragma(warning(push, 0))
#include <Eigen/Eigen>
    __pragma(warning(pop))

#include <tbb/tbb.h>

#include <iostream>

        namespace Ilum::geometry
{
	std::vector<glm::vec3> BezierSpline::generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
	{
		BezierCurve bezier_curve;

		if (control_points.size() <= 2)
		{
			return bezier_curve.generateVertices(control_points, sample);
		}

		if (this->control_points.empty())
		{
			generateControlPoints(control_points);
		}

		uint32_t patch = sample / static_cast<uint32_t>(control_points.size() - 1);

		std::vector<glm::vec3> result;

		for (uint32_t i = 0; i < control_points.size() - 1; i++)
		{
			auto frag = std::move(bezier_curve.generateVertices({this->control_points[3 * i],
			                                                     this->control_points[3 * i + 1],
			                                                     this->control_points[3 * i + 2],
			                                                     this->control_points[3 * i + 3]},
			                                                    patch));
			result.insert(result.end(), std::make_move_iterator(frag.begin()), std::make_move_iterator(frag.end()));
		}

		return result;
	}

	glm::vec3 BezierSpline::value(const std::vector<glm::vec3> &control_points, float t)
	{
		BezierCurve bezier_curve;

		if (control_points.size() <= 2)
		{
			return bezier_curve.value(control_points, t);
		}

		if (this->control_points.empty())
		{
			generateControlPoints(control_points);
		}

		return bezier_curve.value(this->control_points, t);
	}

	void BezierSpline::generateControlPoints(const std::vector<glm::vec3> &control_points)
	{
		if (control_points.size() < 2)
		{
			return;
		}

		size_t n = control_points.size() - 1;

		Eigen::MatrixXf A(3 * n + 1, 3 * n + 1);
		Eigen::MatrixXf b(3 * n + 1, 3);
		A.setZero();
		b.setZero();

		tbb::parallel_for(tbb::blocked_range<size_t>(0, n + 1u), [this, n, control_points, &A, &b](const tbb::blocked_range<size_t> &count) {
			for (size_t i = count.begin(); i != count.end(); i++)
			{
				A(i, 3 * i) = 1;

				if (i >= 1 && i <= n - 1)
				{
					A(n + i, 3 * i - 1) = -1;
					A(n + i, 3 * i)     = 2;
					A(n + i, 3 * i + 1) = -1;

					A(2 * n + i - 1, 3 * i - 2) = 1;
					A(2 * n + i - 1, 3 * i - 1) = -2;
					A(2 * n + i - 1, 3 * i)     = 0;
					A(2 * n + i - 1, 3 * i + 1) = 2;
					A(2 * n + i - 1, 3 * i + 2) = -1;
				}

				b(i, 0) = control_points[i].x;
				b(i, 1) = control_points[i].y;
				b(i, 2) = control_points[i].z;
			}
		});

		// End condition
		A(3 * n - 1, 0) = 1;
		A(3 * n - 1, 1) = -2;
		A(3 * n - 1, 2) = 1;

		A(3 * n, 3 * n - 2) = 1;
		A(3 * n, 3 * n - 1) = -2;
		A(3 * n, 3 * n)     = 1;

		Eigen::MatrixXf x = A.colPivHouseholderQr().solve(b);

		this->control_points.resize(3 * n + 1);

		for (size_t i = 0; i < 3 * n + 1; i++)
		{
			this->control_points[i].x = x(i, 0);
			this->control_points[i].y = x(i, 1);
			this->control_points[i].z = x(i, 2);
		}
	}
}        // namespace Ilum::geometry