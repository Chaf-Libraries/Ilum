#include "BSpline.hpp"

__pragma(warning(push, 0))
#include <Eigen/Eigen>
    __pragma(warning(pop))

#include <tbb/tbb.h>

        namespace Ilum::geometry
{
	inline float genBasis(const std::vector<float> &T, float t, size_t i, size_t k)
	{
		if (k == 1)
		{
			if ((t >= T[i] && t < T[i + 1]) || (t >= T[i] && t <= T[i + 1] && T[i + 1] == T.back()))
			{
				return 1.0;
			}
			else
			{
				return 0.0;
			}
		}
		float left = 0.f;
		left       = T[i + k - 1ull] - T[i] == 0.f ? 0.f : (t - T[i]) / (T[i + k - 1] - T[i]) * genBasis(T, t, i, k - 1ull);

		float right = 0.f;
		right       = T[i + k] - T[i + 1ull] == 0.f ? 0.f : (T[i + k] - t) / (T[i + k] - T[i + 1ull]) * genBasis(T, t, i + 1ull, k - 1ull);

		return left + right;
	}

	std::vector<glm::vec3> BSpline::generateVertices(const std::vector<glm::vec3> &control_points, uint32_t sample)
	{
		if (control_points.empty())
		{
			return {};
		}

		if (m_de_boor.empty())
		{
			genDeBoorPoints(control_points);
		}

		std::vector<glm::vec3> vertices(sample + 1);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, sample + 1u), [this, control_points, sample, &vertices](const tbb::blocked_range<size_t> &count) {
			for (size_t i = count.begin(); i != count.end(); i++)
			{
				float t     = m_knots.back() / static_cast<float>(sample) * static_cast<float>(i);
				vertices[i] = value(control_points, t);
			}
		});
		return vertices;
	}

	glm::vec3 BSpline::value(const std::vector<glm::vec3> &control_points, float t)
	{
		if (m_de_boor.empty())
		{
			genDeBoorPoints(control_points);
		}

		glm::vec3 result = glm::vec3(0.f);

		for (size_t i = 0; i < m_de_boor.size(); i++)
		{
			result += genBasis(m_knots, t, i, 4) * m_de_boor[i];
		}

		return result;
	}

	void BSpline::genDeBoorPoints(const std::vector<glm::vec3> &control_points)
	{
		std::vector<float> s;

		// Knot sequence
		m_knots.clear();
		m_knots.push_back(0);
		m_knots.push_back(0);
		m_knots.push_back(0);

		for (size_t i = 0; i < control_points.size(); i++)
		{
			m_knots.push_back(static_cast<float>(i));
			s.push_back(static_cast<float>(i));
		}

		m_knots.push_back(static_cast<float>(control_points.size() - 1));
		m_knots.push_back(static_cast<float>(control_points.size() - 1));
		m_knots.push_back(static_cast<float>(control_points.size() - 1));

		size_t n = m_knots.size() - 4;

		m_de_boor.resize(n);

		Eigen::MatrixXf A(n, n);
		Eigen::MatrixXf b(n, 3);

		A.setZero();
		b.setZero();

		// Begin
		{
			A(0, 0) = 1;

			A(1, 0) = 2;
			A(1, 1) = -3;
			A(1, 2) = 1;

			b(0, 0) = control_points[0].x;
			b(0, 1) = control_points[0].y;
			b(0, 2) = control_points[0].z;
		}

		// Inner
		{
			for (size_t i = 1; i < control_points.size() - 1; i++)
			{
				A(i + 1, i)     = genBasis(m_knots, s[i], i, 4);
				A(i + 1, i + 1) = genBasis(m_knots, s[i], i + 1, 4);
				A(i + 1, i + 2) = genBasis(m_knots, s[i], i + 2, 4);
				b(i + 1, 0)     = control_points[i].x;
				b(i + 1, 1)     = control_points[i].y;
				b(i + 1, 2)     = control_points[i].z;
			}
		}

		// End
		{
			A(n - 2, n - 3) = 1;
			A(n - 2, n - 2) = -3;
			A(n - 2, n - 1) = 2;

			A(n - 1, n - 1) = 1;

			b(n - 1, 0) = control_points.back().x;
			b(n - 1, 1) = control_points.back().y;
			b(n - 1, 2) = control_points.back().z;
		}

		Eigen::MatrixXf res = A.colPivHouseholderQr().solve(b);

		for (size_t i = 0; i < m_de_boor.size(); i++)
		{
			m_de_boor[i] = glm::vec3(res(i, 0), res(i, 1), res(i, 2));
		}
	}
}        // namespace Ilum::geometry