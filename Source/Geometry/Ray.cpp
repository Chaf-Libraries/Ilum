#include "Ray.hpp"
#include "Bound.hpp"

namespace Ilum::Geo
{
Ray::Ray(const glm::vec3 &origin, const glm::vec3 &direction) :
    m_origin(origin), m_direction(direction)
{
}

glm::vec3 Ray::Project(const glm::vec3 &point) const
{
	return m_origin + glm::dot((point - m_origin), m_direction) * m_direction;
}

float Ray::Distance(const glm::vec3 &point) const
{
	return glm::length((point - Project(point)));
}

float Ray::Hit(const Bound &bbox)
{
	if (!bbox)
	{
		return std::numeric_limits<float>::infinity();
	}

	if (bbox.IsInside(m_origin))
	{
		return 0.f;
	}

	float distance = std::numeric_limits<float>::infinity();

	// Check X
	if (m_origin.x < bbox.GetMin().x && m_direction.x > 0.0f)
	{
		float x = (bbox.GetMin().x - m_origin.x) / m_direction.x;
		if (x < distance)
		{
			glm::vec3 point = m_origin + x * m_direction;
			if (point.y >= bbox.GetMin().y && point.y <= bbox.GetMax().y && point.z >= bbox.GetMin().z && point.z <= bbox.GetMax().z)
			{
				distance = x;
			}
		}
	}

	if (m_origin.x > bbox.GetMax().x && m_direction.x < 0.0f)
	{
		float x = (bbox.GetMax().x - m_origin.x) / m_direction.x;
		if (x < distance)
		{
			glm::vec3 point = m_origin + x * m_direction;
			if (point.y >= bbox.GetMin().y && point.y <= bbox.GetMax().y && point.z >= bbox.GetMin().z && point.z <= bbox.GetMax().z)
			{
				distance = x;
			}
		}
	}

	// Check Y
	if (m_origin.y < bbox.GetMin().y && m_direction.y > 0.0f)
	{
		float y = (bbox.GetMin().y - m_origin.y) / m_direction.y;
		if (y < distance)
		{
			glm::vec3 point = m_origin + y * m_direction;
			if (point.x >= bbox.GetMin().x && point.x <= bbox.GetMax().x && point.z >= bbox.GetMin().z && point.z <= bbox.GetMax().z)
			{
				distance = y;
			}
		}
	}

	if (m_origin.y > bbox.GetMax().y && m_direction.y < 0.0f)
	{
		float y = (bbox.GetMax().y - m_origin.y) / m_direction.y;
		if (y < distance)
		{
			glm::vec3 point = m_origin + y * m_direction;
			if (point.x >= bbox.GetMin().x && point.x <= bbox.GetMax().x && point.z >= bbox.GetMin().z && point.z <= bbox.GetMax().z)
			{
				distance = y;
			}
		}
	}

	// Check Z
	if (m_origin.z < bbox.GetMin().z && m_direction.z > 0.0f)
	{
		float z = (bbox.GetMin().z - m_origin.z) / m_direction.z;
		if (z < distance)
		{
			glm::vec3 point = m_origin + z * m_direction;
			if (point.x >= bbox.GetMin().x && point.x <= bbox.GetMax().x && point.y >= bbox.GetMin().y && point.y <= bbox.GetMax().y)
			{
				distance = z;
			}
		}
	}

	if (m_origin.z > bbox.GetMax().z && m_direction.z < 0.0f)
	{
		float z = (bbox.GetMax().z - m_origin.z) / m_direction.z;
		if (z < distance)
		{
			glm::vec3 point = m_origin + z * m_direction;
			if (point.x >= bbox.GetMin().x && point.x <= bbox.GetMax().x && point.y >= bbox.GetMin().y && point.y <= bbox.GetMax().y)
			{
				distance = z;
			}
		}
	}

	return distance;
}
}