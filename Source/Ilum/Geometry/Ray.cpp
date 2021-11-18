#include "Ray.hpp"
#include "BoundingBox.hpp"

namespace Ilum::geometry
{
Ray::Ray(const glm::vec3 &origin, const glm::vec3 &direction) :
    origin(origin), direction(direction)
{
}

glm::vec3 Ray::project(const glm::vec3 &point) const
{
	return origin + glm::dot((point - origin), direction) * direction;
}

float Ray::distance(const glm::vec3 &point) const
{
	return glm::length((point-project(point)));
}

float Ray::hit(const BoundingBox &bbox)
{
	if (!bbox.valid())
	{
		return std::numeric_limits<float>::infinity();
	}

	if (bbox.isInside(origin))
	{
		return 0.f;
	}

	float distance = std::numeric_limits<float>::infinity();

	// Check X
	if (origin.x < bbox.min_.x && direction.x > 0.0f)
	{
		float x = (bbox.min_.x - origin.x) / direction.x;
		if (x < distance)
		{
			glm::vec3 point = origin + x * direction;
			if (point.y >= bbox.min_.y && point.y <= bbox.max_.y && point.z >= bbox.min_.z && point.z <= bbox.max_.z)
			{
				distance = x;
			}
		}
	}

	if (origin.x > bbox.max_.x && direction.x < 0.0f)
	{
		float x = (bbox.max_.x - origin.x) / direction.x;
		if (x < distance)
		{
			glm::vec3 point = origin + x * direction;
			if (point.y >= bbox.min_.y && point.y <= bbox.max_.y && point.z >= bbox.min_.z && point.z <= bbox.max_.z)
			{
				distance = x;
			}
		}
	}

	// Check Y
	if (origin.y < bbox.min_.y && direction.y > 0.0f)
	{
		float y = (bbox.min_.y - origin.y) / direction.y;
		if (y < distance)
		{
			glm::vec3 point = origin + y * direction;
			if (point.x >= bbox.min_.x && point.x <= bbox.max_.x && point.z >= bbox.min_.z && point.z <= bbox.max_.z)
			{
				distance = y;
			}
		}
	}

	if (origin.y > bbox.max_.y && direction.y < 0.0f)
	{
		float y = (bbox.max_.y - origin.y) / direction.y;
		if (y < distance)
		{
			glm::vec3 point = origin + y * direction;
			if (point.x >= bbox.min_.x && point.x <= bbox.max_.x && point.z >= bbox.min_.z && point.z <= bbox.max_.z)
			{
				distance = y;
			}
		}
	}

	// Check Z
	if (origin.z < bbox.min_.z && direction.z > 0.0f)
	{
		float z = (bbox.min_.z - origin.z) / direction.z;
		if (z < distance)
		{
			glm::vec3 point = origin + z * direction;
			if (point.x >= bbox.min_.x && point.x <= bbox.max_.x && point.y >= bbox.min_.y && point.y <= bbox.max_.y)
			{
				distance = z;
			}
		}
	}

	if (origin.z > bbox.max_.z && direction.z < 0.0f)
	{
		float z = (bbox.max_.z - origin.z) / direction.z;
		if (z < distance)
		{
			glm::vec3 point = origin + z * direction;
			if (point.x >= bbox.min_.x && point.x <= bbox.max_.x && point.y >= bbox.min_.y && point.y <= bbox.max_.y)
			{
				distance = z;
			}
		}
	}

	return distance;
}
}        // namespace Ilum::geometry