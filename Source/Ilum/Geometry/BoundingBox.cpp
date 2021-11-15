#include "BoundingBox.hpp"

#include <algorithm>

namespace Ilum::geometry
{
BoundingBox::BoundingBox(const glm::vec3 &_min, const glm::vec3 &_max) :
    min_(_min), max_(_max)
{
}

BoundingBox::operator bool() const
{
	return min_.x < max_.x && min_.y < max_.y && min_.z < max_.z;
}

void BoundingBox::merge(const glm::vec3 &point)
{
	max_ = glm::max(max_, point);
	min_ = glm::min(min_, point);
}

void BoundingBox::merge(const std::vector<glm::vec3> &points)
{
	std::for_each(points.begin(), points.end(), [this](const glm::vec3 &p) { merge(p); });
}

void BoundingBox::merge(const BoundingBox &bounding_box)
{
	min_.x = bounding_box.min_.x < min_.x ? bounding_box.min_.x : min_.x;
	min_.y = bounding_box.min_.y < min_.y ? bounding_box.min_.y : min_.y;
	min_.z = bounding_box.min_.z < min_.z ? bounding_box.min_.z : min_.z;
	max_.x = bounding_box.max_.x > max_.x ? bounding_box.max_.x : max_.x;
	max_.y = bounding_box.max_.y > max_.y ? bounding_box.max_.y : max_.y;
	max_.z = bounding_box.max_.z > max_.z ? bounding_box.max_.z : max_.z;
}

BoundingBox BoundingBox::transform(const glm::mat4 &trans) const
{
	glm::vec3 new_center = trans * glm::vec4(center(), 1.f);
	glm::vec3 old_edge   = scale() * 0.5f;
	glm::vec3 new_edge   = glm::vec3(
        std::fabs(trans[0][0]) * old_edge.x + std::fabs(trans[0][1]) * old_edge.y + std::fabs(trans[0][2]) * old_edge.z,
        std::fabs(trans[1][0]) * old_edge.x + std::fabs(trans[1][1]) * old_edge.y + std::fabs(trans[1][2]) * old_edge.z,
        std::fabs(trans[2][0]) * old_edge.x + std::fabs(trans[2][1]) * old_edge.y + std::fabs(trans[2][2]) * old_edge.z);

	return BoundingBox(new_center - new_edge, new_center + new_edge);
}

const glm::vec3 BoundingBox::center() const
{
	return (max_ + min_) / 2.f;
}

const glm::vec3 BoundingBox::scale() const
{
	return max_ - min_;
}

bool BoundingBox::isInside(const glm::vec3 &point) const
{
	return !(point.x < min_.x || point.x > max_.x || point.y < min_.y || point.y > max_.y || point.z < min_.z || point.z > max_.z);
}

bool BoundingBox::valid() const
{
	return min_.x <= max_.x && min_.y <= max_.y && min_.z <= max_.z;
}
}        // namespace Ilum::geometry