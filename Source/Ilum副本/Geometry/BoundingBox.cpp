#include "BoundingBox.hpp"

#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

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
	glm::vec3 v[2], xa, xb, ya, yb, za, zb;

	xa = trans[0] * min_[0];
	xb = trans[0] * max_[0];

	ya = trans[1] * min_[1];
	yb = trans[1] * max_[1];

	za = trans[2] * min_[2];
	zb = trans[2] * max_[2];

	v[0] = trans[3];
	v[0] += glm::min(xa, xb);
	v[0] += glm::min(ya, yb);
	v[0] += glm::min(za, zb);

	v[1] = trans[3];
	v[1] += glm::max(xa, xb);
	v[1] += glm::max(ya, yb);
	v[1] += glm::max(za, zb);

	return BoundingBox(v[0], v[1]);
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