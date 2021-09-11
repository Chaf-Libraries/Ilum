/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Vector2.h"
#include "Utility.h"

namespace Math
{
Vector2::Vector2(float x, float y) :
    x(x), y(y)
{
}

Vector2::Vector2(float value) :
    x(value), y(value)
{
}

Vector2 Vector2::operator-() const
{
	return {-x, -y};
}

Vector2 Vector2::operator+(const Vector2 &v) const
{
	return {x + v.x, y + v.y};
}

Vector2 Vector2::operator-(const Vector2 &v) const
{
	return *this + (-v);
}

Vector2 Vector2::operator|(const Vector2 &v) const
{
	return {x * v.x, y * v.y};
}

Vector2 Vector2::operator*(float t) const
{
	return {t * x, t * y};
}

Vector2 Vector2::operator/(float t) const
{
	return {x / t, y / t};
}

Vector2 &Vector2::operator=(const Vector2 &v)
{
	x = v.x;
	y = v.y;
	return *this;
}

Vector2 &Vector2::operator+=(const Vector2 &v)
{
	*this = *this + v;
	return *this;
}

Vector2 &Vector2::operator-=(const Vector2 &v)
{
	*this = *this - v;
	return *this;
}

Vector2 &Vector2::operator|=(const Vector2 &v)
{
	*this = *this | v;
	return *this;
}

Vector2 &Vector2::operator*=(float t)
{
	*this = *this * t;
	return *this;
}

Vector2 &Vector2::operator/=(float t)
{
	*this = *this / t;
	return *this;
}

bool Vector2::operator==(const Vector2 &v) const
{
	return x == v.x && y == v.y;
}

bool Vector2::operator!=(const Vector2 &v) const
{
	return !(*this == v);
}

Vector2 Vector2::normalize() const
{
	return *this / norm();
}

Vector2 Vector2::clamp(float min, float max) const
{
	return {std::clamp(x, min, max), std::clamp(y, min, max)};
}

float Vector2::norm2() const
{
	return x * x + y * y;
}

float Vector2::norm() const
{
	return sqrtf(norm2());
}

float Vector2::dot(const Vector2 &v) const
{
	return x * v.x + y * v.y;
}

float Vector2::cross(const Vector2 &v) const
{
	return x * v.y + y * v.x;
}

bool Vector2::isParallelWith(const Vector2 &v) const
{
	return fabs(cross(v)) < EPSILON;
}

bool Vector2::isVerticalWith(const Vector2 &v) const
{
	return fabs(dot(v)) < EPSILON;
}

Vector2 Vector2::lerp(const Vector2 &start, const Vector2 &end, float alpha)
{
	return start * alpha + end * (1 - alpha);
}

float Vector2::angle(const Vector2 &from, const Vector2 &to)
{
	return acosf(from.dot(to) / (from.norm() * to.norm()));
}
}        // namespace Math
