/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Vector3.h"
#include "Matrix3.h"
#include "Utility.h"

namespace Math
{
Vector3 Vector3::One     = {1.f, 1.f, 1.f};
Vector3 Vector3::Zero    = {0.f, 0.f, 0.f};
Vector3 Vector3::Forward = {0.f, 0.f, 1.f};
Vector3 Vector3::Right   = {1.f, 0.f, 0.f};
Vector3 Vector3::Up      = {0.f, 1.f, 0.f};

Vector3::Vector3(float x, float y, float z) :
    x(x), y(y), z(z)
{
}

Vector3::Vector3(float value) :
    x(value), y(value), z(value)
{
}

Vector3 Vector3::operator-() const
{
	return {-x, -y, -z};
}

Vector3 Vector3::operator+(const Vector3 &v) const
{
	return {x + v.x, y + v.y, z + v.z};
}

Vector3 Vector3::operator-(const Vector3 &v) const
{
	return *this + (-v);
}

Vector3 Vector3::operator|(const Vector3 &v) const
{
	return {x * v.x, y * v.y, z * v.z};
}

Vector3 Vector3::operator*(float t) const
{
	return {t * x, t * y, t * z};
}

Vector3 Vector3::operator*(const Matrix3 &m) const
{
	return {
	    m[0] * x + m[3] * y + m[6] * z,
	    m[1] * x + m[4] * y + m[7] * z,
	    m[2] * x + m[5] * y + m[8] * z};
}

Vector3 Vector3::operator/(float t) const
{
	return *this * (1 / t);
}

Vector3 &Vector3::operator=(const Vector3 &v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	return *this;
}

Vector3 &Vector3::operator+=(const Vector3 &v)
{
	*this = *this + v;
	return *this;
}

Vector3 &Vector3::operator-=(const Vector3 &v)
{
	*this = *this - v;
	return *this;
}

Vector3 &Vector3::operator|=(const Vector3 &v)
{
	*this = *this | v;
	return *this;
}

Vector3 &Vector3::operator*=(float t)
{
	*this = *this * t;
	return *this;
}

Vector3 &Vector3::operator*=(const Matrix3 &m)
{
	*this = *this * m;
	return *this;
}

Vector3 &Vector3::operator/=(float t)
{
	*this = *this / t;
	return *this;
}

bool Vector3::operator==(const Vector3 &v) const
{
	return x == v.x && y == v.y && z == v.z;
}

bool Vector3::operator!=(const Vector3 &v) const
{
	return !(v == *this);
}

float Vector3::norm2() const
{
	return dot(*this);
}

float Vector3::norm() const
{
	return sqrtf(norm2());
}

float Vector3::dot(const Vector3 &v) const
{
	return x * v.x + y * v.y + z * v.z;
}

Vector3 Vector3::cross(const Vector3 &v) const
{
	return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
}

bool Vector3::isParallelWith(const Vector3 &v) const
{
	return cross(v) == Vector3(0.f, 0.f, 0.f);
}

bool Vector3::isVerticalWith(const Vector3 &v) const
{
	return dot(v) == 0.f;
}

Vector3 Vector3::normalize() const
{
	return *this / norm();
}

Vector3 Vector3::clamp(float min, float max) const
{
	return {
	    std::clamp(x, min, max),
	    std::clamp(y, min, max),
	    std::clamp(z, min, max)};
}

Vector3 Vector3::lerp(const Vector3 &start, const Vector3 &end, float alpha)
{
	alpha = std::clamp(alpha, 0.f, 1.f);
	return start * alpha + end * (1 - alpha);
}

float Vector3::angle(const Vector3 &from, const Vector3 &to)
{
	float length_product = from.norm() * to.norm();

	if (length_product > 0.f)
	{
		float result = from.dot(to) / length_product;
		if (result > -1.f && result <= 1.f)
		{
			return acosf(result);
		}
	}

	return 0.0f;
}
}        // namespace Math