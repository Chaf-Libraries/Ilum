/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Vector4.h"
#include "Matrix4.h"
#include "Utility.h"

namespace Ilum
{
Vector4::Vector4(float x, float y, float z, float w) :
    x(x), y(y), z(z), w(w)
{
}

Vector4::Vector4(float value) :
    x(value), y(value), z(value), w(value)
{
}

Vector4 Vector4::operator-() const
{
	return {-x, -y, -z, -w};
}

Vector4 Vector4::operator+(const Vector4 &v) const
{
	return {x + v.x, y + v.y, z + v.z, w + v.w};
}

Vector4 Vector4::operator-(const Vector4 &v) const
{
	return *this + (-v);
}

Vector4 Vector4::operator|(const Vector4 &v) const
{
	return {x * v.x, y * v.y, z * v.z, w * v.w};
}

Vector4 Vector4::operator*(float t) const
{
	return {x * t, y * t, z * t, w * t};
}

Vector4 Vector4::operator*(const Matrix4 &m) const
{
	return {
	    m[0] * x + m[4] * y + m[8] * z + m[12] * w,
	    m[1] * x + m[5] * y + m[9] * z + m[13] * w,
	    m[2] * x + m[6] * y + m[10] * z + m[14] * w,
	    m[3] * x + m[7] * y + m[11] * z + m[15] * w};
}

Vector4 Vector4::operator/(float t) const
{
	return *this * (1 / t);
}

Vector4 &Vector4::operator=(const Vector4 &v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	w = v.w;
	return *this;
}

Vector4 &Vector4::operator+=(const Vector4 &v)
{
	*this = *this + v;
	return *this;
}

Vector4 &Vector4::operator-=(const Vector4 &v)
{
	*this = *this - v;
	return *this;
}

Vector4 &Vector4::operator|=(const Vector4 &v)
{
	*this = *this | v;
	return *this;
}

Vector4 &Vector4::operator*=(float t)
{
	*this = *this * t;
	return *this;
}

Vector4 &Vector4::operator*=(const Matrix4 &m)
{
	*this = *this * m;
	return *this;
}

Vector4 &Vector4::operator/=(float t)
{
	*this = *this / t;
	return *this;
}

bool Vector4::operator==(const Vector4 &v) const
{
	return x == v.x && y == v.y && z == v.z && w == v.w;
}

bool Vector4::operator!=(const Vector4 &v) const
{
	return !(v == *this);
}

Vector4 Vector4::normalize() const
{
	return *this / norm();
}

Vector4 Vector4::clamp(float min, float max) const
{
	return {
	    std::clamp(x, min, max),
	    std::clamp(y, min, max),
	    std::clamp(z, min, max),
	    std::clamp(w, min, max)};
}

float Vector4::norm2() const
{
	return dot(*this);
}

float Vector4::norm() const
{
	return sqrtf(norm2());
}

float Vector4::dot(const Vector4 &v) const
{
	return x * v.x + y * v.y + z * v.z + w * v.w;
}

Vector4 Vector4::hadamard(const Vector4 &v) const
{
	return {x * v.x, y * v.y, z * v.z, w * v.w};
}

bool Vector4::isParallelWith(const Vector4 &v) const
{
	return v.x / x == v.y / y && v.y / y == v.z / z && v.z / z == v.w / w;
}
bool Vector4::isVerticalWith(const Vector4 &v) const
{
	return dot(v) == 0.f;
}
Vector4 Vector4::lerp(const Vector4 &start, const Vector4 &end, float alpha)
{
	alpha = std::clamp(alpha, 0.f, 1.f);
	return start * alpha + end * (1 - alpha);
}
float Vector4::angle(const Vector4 &from, const Vector4 &to)
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
}        // namespace Ilum