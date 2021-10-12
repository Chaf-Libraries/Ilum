/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

namespace Ilum
{
class Matrix3;

class Vector3
{
  public:
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;

  public:
	Vector3() = default;
	Vector3(float x, float y, float z);
	Vector3(float value);

	static Vector3 One;
	static Vector3 Zero;
	static Vector3 Forward;
	static Vector3 Right;
	static Vector3 Up;

	Vector3 operator-() const;
	Vector3 operator+(const Vector3 &v) const;
	Vector3 operator-(const Vector3 &v) const;
	Vector3 operator|(const Vector3 &v) const;        // Hadamard Product
	Vector3 operator*(float t) const;
	Vector3 operator*(const Matrix3 &m) const;
	Vector3 operator/(float t) const;

	Vector3 &operator=(const Vector3 &v);
	Vector3 &operator+=(const Vector3 &v);
	Vector3 &operator-=(const Vector3 &v);
	Vector3 &operator|=(const Vector3 &v);        // Hadamard Product
	Vector3 &operator*=(float t);
	Vector3 &operator*=(const Matrix3 &m);
	Vector3 &operator/=(float t);

	bool operator==(const Vector3 &v) const;
	bool operator!=(const Vector3 &v) const;

	Vector3 normalize() const;
	Vector3 clamp(float min, float max) const;
	float   norm2() const;
	float   norm() const;
	float   dot(const Vector3 &v) const;
	Vector3 cross(const Vector3 &v) const;
	bool    isParallelWith(const Vector3 &v) const;
	bool    isVerticalWith(const Vector3 &v) const;

	static Vector3 lerp(const Vector3 &start, const Vector3 &end, float alpha);
	static float   angle(const Vector3 &from, const Vector3 &to);
};

using Point3 = Vector3;
using Rgb    = Vector3;
}        // namespace Ilum