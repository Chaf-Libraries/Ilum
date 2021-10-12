/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

namespace Ilum
{
class Matrix4;

class Vector4
{
  public:
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	float w = 0.f;

  public:
	Vector4() = default;
	Vector4(float x, float y, float z, float w);
	Vector4(float value);

	Vector4 operator-() const;
	Vector4 operator+(const Vector4 &v) const;
	Vector4 operator-(const Vector4 &v) const;
	Vector4 operator|(const Vector4 &v) const;        // Hadamard Product
	Vector4 operator*(float t) const;
	Vector4 operator*(const Matrix4& m) const;
	Vector4 operator/(float t) const;

	Vector4 &operator=(const Vector4 &v);
	Vector4 &operator+=(const Vector4 &v);
	Vector4 &operator-=(const Vector4 &v);
	Vector4 &operator|=(const Vector4 &v);        // Hadamard Product
	Vector4 &operator*=(float t);
	Vector4  &operator*=(const Matrix4 &m);
	Vector4 &operator/=(float t);

	bool operator==(const Vector4 &v) const;
	bool operator!=(const Vector4 &v) const;

	Vector4 normalize() const;
	Vector4 clamp(float min, float max) const;
	float   norm2() const;
	float   norm() const;
	float   dot(const Vector4 &v) const;
	Vector4 hadamard(const Vector4 &v) const;
	bool    isParallelWith(const Vector4 &v) const;
	bool    isVerticalWith(const Vector4 &v) const;

	static Vector4 lerp(const Vector4 &start, const Vector4 &end, float alpha);
	static float   angle(const Vector4 &from, const Vector4 &to);
};

using Rgba = Vector4;
}        // namespace Ilum