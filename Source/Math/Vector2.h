/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

namespace Ilum
{
class Vector2
{
  public:
	float x = 0.f;
	float y = 0.f;

  public:
	Vector2() = default;
	Vector2(float x, float y);
	Vector2(float value);

	Vector2 operator-() const;
	Vector2 operator+(const Vector2 &v) const;
	Vector2 operator-(const Vector2 &v) const;
	Vector2 operator|(const Vector2 &v) const;        // Hadamard Product
	Vector2 operator*(float t) const;
	Vector2 operator/(float t) const;

	Vector2 &operator=(const Vector2 &v);
	Vector2 &operator+=(const Vector2 &v);
	Vector2 &operator-=(const Vector2 &v);
	Vector2 &operator|=(const Vector2 &v);        // Hadamard Product
	Vector2 &operator*=(float t);
	Vector2 &operator/=(float t);

	bool operator==(const Vector2 &v) const;
	bool operator!=(const Vector2 &v) const;

	Vector2 normalize() const;
	Vector2 clamp(float min, float max) const;
	float   norm2() const;
	float   norm() const;
	float   dot(const Vector2 &v) const;
	float   cross(const Vector2 &v) const;
	bool    isParallelWith(const Vector2 &v) const;
	bool    isVerticalWith(const Vector2 &v) const;

	static Vector2 lerp(const Vector2 &start, const Vector2 &end, float alpha);
	static float   angle(const Vector2 &from, const Vector2 &to);
};
}        // namespace Ilum