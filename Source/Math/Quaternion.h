/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

namespace Ilum
{
class Matrix3;
class Matrix4;
class Vector3;
class Vector4;

class Quaternion
{
  public:
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	float w = 0.f;

  public:
	Quaternion() = default;

	Quaternion(float x, float y, float z, float w);

	Quaternion(const Matrix4 &m);

	Quaternion(const Matrix3 &m);

	// Degree
	Quaternion(const Vector3 &euler);

	static Quaternion identity();

	Quaternion operator-() const;
	Quaternion operator+(const Quaternion &q) const;
	Quaternion operator-(const Quaternion &q) const;
	Quaternion operator*(float t) const;
	Quaternion operator*(const Quaternion &q) const;
	Matrix3    operator*(const Matrix3 &m) const;
	Vector3    operator*(const Vector3 &v) const;
	Quaternion operator/(float t) const;
	Quaternion operator|(const Quaternion &q) const;

	Quaternion &operator+=(const Quaternion &q);
	Quaternion &operator-=(const Quaternion &q);
	Quaternion &operator*=(float t);
	Quaternion &operator*=(const Quaternion &q);
	Quaternion &operator/=(float t);
	Quaternion &operator|=(const Quaternion &q);

	bool operator==(const Quaternion &q) const;
	bool operator!=(const Quaternion &q) const;

	float      dot(const Quaternion &q) const;
	Quaternion normalize() const;
	float      length() const;
	float      lengthSquare() const;
	float      angle() const;
	Vector3    rotationAxis() const;
	Quaternion inverse() const;
	Quaternion conjugate() const;
	Quaternion square() const;
	Vector3    rotatePoint(const Vector3 &point) const;
	Vector3    rotatePoint(const Vector3 &point, const Vector3 &pivot) const;

	// degree
	Vector3    eulerAngles() const;
	Matrix3    toMatrix3() const;
	Matrix4    toMatrix4() const;

	static float      angularDistance(const Quaternion &q1, const Quaternion &q2);
	static Quaternion lerp(const Quaternion &start, const Quaternion &end, float alpha);
	static Quaternion slerp(const Quaternion &start, const Quaternion &end, float alpha);
	static Quaternion nlerp(const Quaternion &start, const Quaternion &end, float alpha);
};
}        // namespace Ilum