/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

#include "Utility.h"

namespace Ilum
{
class Vector3;

class Matrix3
{
  public:
	Matrix3() = default;

	Matrix3(
	    float a11, float a12, float a13,
	    float a21, float a22, float a23,
	    float a31, float a32, float a33);

	Matrix3(float value);

	static Matrix3 identity();

	float  operator()(uint32_t x, uint32_t y) const;
	float &operator()(uint32_t x, uint32_t y);
	float  operator[](uint32_t idx) const;
	float &operator[](uint32_t idx);

	Matrix3 operator-() const;
	Matrix3 operator+(const Matrix3 &m) const;
	Matrix3 operator-(const Matrix3 &m) const;
	Matrix3 operator*(float t) const;
	Vector3 operator*(const Vector3 &v) const;
	Matrix3 operator*(const Matrix3 &m) const;
	Matrix3 operator|(const Matrix3 &m) const;
	Matrix3 operator/(float t) const;

	Matrix3 &operator=(const Matrix3 &m);
	Matrix3 &operator+=(const Matrix3 &m);
	Matrix3 &operator-=(const Matrix3 &m);
	Matrix3 &operator*=(float t);
	Matrix3 &operator*=(const Matrix3 &m);
	Matrix3 &operator|=(const Matrix3 &m);
	Matrix3 &operator/=(float t);

	bool operator==(const Matrix3 &m) const;
	bool operator!=(const Matrix3 &m) const;

	friend std::ostream &operator<<(std::ostream& output, const Matrix3 &m);

	float   det() const;
	Matrix3 adjugate() const;
	Matrix3 inverse() const;
	Matrix3 transpose() const;

	static Matrix3 lerp(const Matrix3 &start, const Matrix3 &end, float alpha);

  private:
	float data[9] = {0.f};
};
}        // namespace Ilum