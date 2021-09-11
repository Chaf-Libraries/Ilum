/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

#include "Utility.h"

namespace Math
{
class Vector4;

class Matrix4
{
  public:
	Matrix4() = default;

	Matrix4(
	    float a11, float a12, float a13, float a14,
	    float a21, float a22, float a23, float a24,
	    float a31, float a32, float a33, float a34,
	    float a41, float a42, float a43, float a44);

	Matrix4(float value);

	static Matrix4 Identity;

	float  operator()(uint32_t x, uint32_t y) const;
	float &operator()(uint32_t x, uint32_t y);
	float  operator[](uint32_t idx) const;
	float &operator[](uint32_t idx);

	Matrix4 operator-() const;
	Matrix4 operator+(const Matrix4 &m) const;
	Matrix4 operator-(const Matrix4 &m) const;
	Matrix4 operator*(float t) const;
	Vector4 operator*(const Vector4 &v) const;
	Matrix4 operator*(const Matrix4 &m) const;
	Matrix4 operator|(const Matrix4 &m) const;
	Matrix4 operator/(float t) const;

	Matrix4 &operator=(const Matrix4 &m);
	Matrix4 &operator+=(const Matrix4 &m);
	Matrix4 &operator-=(const Matrix4 &m);
	Matrix4 &operator*=(float t);
	Matrix4 &operator*=(const Matrix4 &m);
	Matrix4 &operator|=(const Matrix4 &m);
	Matrix4 &operator/=(float t);

	bool operator==(const Matrix4 &m) const;
	bool operator!=(const Matrix4 &m) const;

	friend std::ostream &operator<<(std::ostream &output, const Matrix4 &m);

	float   det() const;
	Matrix4 adjugate() const;
	Matrix4 inverse() const;
	Matrix4 transpose() const;

	static Matrix4 lerp(const Matrix4 &start, const Matrix4 &end, float alpha);

  private:
	float data[16] = {0.f};
};
}        // namespace Math