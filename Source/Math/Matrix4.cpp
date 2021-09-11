/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Matrix4.h"
#include "Matrix3.h"
#include "Vector4.h"

namespace Math
{
Matrix4 Matrix4::Identity = {
    1.f, 0.f, 0.f, 0.f,
    0.f, 1.f, 0.f, 0.f,
    0.f, 0.f, 1.f, 0.f,
    0.f, 0.f, 0.f, 1.f};

Matrix4::Matrix4(
    float a11, float a12, float a13, float a14,
    float a21, float a22, float a23, float a24,
    float a31, float a32, float a33, float a34,
    float a41, float a42, float a43, float a44) :
    data{a11, a21, a31, a41, a12, a22, a32, a42, a13, a23, a33, a43, a14, a24, a34, a44}
{
}

Matrix4::Matrix4(float value) :
    data{value, value, value, value, value, value, value, value, value, value, value, value, value, value, value, value}
{
}

float Matrix4::operator()(uint32_t x, uint32_t y) const
{
	assert(x >= 0 && y >= 0 && x < 4 && y < 4);
	return data[y * 4 + x];
}

float &Matrix4::operator()(uint32_t x, uint32_t y)
{
	assert(x >= 0 && y >= 0 && x < 4 && y < 4);
	return data[y * 4 + x];
}

float Matrix4::operator[](uint32_t idx) const
{
	return data[idx];
}

float &Matrix4::operator[](uint32_t idx)
{
	return data[idx];
}

Matrix4 Matrix4::operator-() const
{
	return {
	    -data[0], -data[4], -data[8], -data[12],
	    -data[1], -data[5], -data[9], -data[13],
	    -data[2], -data[6], -data[10], -data[14],
	    -data[3], -data[7], -data[11], -data[15]};
}

Matrix4 Matrix4::operator+(const Matrix4 &m) const
{
	return {
	    data[0] + m.data[0], data[4] + m.data[4], data[8] + m.data[8], data[12] + m.data[12],
	    data[1] + m.data[1], data[5] + m.data[5], data[9] + m.data[9], data[13] + m.data[13],
	    data[2] + m.data[2], data[6] + m.data[6], data[10] + m.data[10], data[14] + m.data[14],
	    data[3] + m.data[3], data[7] + m.data[7], data[11] + m.data[11], data[15] + m.data[15]};
}

Matrix4 Matrix4::operator-(const Matrix4 &m) const
{
	return *this + (-m);
}

Matrix4 Matrix4::operator*(float t) const
{
	return {
	    t * data[0], t * data[4], t * data[8], t * data[12],
	    t * data[1], t * data[5], t * data[9], t * data[13],
	    t * data[2], t * data[6], t * data[10], t * data[14],
	    t * data[3], t * data[7], t * data[11], t * data[15]};
}

Vector4 Matrix4::operator*(const Vector4 &v) const
{
	return {
	    data[0] * v.x + data[4] * v.y + data[8] * v.z + data[12] * v.w,
	    data[1] * v.x + data[5] * v.y + data[9] * v.z + data[13] * v.w,
	    data[2] * v.x + data[6] * v.y + data[10] * v.z + data[14] * v.w,
	    data[3] * v.x + data[7] * v.y + data[11] * v.z + data[15] * v.w};
}

Matrix4 Matrix4::operator*(const Matrix4 &m) const
{
	return {
	    data[0] * m.data[0] + data[4] * m.data[1] + data[8] * m.data[2] + data[12] * m.data[3],
	    data[0] * m.data[4] + data[4] * m.data[5] + data[8] * m.data[6] + data[12] * m.data[7],
	    data[0] * m.data[8] + data[4] * m.data[9] + data[8] * m.data[10] + data[12] * m.data[11],
	    data[0] * m.data[12] + data[4] * m.data[13] + data[8] * m.data[14] + data[12] * m.data[15],

	    data[1] * m.data[0] + data[5] * m.data[1] + data[9] * m.data[2] + data[13] * m.data[3],
	    data[1] * m.data[4] + data[5] * m.data[5] + data[9] * m.data[6] + data[13] * m.data[7],
	    data[1] * m.data[8] + data[5] * m.data[9] + data[9] * m.data[10] + data[13] * m.data[11],
	    data[1] * m.data[12] + data[5] * m.data[13] + data[9] * m.data[14] + data[13] * m.data[15],

	    data[2] * m.data[0] + data[6] * m.data[1] + data[10] * m.data[2] + data[14] * m.data[3],
	    data[2] * m.data[4] + data[6] * m.data[5] + data[10] * m.data[6] + data[14] * m.data[7],
	    data[2] * m.data[8] + data[6] * m.data[9] + data[10] * m.data[10] + data[14] * m.data[11],
	    data[2] * m.data[12] + data[6] * m.data[13] + data[10] * m.data[14] + data[14] * m.data[15],

	    data[3] * m.data[0] + data[7] * m.data[1] + data[11] * m.data[2] + data[15] * m.data[3],
	    data[3] * m.data[4] + data[7] * m.data[5] + data[11] * m.data[6] + data[15] * m.data[7],
	    data[3] * m.data[8] + data[7] * m.data[9] + data[11] * m.data[10] + data[15] * m.data[11],
	    data[3] * m.data[12] + data[7] * m.data[13] + data[11] * m.data[14] + data[15] * m.data[15],
	};
}

Matrix4 Matrix4::operator|(const Matrix4 &m) const
{
	return {
	    data[0] * m.data[0], data[4] * m.data[4], data[8] * m.data[8], data[12] * m.data[12],
	    data[1] * m.data[1], data[5] * m.data[5], data[9] * m.data[9], data[13] * m.data[13],
	    data[2] * m.data[2], data[6] * m.data[6], data[10] * m.data[10], data[14] * m.data[14],
	    data[3] * m.data[3], data[7] * m.data[7], data[11] * m.data[11], data[15] * m.data[15]};
}

Matrix4 Matrix4::operator/(float t) const
{
	return *this * (1 / t);
}

Matrix4 &Matrix4::operator=(const Matrix4 &m)
{
	memcpy(data, m.data, 16 * sizeof(float));
	return *this;
}

Matrix4 &Matrix4::operator+=(const Matrix4 &m)
{
	*this = *this + m;
	return *this;
}

Matrix4 &Matrix4::operator-=(const Matrix4 &m)
{
	*this = *this - m;
	return *this;
}

Matrix4 &Matrix4::operator*=(float t)
{
	*this = *this * t;
	return *this;
}

Matrix4 &Matrix4::operator*=(const Matrix4 &m)
{
	*this = *this * m;
	return *this;
}

Matrix4 &Matrix4::operator|=(const Matrix4 &m)
{
	*this = *this | m;
	return *this;
}

Matrix4 &Matrix4::operator/=(float t)
{
	*this = *this / t;
	return *this;
}

bool Matrix4::operator==(const Matrix4 &m) const
{
	return memcmp(data, m.data, 16 * sizeof(float));
}

bool Matrix4::operator!=(const Matrix4 &m) const
{
	return !(*this == m);
}

float Matrix4::det() const
{
	return data[0] * Matrix3(data[5], data[6], data[7], data[9], data[10], data[11], data[13], data[14], data[15]).det() -
	       data[1] * Matrix3(data[4], data[6], data[7], data[8], data[10], data[11], data[12], data[14], data[15]).det() +
	       data[2] * Matrix3(data[4], data[5], data[7], data[8], data[9], data[11], data[12], data[13], data[15]).det() -
	       data[3] * Matrix3(data[4], data[5], data[6], data[8], data[9], data[10], data[12], data[13], data[14]).det();
}

Matrix4 Matrix4::adjugate() const
{
	return {
	    Matrix3(data[5], data[9], data[13], data[6], data[10], data[14], data[7], data[11], data[15]).det(),
	    -Matrix3(data[4], data[8], data[12], data[6], data[10], data[14], data[7], data[11], data[15]).det(),
	    Matrix3(data[4], data[8], data[12], data[5], data[9], data[13], data[7], data[11], data[15]).det(),
	    -Matrix3(data[4], data[8], data[12], data[5], data[9], data[13], data[6], data[10], data[14]).det(),

	    -Matrix3(data[1], data[9], data[13], data[2], data[10], data[14], data[3], data[11], data[15]).det(),
	    Matrix3(data[0], data[8], data[12], data[2], data[10], data[14], data[3], data[11], data[15]).det(),
	    -Matrix3(data[0], data[8], data[12], data[1], data[9], data[13], data[3], data[7], data[15]).det(),
	    Matrix3(data[0], data[8], data[12], data[1], data[9], data[13], data[2], data[10], data[14]).det(),

	    Matrix3(data[1], data[5], data[13], data[2], data[6], data[14], data[3], data[7], data[15]).det(),
	    -Matrix3(data[0], data[4], data[12], data[2], data[6], data[14], data[3], data[7], data[15]).det(),
	    Matrix3(data[0], data[4], data[12], data[1], data[5], data[13], data[3], data[7], data[15]).det(),
	    -Matrix3(data[0], data[4], data[12], data[1], data[5], data[13], data[2], data[6], data[14]).det(),

	    -Matrix3(data[1], data[5], data[9], data[2], data[6], data[10], data[3], data[7], data[11]).det(),
	    Matrix3(data[0], data[4], data[8], data[2], data[6], data[10], data[3], data[7], data[11]).det(),
	    -Matrix3(data[0], data[4], data[8], data[1], data[5], data[9], data[3], data[7], data[11]).det(),
	    Matrix3(data[0], data[4], data[8], data[1], data[5], data[9], data[2], data[6], data[10]).det()};
}

Matrix4 Matrix4::inverse() const
{
	return adjugate() / det();
}

Matrix4 Matrix4::transpose() const
{
	return {
	    data[0], data[1], data[2], data[3],
	    data[4], data[5], data[6], data[7],
	    data[8], data[9], data[10], data[11],
	    data[12], data[13], data[14], data[15]};
}

Matrix4 Matrix4::lerp(const Matrix4 &start, const Matrix4 &end, float alpha)
{
	alpha = std::clamp(alpha, 0.f, 1.f);
	return start * alpha + end * (1 - alpha);
}

std::ostream &operator<<(std::ostream &output, const Matrix4 &m)
{
	output << "4x4 Matrix:" << std::endl;
	for (uint32_t i = 0; i < 4; i++)
	{
		for (uint32_t j = 0; j < 4; j++)
		{
			output << m(i, j) << " ";
		}
		output << std::endl;
	}

	return output;
}
}        // namespace Math