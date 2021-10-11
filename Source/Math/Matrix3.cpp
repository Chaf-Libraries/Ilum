/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Matrix3.h"
#include "Vector3.h"

namespace Ilum
{
Matrix3::Matrix3(
    float a11, float a12, float a13,
    float a21, float a22, float a23,
    float a31, float a32, float a33) :
    data{a11, a21, a31, a12, a22, a32, a13, a23, a33}
{
}

Matrix3::Matrix3(float value) :
    data{value, value, value, value, value, value, value, value, value}
{
}

Matrix3 Matrix3::identity()
{
	return {
	    1.f, 0.f, 0.f,
	    0.f, 1.f, 0.f,
	    0.f, 0.f, 1.f};
}

float Matrix3::operator()(uint32_t x, uint32_t y) const
{
	assert(x < 3 && x >= 0 && y < 3 && y >= 0);
	return data[3 * y + x];
}

float &Matrix3::operator()(uint32_t x, uint32_t y)
{
	assert(x < 3 && x >= 0 && y < 3 && y >= 0);
	return data[3 * y + x];
}

float Matrix3::operator[](uint32_t idx) const
{
	return data[idx];
}

float &Matrix3::operator[](uint32_t idx)
{
	return data[idx];
}

Matrix3 Matrix3::operator-() const
{
	return {
	    -data[0], -data[3], -data[6],
	    -data[1], -data[4], -data[7],
	    -data[2], -data[5], -data[8]};
}

Matrix3 Matrix3::operator+(const Matrix3 &m) const
{
	return {
	    data[0] + m.data[0],
	    data[3] + m.data[3],
	    data[6] + m.data[6],

	    data[1] + m.data[1],
	    data[4] + m.data[4],
	    data[7] + m.data[7],

	    data[2] + m.data[2],
	    data[5] + m.data[5],
	    data[8] + m.data[8]};
}

Matrix3 Matrix3::operator-(const Matrix3 &m) const
{
	return *this + (-m);
}

Matrix3 Matrix3::operator*(float t) const
{
	return {
	    t * data[0], t * data[3], t * data[6],
	    t * data[1], t * data[4], t * data[7],
	    t * data[2], t * data[5], t * data[8]};
}

Vector3 Matrix3::operator*(const Vector3 &v) const
{
	return {
	    data[0] * v.x + data[3] * v.y + data[6] * v.z,
	    data[1] * v.x + data[4] * v.y + data[7] * v.z,
	    data[2] * v.x + data[5] * v.y + data[8] * v.z};
}

Matrix3 Matrix3::operator*(const Matrix3 &m) const
{
	return {
	    data[0] * m.data[0] + data[3] * m.data[1] + data[6] * m.data[2],
	    data[0] * m.data[3] + data[3] * m.data[4] + data[6] * m.data[5],
	    data[0] * m.data[6] + data[3] * m.data[7] + data[6] * m.data[8],

	    data[1] * m.data[0] + data[4] * m.data[1] + data[7] * m.data[2],
	    data[1] * m.data[3] + data[4] * m.data[4] + data[7] * m.data[5],
	    data[1] * m.data[6] + data[4] * m.data[7] + data[7] * m.data[8],

	    data[2] * m.data[0] + data[5] * m.data[1] + data[8] * m.data[2],
	    data[2] * m.data[3] + data[5] * m.data[4] + data[8] * m.data[5],
	    data[2] * m.data[6] + data[5] * m.data[7] + data[8] * m.data[8],
	};
}

Matrix3 Matrix3::operator|(const Matrix3 &m) const
{
	return {
	    data[0] * m.data[0], data[3] * m.data[3], data[6] * m.data[6],
	    data[1] * m.data[1], data[4] * m.data[4], data[7] * m.data[7],
	    data[2] * m.data[2], data[5] * m.data[5], data[8] * m.data[8]};
}

Matrix3 Matrix3::operator/(float t) const
{
	return *this * (1 / t);
}

Matrix3 &Matrix3::operator=(const Matrix3 &m)
{
	memcpy(data, m.data, 9 * sizeof(float));
	return *this;
}

Matrix3 &Matrix3::operator+=(const Matrix3 &m)
{
	*this = *this + m;
	return *this;
}

Matrix3 &Matrix3::operator-=(const Matrix3 &m)
{
	*this = *this - m;
	return *this;
}

Matrix3 &Matrix3::operator*=(float t)
{
	*this = (*this) * t;
	return *this;
}

Matrix3 &Matrix3::operator*=(const Matrix3 &m)
{
	*this = *this * m;
	return *this;
}

Matrix3 &Matrix3::operator|=(const Matrix3 &m)
{
	*this = *this | m;
	return *this;
}

Matrix3 &Matrix3::operator/=(float t)
{
	*this = (*this) / t;
	return *this;
}

bool Matrix3::operator==(const Matrix3 &m) const
{
	return memcmp(data, m.data, 9 * sizeof(float));
}

bool Matrix3::operator!=(const Matrix3 &m) const
{
	return !(*this == m);
}

float Matrix3::det() const
{
	return data[0] * (data[4] * data[8] - data[5] * data[7]) +
	       data[1] * (data[5] * data[6] - data[3] * data[8]) +
	       data[2] * (data[3] * data[7] - data[4] * data[6]);
}

Matrix3 Matrix3::adjugate() const
{
	return {
	    data[4] * data[8] - data[5] * data[7],
	    -(data[3] * data[8] - data[5] * data[6]),
	    data[3] * data[7] - data[4] * data[6],

	    -(data[1] * data[8] - data[2] * data[7]),
	    data[0] * data[8] - data[2] * data[6],
	    -(data[0] * data[7] - data[1] * data[6]),

	    data[1] * data[5] - data[2] * data[4],
	    -(data[0] * data[5] - data[2] * data[3]),
	    data[0] * data[4] - data[1] * data[3]};
}

Matrix3 Matrix3::inverse() const
{
	return adjugate() / det();
}

Matrix3 Matrix3::transpose() const
{
	return {
	    data[0], data[1], data[2],
	    data[3], data[4], data[5],
	    data[6], data[7], data[8]};
}

Matrix3 Matrix3::lerp(const Matrix3 &start, const Matrix3 &end, float alpha)
{
	alpha = std::clamp(alpha, 0.f, 1.f);
	return start * alpha + end * (1 - alpha);
}

std::ostream &operator<<(std::ostream &output, const Matrix3 &m)
{
	output << "3x3 Matrix:" << std::endl;
	for (uint32_t i = 0; i < 3; i++)
	{
		for (uint32_t j = 0; j < 3; j++)
		{
			output << m(i, j) << " ";
		}
		output << std::endl;
	}

	return output;
}
}        // namespace Ilum