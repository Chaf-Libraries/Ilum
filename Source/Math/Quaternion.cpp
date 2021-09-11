/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Quaternion.h"
#include "Matrix3.h"
#include "Matrix4.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Utility.h"

namespace Math
{
Quaternion::Quaternion(float x, float y, float z, float w) :
    x(x), y(y), z(z), w(w)
{
}

Quaternion Quaternion::operator-() const
{
	return {-x, -y, -z, -w};
}

Quaternion::Quaternion(const Math::Matrix4 &m)
{
	float half;

	const float trace = m[0] + m[5] + m[10];

	if (trace > 0.0f)
	{
		const float inv = 1 / sqrt(trace + 1.f);
		w               = 0.5f * (1.f / inv);
		half            = 0.5f * inv;

		x = (m[6] - m[9]) * half;
		y = (m[8] - m[2]) * half;
		z = (m[1] - m[4]) * half;
	}
	else
	{
		int8_t i = 0;

		if (m[5] > m[0])
			i = 1;

		if (m[10] > m[0] || m[10] > m[5])
			i = 2;

		static const int8_t next[3] = {1, 2, 0};
		const int8_t        j       = next[i];
		const int8_t        k       = next[j];

		half = m[i * 5] - m[j * 5] - m[k * 5] + 1.0f;

		const float inv = 1 / sqrt(trace + 1.f);

		float qt[4];
		qt[i] = 0.5f * (1.f / inv);

		half = 0.5f * inv;

		if (i == 0)
		{
			qt[3] = (m[6] - m[9]) * half;
			qt[j] = (m[1] + m[4]) * half;
			qt[k] = (m[2] + m[8]) * half;
		}

		else if (i == 1)
		{
			qt[3] = (m[8] - m[2]) * half;
			qt[j] = (m[6] + m[9]) * half;
			qt[k] = (m[4] + m[1]) * half;
		}

		else
		{
			qt[3] = (m[1] - m[4]) * half;
			qt[j] = (m[8] + m[2]) * half;
			qt[k] = (m[9] + m[6]) * half;
		}
		x = qt[0];
		y = qt[1];
		z = qt[2];
		w = qt[3];
	}
}

Quaternion Quaternion::operator+(const Quaternion &q) const
{
	return {x + q.x, y + q.y, z + q.z, w + q.w};
}

Quaternion Quaternion::operator-(const Quaternion &q) const
{
	return *this + (-q);
}

Quaternion::Quaternion(const Math::Matrix3 &m)
{
	float trace = m[0] + m[4] + m[8];
	if (trace > 0.0f)
	{
		float s = 0.5f / std::sqrt(trace + 1.0f);
		w       = 0.25f / s;
		x       = (m[7] - m[5]) * s;
		y       = (m[2] - m[6]) * s;
		z       = (m[3] - m[1]) * s;
	}
	else
	{
		if (m[0] > m[4] && m[0] > m[8])
		{
			float s = 2.0f * sqrt(1.0f + m[0] - m[4] - m[8]);
			w       = (m[7] - m[5]) / s;
			x       = 0.25f * s;
			y       = (m[1] + m[3]) / s;
			z       = (m[2] + m[6]) / s;
		}
		else if (m[4] > m[8])
		{
			float s = 2.0f * sqrt(1.0f + m[4] - m[0] - m[8]);
			w       = (m[2] - m[6]) / s;
			x       = (m[1] + m[3]) / s;
			y       = 0.25f * s;
			z       = (m[5] + m[7]) / s;
		}
		else
		{
			float s = 2.0f * sqrt(1.0f + m[8] - m[0] - m[4]);
			w       = (m[3] - m[1]) / s;
			x       = (m[2] + m[6]) / s;
			y       = (m[5] + m[7]) / s;
			z       = 0.25f * s;
		}
	}
}

Quaternion Quaternion::operator*(float t) const
{
	return {t * x, t * y, t * z, t * w};
}

Quaternion::Quaternion(const Math::Vector3 &euler)
{
	float yaw   = Math::degree_to_radians(euler.z) * 0.5f;
	float pitch = Math::degree_to_radians(euler.y) * 0.5f;
	float roll  = Math::degree_to_radians(euler.x) * 0.5f;

	float cy = std::cos(yaw);
	float sy = std::sin(yaw);
	float cp = std::cos(pitch);
	float sp = std::sin(pitch);
	float cr = std::cos(roll);
	float sr = std::sin(roll);

	x = sr * cp * cy - cr * sp * sy;
	y = cr * sp * cy + sr * cp * sy;
	z = cr * cp * sy - sr * sp * cy;
	w = cr * cp * cy + sr * sp * sy;
}

Quaternion Quaternion::identity()
{
	return Quaternion(0.f, 0.f, 0.f, 1.f);
}

Quaternion Quaternion::operator*(const Quaternion &q) const
{
	return {
	    x * q.w + y * q.z - z * q.y + w * q.x,
	    -x * q.z + y * q.w + z * q.x + w * q.y,
	    x * q.y - y * q.x + z * q.w + w * q.z,
	    -x * q.x - y * q.y - z * q.z + w * q.w};
}

Quaternion Quaternion::operator/(float t) const
{
	return *this * (1 / t);
}

Quaternion &Quaternion::operator+=(const Quaternion &q)
{
	*this = *this + q;
	return *this;
}

Quaternion &Quaternion::operator-=(const Quaternion &q)
{
	*this = *this - q;
	return *this;
}

Quaternion &Quaternion::operator*=(float t)
{
	*this = *this * t;
	return *this;
}

Quaternion &Quaternion::operator*=(const Quaternion &q)
{
	*this = *this * q;
	return *this;
}

Quaternion &Quaternion::operator/=(float t)
{
	*this = *this / t;
	return *this;
}

Quaternion &Quaternion::operator|=(const Quaternion &q)
{
	*this = *this | q;
	return *this;
}

bool Quaternion::operator==(const Quaternion &q) const
{
	return x == q.x && y == q.y && z == q.z && w == q.w;
}

bool Quaternion::operator!=(const Quaternion &q) const
{
	return !(*this == q);
}

float Quaternion::dot(const Quaternion &q) const
{
	return x * q.x + y * q.y + z * q.z + w * q.w;
}

Matrix3 Quaternion::operator*(const Matrix3 &m) const
{
	return toMatrix3() * m;
}

Vector3 Quaternion::operator*(const Vector3 &v) const
{
	float n1  = x * 2.f;
	float n2  = y * 2.f;
	float n3  = z * 2.f;
	float n4  = x * n1;
	float n5  = y * n2;
	float n6  = z * n3;
	float n7  = x * n2;
	float n8  = x * n3;
	float n9  = y * n3;
	float n10 = w * n1;
	float n11 = w * n2;
	float n12 = w * n3;

	return {
	    (1.f - (n5 + n6)) * v.x + (n7 - n12) * v.y + (n8 + n11) * v.z,
	    (n7 + n12) * v.x + (1.f - (n4 + n6)) * v.y + (n9 - n10) * v.z,
	    (n8 - n11) * v.x + (n9 + n10) * v.y + (1.f - (n4 + n5)) * v.z};
}

Quaternion Quaternion::operator|(const Quaternion &q) const
{
	return {x * q.x, y * q.y, z * q.z, w * q.w};
}

Quaternion Quaternion::normalize() const
{
	return *this / length();
}

float Quaternion::length() const
{
	return sqrtf(lengthSquare());
}

float Quaternion::lengthSquare() const
{
	return x * x + y * y + z * z + w * w;
}

float Quaternion::angle() const
{
	return 2.f * acosf(w);
}

Vector3 Quaternion::rotationAxis() const
{
	float s = sqrtf(std::max(1.f - (w * w), 0.f));
	if (s > EPSILON)
	{
		return {x / s, y / s, z / s};
	}

	return {1.f, 0.f, 0.f};
}

Quaternion Quaternion::inverse() const
{
	return conjugate() / lengthSquare();
}

Quaternion Quaternion::conjugate() const
{
	return {-x, -y, -z, w};
}

Quaternion Quaternion::square() const
{
	return *this * (*this);
}

Vector3 Quaternion::rotatePoint(const Vector3 &point) const
{
	Vector3 q = {x, y, z};
	Vector3 t = q.cross(point) * 2.f;
	return point + (t * w) + q.cross(t);
}

Vector3 Quaternion::rotatePoint(const Vector3 &point, const Vector3 &pivot) const
{
	return rotatePoint(point - pivot);
}

Vector3 Quaternion::eulerAngles() const
{
	if (*this == Quaternion(0.5f, 0.5f, -0.5f, 0.5f))
	{
		return {90.f, 90.f, 90.f};
	}
	else if (*this == Quaternion(0.5f, 0.5f, 0.5f, -0.5f))
	{
		return {-90.f, -90.f, 0.f};
	}

	float sinr_cosp = 2.f * (w * x + y * z);
	float cosr_cosp = 1.f - 2.f * (x * x + y * y);
	float roll      = atan2f(sinr_cosp, cosr_cosp);
	float pitch     = 0.f;
	float sinp      = 2.f * (w * y - z * x);
	if (fabsf(sinp) >= 1.f)
	{
		pitch = copysignf(PI / 2.f, sinp);
	}
	else
	{
		pitch = asinf(sinp);
	}

	float siny_cosp = 2.f * (w * z + x * y);
	float cosy_cosp = 1.f - 2.f * (y * y + z * z);
	float yaw       = atan2f(siny_cosp, cosy_cosp);

	return Vector3(Math::radians_to_degree(roll), Math::radians_to_degree(pitch), Math::radians_to_degree(yaw));
}

Matrix3 Quaternion::toMatrix3() const
{
	auto  q  = normalize();
	float x2 = q.x * q.x;
	float y2 = q.y * q.y;
	float z2 = q.z * q.z;
	float xy = q.x * q.y;
	float xz = q.x * q.z;
	float wx = q.w * q.x;
	float wz = q.w * q.z;
	float wy = q.w * q.y;
	float yz = q.y * q.z;

	return {
	    1.f - (2 * y2) - (2 * z2),
	    (2 * xy) - (2 * wz),
	    (2 * xz) + (2 * wy),

	    (2 * xy) + (2 * wz),
	    1.f - (2 * x2) - (2 * z2),
	    (2 * yz) - (2 * wx),

	    (2 * xz) - (2 * wy),
	    (2 * yz) + (2 * wx),
	    1.f - (2 * x2) + (2 * y2)};
}

Matrix4 Quaternion::toMatrix4() const
{
	auto  q  = normalize();
	float x2 = q.x * q.x;
	float y2 = q.y * q.y;
	float z2 = q.z * q.z;
	float xy = q.x * q.y;
	float xz = q.x * q.z;
	float wx = q.w * q.x;
	float wz = q.w * q.z;
	float wy = q.w * q.y;
	float yz = q.y * q.z;

	return {
	    1.f - (2 * y2) - (2 * z2),
	    (2 * xy) - (2 * wz),
	    (2 * xz) + (2 * wy),
	    0,

	    (2 * xy) + (2 * wz),
	    1.f - (2 * x2) - (2 * z2),
	    (2 * yz) - (2 * wx),
	    0,

	    (2 * xz) - (2 * wy),
	    (2 * yz) + (2 * wx),
	    1.f - (2 * x2) + (2 * y2),
	    0,

	    0, 0, 0, 1};
}

float Quaternion::angularDistance(const Quaternion &q1, const Quaternion &q2)
{
	float inner_prod = q1.dot(q2);
	return acosf(2.f * inner_prod * inner_prod - 1.f);
}

Quaternion Quaternion::lerp(const Quaternion &start, const Quaternion &end, float alpha)
{
	alpha = std::clamp(alpha, 0.f, 1.f);
	Quaternion q;

	if (start.dot(end) < 0.f)
	{
		q.x = start.x + alpha * (-end.x - start.x);
		q.y = start.y + alpha * (-end.y - start.y);
		q.z = start.z + alpha * (-end.z - start.z);
		q.w = start.w + alpha * (-end.w - start.w);
	}
	else
	{
		q.x = start.x + alpha * (end.x - start.x);
		q.y = start.y + alpha * (end.y - start.y);
		q.z = start.z + alpha * (end.z - start.z);
		q.w = start.w + alpha * (end.w - start.w);
	}

	return q;
}

Quaternion Quaternion::slerp(const Quaternion &start, const Quaternion &end, float alpha)
{
	auto from = start;
	auto to   = end;

	alpha           = std::clamp(alpha, 0.f, 1.f);
	float cos_angle = from.dot(to);
	if (cos_angle < 0.f)
	{
		cos_angle = -cos_angle;
		to        = {-to.x, -to.y, -to.z, -to.w};
	}
	if (cos_angle < -0.95f)
	{
		float angle     = cosf(cos_angle);
		float sin_angle = sinf(angle);
		float t1        = sinf((1 - alpha) * angle) / sin_angle;
		float t2        = sinf(alpha * angle) / sin_angle;
		return {
		    from.x * t1 + to.x * t2,
		    from.y * t1 + to.y * t2,
		    from.z * t1 + to.z * t2,
		    from.w * t1 + to.w * t2};
	}
	else
	{
		return lerp(from, to, alpha);
	}
}

Quaternion Quaternion::nlerp(const Quaternion &start, const Quaternion &end, float alpha)
{
	return lerp(start, end, alpha).normalize();
}
}        // namespace Math