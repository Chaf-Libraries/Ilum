/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#pragma once

#include "Matrix3.h"
#include "Matrix4.h"
#include "Vector2.h"
#include "Vector3.h"
#include "Vector4.h"
#include "Quaternion.h"

namespace Math
{
class Transform
{
  public:
	// Translation
	static Matrix3 translate(const Vector2 &translation);
	static Matrix4 translate(const Vector3 &translation);
	static Vector2 translate(const Vector2 &origin, const Vector2 &translation);
	static Vector3 translate(const Vector3 &origin, const Vector3 &translation);

	// Rotation
	static Matrix3 rotate(float angle);
	static Vector2 rotate(const Vector2 &origin, float angle);

	// Rotation on axis
	static Matrix4 rotateX(float angle);
	static Matrix4 rotateY(float angle);
	static Matrix4 rotateZ(float angle);

	// Scale
	static Matrix3 scale(const Vector2 &scaling);
	static Matrix4 scale(const Vector3 &scaling);
	static Vector2 scale(const Vector2 &origin, const Vector2 &scaling);
	static Vector3 scale(const Vector3 &origin, const Vector3 &scaling);

	// Projection
	static Matrix4 perspective(float fov, float aspect, float near, float far);
	static Matrix4 orthographic(float size, float aspect, float near, float far);

	// Look at
	static Matrix4 lookAt(const Vector3 &eye, const Vector3 &target, const Vector3 &up);

	// Decompose from transform to translation + rotation + scale
	static std::tuple<Math::Vector3, Math::Quaternion, Math::Vector3> decompose(const Math::Matrix4 &transform);
};
}        // namespace Math