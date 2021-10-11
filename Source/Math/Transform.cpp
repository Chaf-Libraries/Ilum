/**
* @project: IlumEngine
* @author: Chaphlagical.
* @licence: MIT
*/

#include "Transform.h"
#include "Utility.h"

namespace Ilum
{
Matrix3 Transform::translate(const Vector2 &translation)
{
	return {
	    1.f, 0.f, translation.x,
	    0.f, 1.f, translation.y,
	    0.f, 0.f, 1.f};
}

Matrix4 Transform::translate(const Vector3 &translation)
{
	return {
	    1.f, 0.f, 0.f, translation.x,
	    0.f, 1.f, 0.f, translation.y,
	    0.f, 0.f, 1.f, translation.z,
	    0.f, 0.f, 0.f, 1.f};
}

Vector2 Transform::translate(const Vector2 &origin, const Vector2 &translation)
{
	return origin + translation;
}

Vector3 Transform::translate(const Vector3 &origin, const Vector3 &translation)
{
	return origin + translation;
}

Matrix3 Transform::rotate(float angle)
{
	return {
	    cosf(angle), -sinf(angle), 0,
	    sinf(angle), cosf(angle), 0,
	    0, 0, 1};
}

Vector2 Transform::rotate(const Vector2 &origin, float angle)
{
	auto res = rotate(angle) * Vector3(origin.x, origin.y, 1.f);
	return {res.x, res.y};
}

Matrix4 Transform::rotateX(float angle)
{
	return {
	    1, 0, 0, 0,
	    0, cosf(angle), -sinf(angle), 0,
	    0, sinf(angle), cosf(angle), 0,
	    0, 0, 0, 1};
}

Matrix4 Transform::rotateY(float angle)
{
	return {
	    cosf(angle), 0, -sinf(angle), 0,
	    0, 1, 0, 0,
	    sinf(angle), 0, cosf(angle), 0,
	    0, 0, 0, 1};
}

Matrix4 Transform::rotateZ(float angle)
{
	return {
	    cosf(angle), -sin(angle), 0, 0,
	    sinf(angle), cosf(angle), 0, 0,
	    0, 0, 1, 0,
	    0, 0, 0, 1};
}

Matrix3 Transform::scale(const Vector2 &scaling)
{
	return {
	    scaling.x, 0, 0,
	    0, scaling.y, 0,
	    0, 0, 1};
}

Matrix4 Transform::scale(const Vector3 &scaling)
{
	return {
	    scaling.x, 0, 0, 0,
	    0, scaling.y, 0, 0,
	    0, 0, scaling.z, 0,
	    0, 0, 0, 1};
}

Vector2 Transform::scale(const Vector2 &origin, const Vector2 &scaling)
{
	return origin | scaling;
}

Vector3 Transform::scale(const Vector3 &origin, const Vector3 &scaling)
{
	return origin | scaling;
}

Matrix4 Transform::perspective(float fov, float aspect, float near, float far)
{
	float tangent = tanf(degree_to_radians(fov) / 2.f);

	return {
	    1 / (aspect * tangent), 0, 0, 0,
	    0, 1 / tangent, 0, 0,
	    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
	    0, 0, -1, 0};
}

Matrix4 Transform::orthographic(float size, float aspect, float near, float far)
{
	float right = size * aspect;
	float top   = size;

	return {
	    1 / right, 0, 0, 0,
	    0, 1 / top, 0, 0,
	    0, 0, -2 / (far - near), -(far + near) / (far - near),
	    0, 0, 0, 1};
}

Matrix4 Transform::lookAt(const Vector3 &eye, const Vector3 &target, const Vector3 &up)
{
	auto z_axis = (eye - target).normalize();
	auto x_axis = up.cross(z_axis).normalize();
	auto y_axis = z_axis.cross(x_axis).normalize();

	return {
	    x_axis.x, x_axis.y, x_axis.z, -x_axis.dot(eye),
	    y_axis.x, y_axis.y, y_axis.z, -y_axis.dot(eye),
	    z_axis.x, z_axis.y, z_axis.z, -z_axis.dot(eye),
	    0, 0, 0, 1};
}

std::tuple<Vector3, Quaternion, Vector3> Transform::decompose(const Matrix4 &transform)
{
	// Get translation
	Vector3 translation = Vector3(transform(0, 3), transform(1, 3), transform(2, 3));

	Vector3 columns[3] = {
	    {transform(0, 0), transform(1, 0), transform(2, 0)},
	    {transform(0, 1), transform(1, 1), transform(2, 1)},
	    {transform(0, 2), transform(1, 2), transform(2, 2)}};

	// Get scale
	Vector3 scale;
	scale.x = columns[0].norm();
	scale.y = columns[1].norm();
	scale.z = columns[2].norm();

	columns[0] /= scale.x ? scale.x : 1.f;
	columns[1] /= scale.y ? scale.y : 1.f;
	columns[2] /= scale.y ? scale.y : 1.f;

	// Get rotation
	Matrix3 rotation_matrix = {
	    columns[0].x, columns[1].x, columns[2].x,
	    columns[0].y, columns[1].y, columns[2].y,
	    columns[0].z, columns[1].z, columns[2].z};

	Quaternion rotation = Quaternion(rotation_matrix);

	return std::make_tuple(translation, rotation, scale);
}
}        // namespace Ilum