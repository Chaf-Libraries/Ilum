#pragma once

namespace Ilum::Cmpt
{
enum class CameraType
{
	Perspective,
	Orthographic
};

struct Camera
{
	CameraType type = CameraType::Perspective;

	float aspect_ratio = 1.f;
	float fov          = 60.f;
	float far          = 100.f;
	float near         = 0.01f;
};
}