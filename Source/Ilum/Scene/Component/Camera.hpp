#pragma once

#include "Eventing/Event.hpp"

#undef far
#undef near

namespace Ilum::cmpt
{
enum class CameraType
{
	Perspective,
	Orthographic
};

struct Camera
{
	CameraType type = CameraType::Perspective;

	float aspect = 1.f;
	float fov    = 60.f;
	float far    = 100.f;
	float near   = 0.01f;

	Event<> Event_Detach;

	~Camera()
	{
		Event_Detach.invoke();
	}
};
}        // namespace Ilum::Cmpt