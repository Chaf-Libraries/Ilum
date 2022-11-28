#pragma once

#include "Light/DirectionalLight.hpp"
#include "Light/PointLight.hpp"
#include "Light/PolygonLight.hpp"
#include "Light/SpotLight.hpp"

#include "Transform.hpp"

struct ImGuiContext;

extern "C"
{
	EXPORT_API void ConfigureImGui(ImGuiContext *context);
}