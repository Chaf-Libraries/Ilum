#pragma once

#include "Component.hpp"

#include <Core/Macro.hpp>

namespace Ilum
{
STRUCT(LightComponent, Enable) :
    public Component
{
	RTTR_ENABLE(Component);
};

STRUCT(PointLightComponent, Enable) :
    public LightComponent
{
	META(Editor("ColorEdit"))
	glm::vec3 color = {1.f, 1.f, 1.f};

	META(Min(0.f))
	float intensity = 1.f;

	META(Min(0.f))
	float range = 1.f;

	META(Min(0.f))
	float radius = 1.f;

	bool cast_shadow = true;

	RTTR_ENABLE(LightComponent);
};

STRUCT(SpotLightComponent, Enable) :
    public LightComponent
{
	META(Editor("ColorEdit"))
	glm::vec3 color = {1.f, 1.f, 1.f};

	META(Min(0.f))
	float intensity = 1.f;

	META(Min(0.f), Max(90.f), Editor("Slider"))
	float inner_angle = 12.5f;

	META(Min(0.f), Max(90.f), Editor("Slider"))
	float outer_angle = 17.5f;

	bool cast_shadow = true;

	RTTR_ENABLE(LightComponent);
};

STRUCT(DirectionalLightComponent, Enable) :
    public LightComponent
{
	RTTR_ENABLE(LightComponent);
};

STRUCT(RectLightComponent, Enable) :
    public LightComponent
{
	RTTR_ENABLE(LightComponent);
};
}        // namespace Ilum