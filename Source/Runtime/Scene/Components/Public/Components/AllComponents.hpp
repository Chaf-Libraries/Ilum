#pragma once

#include "Light/DirectionalLight.hpp"
#include "Light/PointLight.hpp"
#include "Light/PolygonLight.hpp"
#include "Light/SpotLight.hpp"

#include "Transform.hpp"

struct ImGuiContext;

namespace Ilum
{
namespace Cmpt
{
EXPORT_API void SetImGuiContext(ImGuiContext *context);
}
}        // namespace Ilum