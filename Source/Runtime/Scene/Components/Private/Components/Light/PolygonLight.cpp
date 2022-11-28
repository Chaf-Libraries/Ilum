#include "Light/PolygonLight.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
PolygonLight::PolygonLight(Node *node) :
    Light("Point Light", node)
{
}

void PolygonLight::OnImGui()
{
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

std::type_index PolygonLight::GetType() const
{
	return typeid(PolygonLight);
}

size_t PolygonLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *PolygonLight::GetData() const
{
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum