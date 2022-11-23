#include "Light/PointLight.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
PointLight::PointLight(Node *node) :
    Light("Point Light", node)
{
}

void PointLight::OnImGui()
{
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");

}

std::type_index PointLight::GetType() const
{
	return typeid(PointLight);
}

size_t PointLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *PointLight::GetData() const
{
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum