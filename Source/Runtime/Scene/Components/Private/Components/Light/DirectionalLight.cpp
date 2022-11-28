#include "Light/DirectionalLight.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
DirectionalLight::DirectionalLight(Node *node) :
    Light("Directional Light", node)
{
}

void DirectionalLight::OnImGui()
{
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

std::type_index DirectionalLight::GetType() const
{
	return typeid(DirectionalLight);
}

size_t DirectionalLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *DirectionalLight::GetData() const
{
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum