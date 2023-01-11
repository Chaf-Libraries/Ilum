#include "Components/Light/RectLight.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
RectLight::RectLight(Node *node) :
    Light("Point Light", node)
{
}

void RectLight::OnImGui()
{
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

void RectLight::Save(OutputArchive &archive) const
{
}

void RectLight::Load(InputArchive &archive)
{
}

std::type_index RectLight::GetType() const
{
	return typeid(RectLight);
}

size_t RectLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *RectLight::GetData()
{
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum