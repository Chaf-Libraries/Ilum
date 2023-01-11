#include "Components/Light/PointLight.hpp"
#include "Components/Transform.hpp"
#include "Node.hpp"

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
	ImGui::ColorEdit3("Color", &m_data.color.x);
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::DragFloat("Range", &m_data.range, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::DragFloat("Radius", &m_data.radius, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

void PointLight::Save(OutputArchive &archive) const
{
	archive(m_data.color, m_data.intensity, m_data.position, m_data.radius, m_data.range);
}

void PointLight::Load(InputArchive &archive)
{
	archive(m_data.color, m_data.intensity, m_data.position, m_data.radius, m_data.range);
}

std::type_index PointLight::GetType() const
{
	return typeid(PointLight);
}

size_t PointLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *PointLight::GetData()
{
	m_data.position = p_node->GetComponent<Cmpt::Transform>()->GetTranslation();
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum