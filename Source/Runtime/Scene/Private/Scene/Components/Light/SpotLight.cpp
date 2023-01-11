#include "Components/Light/SpotLight.hpp"
#include "Components/Transform.hpp"
#include "Node.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
SpotLight::SpotLight(Node *node) :
    Light("Spot Light", node)
{
}

void SpotLight::OnImGui()
{
	ImGui::ColorEdit3("Color", &m_data.color.x);
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::DragFloat("Inner Angle", &m_data.inner_angle, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::DragFloat("Outer Angle", &m_data.outer_angle, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

void SpotLight::Save(OutputArchive &archive) const
{
	archive(m_data.color, m_data.intensity, m_data.inner_angle, m_data.outer_angle);
}

void SpotLight::Load(InputArchive &archive)
{
	archive(m_data.color, m_data.intensity, m_data.inner_angle, m_data.outer_angle);
}

std::type_index SpotLight::GetType() const
{
	return typeid(SpotLight);
}

size_t SpotLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *SpotLight::GetData()
{
	auto *transform  = p_node->GetComponent<Cmpt::Transform>();
	m_data.position  = transform->GetTranslation();
	m_data.direction = glm::mat3_cast(glm::qua<float>(glm::radians(transform->GetRotation()))) * glm::vec3(0.f, -1.f, 0.f);
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum