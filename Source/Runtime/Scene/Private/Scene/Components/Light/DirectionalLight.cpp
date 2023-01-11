#include "Components/Light/DirectionalLight.hpp"
#include "Components/Transform.hpp"
#include "Node.hpp"

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
	ImGui::ColorEdit3("Color", &m_data.color.x);
	ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
}

void DirectionalLight::Save(OutputArchive &archive) const
{
	archive(m_data.intensity);
}

void DirectionalLight::Load(InputArchive &archive)
{
	archive(m_data.intensity);
}

std::type_index DirectionalLight::GetType() const
{
	return typeid(DirectionalLight);
}

size_t DirectionalLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *DirectionalLight::GetData()
{
	m_data.direction = glm::mat3_cast(glm::qua<float>(glm::radians(p_node->GetComponent<Cmpt::Transform>()->GetRotation()))) * glm::vec3(0.f, -1.f, 0.f);
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum