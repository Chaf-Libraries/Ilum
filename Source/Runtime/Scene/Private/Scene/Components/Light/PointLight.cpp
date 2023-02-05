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

bool PointLight::OnImGui()
{
	m_update |= ImGui::ColorEdit3("Color", &m_data.color.x);
	m_update |= ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::Text("Shadow Map Setting");
	m_update |= ImGui::SliderInt("Filter Sample", reinterpret_cast<int32_t *>(&m_data.filter_sample), 1, 100);
	m_update |= ImGui::DragFloat("Filter Scale", &m_data.filter_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	m_update |= ImGui::DragFloat("Light Scale", &m_data.light_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	
	return m_update;
}

void PointLight::Save(OutputArchive &archive) const
{
	archive(m_data.color, m_data.intensity, m_data.position, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
}

void PointLight::Load(InputArchive &archive)
{
	archive(m_data.color, m_data.intensity, m_data.position, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
	m_update = true;
}

std::type_index PointLight::GetType() const
{
	return typeid(PointLight);
}

size_t PointLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *PointLight::GetData(Camera *camera)
{
	auto *transform = p_node->GetComponent<Cmpt::Transform>();

	glm::vec3 scale;
	glm::quat rotation;
	glm::vec3 translation;
	glm::vec3 skew;
	glm::vec4 perspective;
	glm::decompose(transform->GetWorldTransform(), scale, rotation, translation, skew, perspective);

	m_data.position = translation;
	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum