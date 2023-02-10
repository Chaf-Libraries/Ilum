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

bool SpotLight::OnImGui()
{
	float inner_angle = glm::degrees(m_data.inner_angle);
	float outer_angle = glm::degrees(m_data.outer_angle);

	m_update |= ImGui::ColorEdit3("Color", &m_data.color.x);
	m_update |= ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	m_update |= ImGui::DragFloat("Inner Angle", &inner_angle, 0.1f, 0.f, outer_angle, "%.1f");
	m_update |= ImGui::DragFloat("Outer Angle", &outer_angle, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::Text("Shadow Map Setting");
	m_update |= ImGui::SliderInt("Filter Sample", reinterpret_cast<int32_t *>(&m_data.filter_sample), 1, 100);
	m_update |= ImGui::DragFloat("Filter Scale", &m_data.filter_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	m_update |= ImGui::DragFloat("Light Scale", &m_data.light_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");

	m_data.inner_angle = glm::radians(inner_angle);
	m_data.outer_angle = glm::radians(outer_angle);

	return m_update;
}

void SpotLight::Save(OutputArchive &archive) const
{
	archive(m_data.color, m_data.intensity, m_data.inner_angle, m_data.outer_angle, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
}

void SpotLight::Load(InputArchive &archive)
{
	archive(m_data.color, m_data.intensity, m_data.inner_angle, m_data.outer_angle, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
	m_update = true;
}

std::type_index SpotLight::GetType() const
{
	return typeid(SpotLight);
}

size_t SpotLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *SpotLight::GetData(Camera *camera)
{
	auto *transform = p_node->GetComponent<Cmpt::Transform>();

	glm::vec3 scale;
	glm::quat rotation;
	glm::vec3 translation;
	glm::vec3 skew;
	glm::vec4 perspective;
	glm::decompose(transform->GetWorldTransform(), scale, rotation, translation, skew, perspective);

	m_data.position        = translation;
	m_data.direction       = glm::mat3_cast(rotation) * glm::vec3(0.f, 0.f, -1.f);
	m_data.view_projection = glm::perspective(2.f * m_data.outer_angle, 1.0f, 0.01f, 1000.f) * glm::lookAt(translation, translation + m_data.direction, glm::vec3(0.f, 1.f, 0.f));

	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum