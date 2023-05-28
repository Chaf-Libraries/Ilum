#include "Components/Light/RectLight.hpp"
#include "Components/Transform.hpp"
#include "Node.hpp"

#include <imgui.h>

namespace Ilum
{
namespace Cmpt
{
RectLight::RectLight(Node *node) :
    Light("Rect Light", node)
{
}

bool RectLight::OnImGui()
{
	m_update |= ImGui::ColorEdit3("Color", &m_data.color.x);
	m_update |= ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	m_update |= ImGui::Checkbox("Two Side", reinterpret_cast<bool *>(&m_data.two_side));
	return m_update;
}

void RectLight::Save(OutputArchive &archive) const
{
	archive(m_data.color, m_data.intensity, m_data.texture_id, m_data.corner);
}

void RectLight::Load(InputArchive &archive)
{
	archive(m_data.color, m_data.intensity, m_data.texture_id, m_data.corner);
	m_update = true;
}

std::type_index RectLight::GetType() const
{
	return typeid(RectLight);
}

size_t RectLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *RectLight::GetData(Camera *camera)
{
	glm::mat4 world_transform = p_node->GetComponent<Cmpt::Transform>()->GetWorldTransform();

	m_data.corner[0] = world_transform * glm::vec4(-0.5f, 0.f, -0.5f, 1.f);
	m_data.corner[1] = world_transform * glm::vec4(0.5f, 0.f, -0.5f, 1.f);
	m_data.corner[2] = world_transform * glm::vec4(0.5f, 0.f, 0.5f, 1.f);
	m_data.corner[3] = world_transform * glm::vec4(-0.5f, 0.f, 0.5f, 1.f);

	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum
