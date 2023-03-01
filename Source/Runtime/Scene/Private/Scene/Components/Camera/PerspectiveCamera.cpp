#include "Components/Camera/PerspectiveCamera.hpp"
#include "Components/Transform.hpp"

#include <Scene/Node.hpp>

#include <imgui.h>

#include <glm/gtc/matrix_transform.hpp>

namespace Ilum
{
namespace Cmpt
{
PerspectiveCamera::PerspectiveCamera(Node *node) :
    Camera("Perspective Camera", node)
{
}

bool PerspectiveCamera::OnImGui()
{
	m_update |= ImGui::DragFloat("Aspect", &m_aspect, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	m_update |= ImGui::DragFloat("Fov", &m_fov, 0.01f, 0.f, 90.f, "%.3f");
	m_update |= ImGui::DragFloat("Near Plane", &m_near, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	m_update |= ImGui::DragFloat("Far Plane", &m_far, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	
	return m_update;
}

void PerspectiveCamera::Save(OutputArchive &archive) const
{
	archive(m_aspect, m_near, m_far, m_view, m_inv_view, m_projection, m_inv_projection, m_view_projection, m_inv_view_projection, m_fov);
}

void PerspectiveCamera::Load(InputArchive &archive)
{
	archive(m_aspect, m_near, m_far, m_view, m_inv_view, m_projection, m_inv_projection, m_view_projection, m_inv_view_projection, m_fov);
	m_update = true;
}

std::type_index PerspectiveCamera::GetType() const
{
	return typeid(PerspectiveCamera);
}

void PerspectiveCamera::SetFov(float fov)
{
	m_update = true;
	m_fov    = fov;
}

float PerspectiveCamera::GetFov() const
{
	return m_fov;
}

void PerspectiveCamera::UpdateProjection()
{
	if (!m_update)
	{
		return;
	}

	m_projection     = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
	m_inv_projection = glm::inverse(m_projection);
}
}        // namespace Cmpt
}        // namespace Ilum