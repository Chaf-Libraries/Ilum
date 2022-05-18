#include "Camera.hpp"
#include "Light.hpp"
#include "Transform.hpp"

#include "Scene/Entity.hpp"

#include <Shaders/ShaderInterop.hpp>

#include <imgui.h>

namespace Ilum::cmpt
{
void Camera::SetType(CameraType type)
{
	m_type = type;
	Update();
}

void Camera::SetAspect(float aspect)
{
	m_aspect = aspect;
	Update();
}

CameraType Camera::GetType() const
{
	return m_type;
}

float Camera::GetAspect() const
{
	return m_aspect;
}

const glm::mat4 &Camera::GetView() const
{
	return m_view;
}

const glm::mat4 &Camera::GetProjection() const
{
	return m_projection;
}

const glm::mat4 &Camera::GetViewProjection() const
{
	return m_view_projection;
}

float Camera::GetNearPlane() const
{
	return m_near_plane;
}

float Camera::GetFarPlane() const
{
	return m_far_plane;
}

Buffer *Camera::GetBuffer()
{
	return m_buffer.get();
}

glm::vec4 Camera::WorldToScreen(glm::vec3 position, glm::vec2 extent, glm::vec2 offset)
{
	glm::vec4 pos = m_view_projection * glm::vec4(position, 1.f);
	pos.x *= std::fabsf(0.5f / pos.w);
	pos.y *= std::fabsf(0.5f / pos.w);
	pos += glm::vec4(0.5f, 0.5f, 0.f, 0.f);
	pos.y = 1.f - pos.y;
	pos.x *= extent.x;
	pos.y *= extent.y;
	pos.x += offset.x;
	pos.y += offset.y;

	return pos;
}

glm::vec3 Camera::ScreenToWorld(glm::vec4 position, glm::vec2 extent, glm::vec2 offset)
{
	glm::vec4 pos = position;
	pos.x -= offset.x;
	pos.y -= offset.y;
	pos.x /= extent.x;
	pos.y /= extent.y;
	pos.y = 1.f - pos.y;
	pos -= glm::vec4(0.5f, 0.5f, 0.f, 0.f);
	pos.x /= std::fabsf(0.5f / pos.w);
	pos.y /= std::fabsf(0.5f / pos.w);
	pos = glm::inverse(m_view_projection) * pos;

	return pos;
}

bool Camera::OnImGui(ImGuiContext &context)
{
	const char *const camera_types[] = {"Perspective", "Orthographic"};
	m_update |= ImGui::Combo("Camera Type", reinterpret_cast<int32_t *>(&m_type), camera_types, 2);

	if (m_type == CameraType::Perspective)
	{
		m_update |= ImGui::DragFloat("Aspect", &m_aspect, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
		m_update |= ImGui::DragFloat("Fov", &m_fov, 0.01f, 0.f, 90.f, "%.3f");
	}
	else
	{
		m_update |= ImGui::DragFloat("Left", &m_left, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		m_update |= ImGui::DragFloat("Right", &m_right, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		m_update |= ImGui::DragFloat("Bottom", &m_bottom, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
		m_update |= ImGui::DragFloat("Top", &m_top, 0.01f, -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), "%.3f");
	}

	m_update |= ImGui::DragFloat("Near Plane", &m_near_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");
	m_update |= ImGui::DragFloat("Far Plane", &m_far_plane, 0.01f, 0.f, std::numeric_limits<float>::max(), "%.3f");

	return m_update;
}

void Camera::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	m_frame_count++;

	if (!m_buffer)
	{
		m_buffer = std::make_unique<Buffer>(device, BufferDesc(sizeof(ShaderInterop::Camera), 1, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU));
	}

	if (m_update)
	{
		m_frame_count = 0;

		ShaderInterop::Camera *camera_data = static_cast<ShaderInterop::Camera *>(m_buffer->Map());

		Entity e = Entity(scene, entity);

		auto &transform = e.GetComponent<cmpt::Transform>();

		m_view            = glm::inverse(transform.GetWorldTransform());
		m_projection      = m_type == cmpt::CameraType::Perspective ?
		                        glm::perspective(glm::radians(m_fov), m_aspect, m_near_plane, m_far_plane) :
                                glm::ortho(m_left, m_right, m_bottom, m_top, m_near_plane, m_far_plane);
		m_view_projection = m_projection * m_view;

		camera_data->view            = m_view;
		camera_data->projection      = m_projection;
		camera_data->inv_view        = transform.GetWorldTransform();
		camera_data->inv_projection  = glm::inverse(m_projection);
		camera_data->view_projection = m_projection * m_view;
		camera_data->position        = transform.GetWorldTransform()[3];
		camera_data->frame_count     = m_frame_count;

		m_buffer->Flush(m_buffer->GetSize());
		m_buffer->Unmap();

		// Main camera update -> update cascade shadow
		if (entity == scene.GetMainCamera())
		{
			auto view = scene.GetRegistry().view<cmpt::Light>();
			view.each([&](entt::entity entity, cmpt::Light &light) {
				if (light.GetType() == LightType::Directional)
				{
					light.Update();
				}
			});
		}

		m_update = false;
	}
}

}        // namespace Ilum::cmpt