#include "Components/Light/DirectionalLight.hpp"
#include "Components/Camera/Camera.hpp"
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

bool DirectionalLight::OnImGui()
{
	m_update |= ImGui::ColorEdit3("Color", &m_data.color.x);
	m_update |= ImGui::DragFloat("Intensity", &m_data.intensity, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	ImGui::Text("Shadow Map Setting");
	m_update |= ImGui::Checkbox("Cast Shadow", reinterpret_cast<bool *>(&m_data.cast_shadow));
	m_update |= ImGui::SliderInt("Filter Sample", reinterpret_cast<int32_t *>(&m_data.filter_sample), 1, 100);
	m_update |= ImGui::DragFloat("Filter Scale", &m_data.filter_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");
	m_update |= ImGui::DragFloat("Light Scale", &m_data.light_scale, 0.1f, 0.f, std::numeric_limits<float>::max(), "%.1f");

	return m_update;
}

void DirectionalLight::Save(OutputArchive &archive) const
{
	archive(m_data.intensity, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
}

void DirectionalLight::Load(InputArchive &archive)
{
	archive(m_data.intensity, m_data.filter_sample, m_data.filter_scale, m_data.light_scale);
	m_update = true;
}

std::type_index DirectionalLight::GetType() const
{
	return typeid(DirectionalLight);
}

bool DirectionalLight::CastShadow() const
{
	return m_data.cast_shadow;
}

void DirectionalLight::SetShadowID(uint32_t &shadow_id)
{
	m_data.shadow_id = m_data.cast_shadow ? shadow_id++ : ~0u;
}

size_t DirectionalLight::GetDataSize() const
{
	return sizeof(m_data);
}

void *DirectionalLight::GetData(Camera *camera)
{
	auto *transform = p_node->GetComponent<Cmpt::Transform>();

	glm::vec3 scale;
	glm::quat rotation;
	glm::vec3 translation;
	glm::vec3 skew;
	glm::vec4 perspective;
	glm::decompose(transform->GetWorldTransform(), scale, rotation, translation, skew, perspective);

	m_data.direction = glm::mat3_cast(rotation) * glm::vec3(0.f, 0.f, -1.f);

	if (camera)
	{
		float cascade_splits[4] = {0.f};

		float near_clip  = camera->GetNearPlane();
		float far_clip   = camera->GetFarPlane();
		float clip_range = far_clip - near_clip;
		float ratio      = far_clip / near_clip;

		// Calculate split depths based on view camera frustum
		for (uint32_t i = 0; i < 4; i++)
		{
			float p           = (static_cast<float>(i) + 1.f) / 4.f;
			float log         = near_clip * std::pow(ratio, p);
			float uniform     = near_clip + clip_range * p;
			float d           = 0.95f * (log - uniform) + uniform;
			cascade_splits[i] = (d - near_clip) / clip_range;
		}

		// Calculate orthographic projection matrix for each cascade
		float last_split_dist = 0.f;
		for (uint32_t i = 0; i < 4; i++)
		{
			float split_dist = cascade_splits[i];

			glm::vec3 frustum_corners[8] = {
			    glm::vec3(-1.0f, 1.0f, 0.0f),
			    glm::vec3(1.0f, 1.0f, 0.0f),
			    glm::vec3(1.0f, -1.0f, 0.0f),
			    glm::vec3(-1.0f, -1.0f, 0.0f),
			    glm::vec3(-1.0f, 1.0f, 1.0f),
			    glm::vec3(1.0f, 1.0f, 1.0f),
			    glm::vec3(1.0f, -1.0f, 1.0f),
			    glm::vec3(-1.0f, -1.0f, 1.0f)};

			// Project frustum corners into world space
			glm::mat4 inv_cam = glm::inverse(camera->GetViewProjectionMatrix());
			for (uint32_t j = 0; j < 8; j++)
			{
				glm::vec4 inv_corner = inv_cam * glm::vec4(frustum_corners[j], 1.f);
				frustum_corners[j]   = glm::vec3(inv_corner / inv_corner.w);
			}

			for (uint32_t j = 0; j < 4; j++)
			{
				glm::vec3 corner_ray   = frustum_corners[j + 4] - frustum_corners[j];
				frustum_corners[j + 4] = frustum_corners[j] + corner_ray * split_dist;
				frustum_corners[j]     = frustum_corners[j] + corner_ray * last_split_dist;
			}

			// Get frustum center
			glm::vec3 frustum_center = glm::vec3(0.0f);
			for (uint32_t j = 0; j < 8; j++)
			{
				frustum_center += frustum_corners[j];
			}
			frustum_center /= 8.0f;

			float radius = 0.0f;
			for (uint32_t j = 0; j < 8; j++)
			{
				float distance = glm::length(frustum_corners[j] - frustum_center);
				radius         = glm::max(radius, distance);
			}
			radius = std::ceil(radius * 16.0f) / 16.0f;

			glm::vec3 max_extents = glm::vec3(radius);
			glm::vec3 min_extents = -max_extents;

			max_extents.z = 1000.f;

			glm::vec3 light_dir = glm::normalize(m_data.direction);

			m_data.shadow_cam_pos[i] = glm::vec4(frustum_center - light_dir * max_extents.z, 1.0);

			glm::mat4 light_view_matrix  = glm::lookAt(glm::vec3(m_data.shadow_cam_pos[i]), frustum_center, glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 light_ortho_matrix = glm::ortho(min_extents.x, max_extents.x, min_extents.y, max_extents.y, -2.f * (max_extents.z - min_extents.z), max_extents.z - min_extents.z);

			// Store split distance and matrix in cascade
			m_data.split_depth[i]     = -(near_clip + split_dist * clip_range);
			m_data.view_projection[i] = light_ortho_matrix * light_view_matrix;

			// Stablize
			// glm::vec3 shadow_origin = glm::vec3(0.0f);
			// shadow_origin           = (m_data.view_projection[i] * glm::vec4(shadow_origin, 1.0f));
			// shadow_origin *= 512.f;

			// glm::vec3 rounded_origin = glm::round(shadow_origin);
			// glm::vec3 round_offset   = rounded_origin - shadow_origin;
			// round_offset             = round_offset / 512.f;
			// round_offset.z           = 0.0f;

			// m_data.view_projection[i][3][0] += round_offset.x;
			// m_data.view_projection[i][3][1] += round_offset.y;

			last_split_dist = cascade_splits[i];
		}
	}

	return (void *) (&m_data);
}
}        // namespace Cmpt
}        // namespace Ilum