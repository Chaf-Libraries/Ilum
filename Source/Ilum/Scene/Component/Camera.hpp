#pragma once

#include "Component.hpp"

#include <Geometry/Frustum.hpp>

#include <RHI/Buffer.hpp>

#include <glm/glm.hpp>

#include <string>

namespace Ilum::cmpt
{
enum class CameraType : int32_t
{
	Perspective,
	Orthographic
};

class Camera : public Component
{
  public:
	Camera() = default;

	bool OnImGui(ImGuiContext &context) override;

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_view, m_projection, m_view_projection, m_near_plane, m_far_plane,
		   m_aspect, m_fov, m_left, m_right, m_bottom, m_top);
	}

	void SetType(CameraType type);
	void SetAspect(float aspect);

	CameraType       GetType() const;
	float            GetAspect() const;
	const glm::mat4 &GetView() const;
	const glm::mat4 &GetProjection() const;
	const glm::mat4 &GetViewProjection() const;
	float            GetNearPlane() const;
	float            GetFarPlane() const;

	Buffer *GetBuffer();

	glm::vec4 WorldToScreen(glm::vec3 position, glm::vec2 extent, glm::vec2 offset);
	glm::vec3 ScreenToWorld(glm::vec4 position, glm::vec2 extent, glm::vec2 offset);

  private:
	CameraType m_type = CameraType::Perspective;

	glm::mat4 m_view = {};
	glm::mat4 m_projection = {};
	glm::mat4 m_view_projection = {};

	float m_near_plane = 0.01f;
	float m_far_plane  = 1000.f;

	// Perspective
	float m_aspect = 1.f;
	float m_fov    = 45.f;

	// Orthographic
	float m_left   = -1.f;
	float m_right  = 1.f;
	float m_bottom = -1.f;
	float m_top    = 1.f;

	uint32_t m_frame_count = 0;

	std::unique_ptr<Buffer> m_buffer = nullptr;
};
}        // namespace Ilum::cmpt