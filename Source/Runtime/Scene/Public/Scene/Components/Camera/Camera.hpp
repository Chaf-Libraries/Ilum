#pragma once

#include <Scene/Component.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class Camera : public Component
{
  public:
	Camera(const char *name, Node *node);

	virtual ~Camera() = default;

	virtual bool OnImGui() override = 0;

	virtual std::type_index GetType() const = 0;

	virtual void Save(OutputArchive &archive) const = 0;

	virtual void Load(InputArchive &archive) = 0;

	void SetAspect(float aspect);

	float GetAspect();

	void SetNearPlane(float near_plane);

	void SetFarPlane(float far_plane);

	float GetNearPlane() const;

	float GetFarPlane() const;

	glm::mat4 GetViewMatrix();

	glm::mat4 GetProjectionMatrix();

	glm::mat4 GetViewProjectionMatrix();

	glm::mat4 GetInvViewMatrix();

	glm::mat4 GetInvProjectionMatrix();

	glm::mat4 GetInvViewProjectionMatrix();

	const std::array<glm::vec4, 6> &GetFrustumPlanes();

  protected:
	virtual void UpdateProjection() = 0;

	void UpdateView();

  protected:
	float m_aspect = 1.f;
	float m_near   = 0.1f;
	float m_far    = 500.f;

	std::array<glm::vec4, 6> m_frustum_planes;

	glm::mat4 m_view                = glm::mat4(1.f);
	glm::mat4 m_inv_view            = glm::mat4(1.f);
	glm::mat4 m_projection          = glm::mat4(1.f);
	glm::mat4 m_inv_projection      = glm::mat4(1.f);
	glm::mat4 m_view_projection     = glm::mat4(1.f);
	glm::mat4 m_inv_view_projection = glm::mat4(1.f);
};
}        // namespace Cmpt
}        // namespace Ilum