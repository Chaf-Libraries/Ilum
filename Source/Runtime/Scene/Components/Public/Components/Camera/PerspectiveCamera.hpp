#pragma once

#include "Camera.hpp"

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API PerspectiveCamera : public Camera
{
  public:
	PerspectiveCamera(Node *node);

	virtual ~PerspectiveCamera() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;

	void SetFov(float fov);

	void SetAspect(float aspect);

	void SetNearPlane(float near_plane);

	void SetFarPlane(float far_plane);

	float GetFov() const;

	float GetAspect() const;

	float GetNearPlane() const;

	float GetFarPlane() const;

  protected:
	virtual void Update() override;

  private:
	bool m_dirty = false;
	float m_fov   = 45.f;
	float m_aspect = 1.f;
	float m_near   = 0.01f;
	float m_far    = 1000.f;
};
}        // namespace Cmpt
}        // namespace Ilum