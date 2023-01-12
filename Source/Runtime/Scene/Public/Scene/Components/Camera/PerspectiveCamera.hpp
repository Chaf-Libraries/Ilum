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

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	void SetFov(float fov);

	float GetFov() const;

  protected:
	virtual void UpdateProjection() override;

  private:
	float m_fov   = 45.f;
};
}        // namespace Cmpt
}        // namespace Ilum