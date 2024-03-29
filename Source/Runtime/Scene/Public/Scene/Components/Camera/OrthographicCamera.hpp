#pragma once

#include "Camera.hpp"

namespace Ilum
{
namespace Cmpt
{
class OrthographicCamera : public Camera
{
  public:
	OrthographicCamera(Node *node);

	virtual ~OrthographicCamera() = default;

	virtual bool OnImGui() override;

	virtual std::type_index GetType() const override;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	void SetScale(float scale);

	void SetOffset(float x, float y);

	float GetScale() const;

	float GetOffsetX() const;

	float GetOffsetY() const;

  protected:
	virtual void UpdateProjection() override;

  private:
	float m_scale    = 1.f;
	float m_offset_x = 0.f;
	float m_offset_y = 0.f;
};
}        // namespace Cmpt
}        // namespace Ilum