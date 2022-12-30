#pragma once

#include "Camera.hpp"

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API OrthographicCamera : public Camera
{
  public:
	OrthographicCamera(Node *node);

	virtual ~OrthographicCamera() = default;

	virtual void OnImGui() override;

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
	bool  m_dirty    = false;
	float m_scale    = 1.f;
	float m_offset_x = 0.f;
	float m_offset_y = 0.f;
};
}        // namespace Cmpt
}        // namespace Ilum