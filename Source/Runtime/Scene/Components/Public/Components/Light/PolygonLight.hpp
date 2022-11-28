#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class PolygonLight : public Light
{
  public:
	PolygonLight(Node *node);

	virtual ~PolygonLight() = default;

	virtual void OnImGui() override;

	virtual std::type_index GetType() const override;

	virtual size_t GetDataSize() const override;

	virtual void *GetData() const override;

  private:
	struct
	{
		float     intensity = 100.f;
		glm::vec3 color     = glm::vec3(1.f);
	} m_data;
};
}        // namespace Cmpt
}        // namespace Ilum