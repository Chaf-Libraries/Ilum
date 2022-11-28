#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class EXPORT_API DirectionalLight : public Light
{
  public:
	DirectionalLight(Node *node);

	virtual ~DirectionalLight() = default;

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