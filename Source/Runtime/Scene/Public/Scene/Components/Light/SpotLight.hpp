#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class SpotLight : public Light
{
  public:
	SpotLight(Node *node);

	virtual ~SpotLight() = default;

	virtual void OnImGui() override;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	virtual std::type_index GetType() const override;

	virtual size_t GetDataSize() const override;

	virtual void *GetData(Camera *camera = nullptr) override;

  private:
	struct
	{
		glm::vec3 color = glm::vec3(1.f);

		float intensity = 100.f;

		glm::vec3 position = glm::vec3(0.f);

		float inner_angle = glm::radians(30.f);

		glm::vec3 direction = glm::vec3(0.f);

		float outer_angle = glm::radians(60.f);

		glm::mat4 view_projection = glm::mat4(1.f);
	} m_data;
};
}        // namespace Cmpt
}        // namespace Ilum