#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class PointLight : public Light
{
  public:
	PointLight(Node *node);

	virtual ~PointLight() = default;

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

		float filter_scale            = 2.f;
		alignas(16) float light_scale = 1.f;
		uint32_t filter_sample        = 10;
	} m_data;
};
}        // namespace Cmpt
}        // namespace Ilum