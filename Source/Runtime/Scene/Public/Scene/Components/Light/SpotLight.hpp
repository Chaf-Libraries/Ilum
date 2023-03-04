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

	virtual bool OnImGui() override;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	virtual std::type_index GetType() const override;

	bool CastShadow() const;

	void SetShadowID(uint32_t &shadow_id);

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

		// Shadow map setting
		float    filter_scale  = 2.f;
		float    light_scale   = 1.f;
		uint32_t filter_sample = 10;
		uint32_t cast_shadow   = 1;

		alignas(16) uint32_t shadow_id = ~0u;
	} m_data;
};
}        // namespace Cmpt
}        // namespace Ilum