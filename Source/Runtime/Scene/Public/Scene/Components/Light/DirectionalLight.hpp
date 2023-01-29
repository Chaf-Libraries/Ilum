#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class DirectionalLight : public Light
{
  public:
	DirectionalLight(Node *node);

	virtual ~DirectionalLight() = default;

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

		glm::vec4 split_depth;
		glm::mat4 view_projection[4];
		glm::vec4 shadow_cam_pos[4];

		alignas(16) glm::vec3 direction = glm::vec3(1.f);
	} m_data;
};
}        // namespace Cmpt
}        // namespace Ilum