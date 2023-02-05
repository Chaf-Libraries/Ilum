#pragma once

#include "Light.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
namespace Cmpt
{
class EnvironmentLight : public Light
{
  public:
	EnvironmentLight(Node *node);

	virtual ~EnvironmentLight() = default;

	virtual bool OnImGui() override;

	virtual void Save(OutputArchive &archive) const override;

	virtual void Load(InputArchive &archive) override;

	virtual std::type_index GetType() const override;

	virtual size_t GetDataSize() const override;

	virtual void *GetData(Camera *camera = nullptr) override;

  private:
	std::string m_texture = "";
};
}        // namespace Cmpt
}        // namespace Ilum