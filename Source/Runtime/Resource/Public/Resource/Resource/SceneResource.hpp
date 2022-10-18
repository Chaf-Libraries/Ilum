#pragma once

#include "Resource.hpp"

namespace Ilum
{
class Scene;

template <>
class TResource<ResourceType::Scene> : public Resource
{
  public:
	explicit TResource(size_t uuid);

	explicit TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context);

	virtual ~TResource() override = default;

	virtual void Load(RHIContext *rhi_context, size_t index) override;

	virtual void Import(RHIContext *rhi_context, const std::string &path) override;

	void Load(Scene *scene);

	void Save(Scene *scene);

  private:
	std::string m_name;
};
}        // namespace Ilum