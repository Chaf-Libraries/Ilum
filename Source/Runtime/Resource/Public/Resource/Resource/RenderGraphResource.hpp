#pragma once

#include "Resource.hpp"

namespace Ilum
{
struct RenderGraphDesc;

template <>
class TResource<ResourceType::RenderGraph> : public Resource
{
  public:
	explicit TResource(size_t uuid);

	explicit TResource(size_t uuid, const std::string &meta, RHIContext *rhi_context);

	virtual ~TResource() override = default;

	virtual void Load(RHIContext *rhi_context, size_t index) override;

	virtual void Import(RHIContext *rhi_context, const std::string &path) override;

	void Load(RenderGraphDesc &desc, std::string& editor_layout);
};
}        // namespace Ilum