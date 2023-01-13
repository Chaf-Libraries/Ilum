#pragma once

#include "../Resource.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
class RenderGraphDesc;
class RenderGraph;
class Renderer;

template <>
class EXPORT_API Resource<ResourceType::RenderPipeline> final : public IResource
{
  public:
	Resource(RHIContext *rhi_context, const std::string &name);

	Resource(RHIContext *rhi_context, const std::string &name, RenderGraphDesc &&desc);

	virtual ~Resource() override;

	virtual bool Validate() const override;

	virtual void Load(RHIContext *rhi_context) override;

	std::unique_ptr<RenderGraph> Compile(RHIContext *rhi_context, Renderer* renderer, glm::vec2 viewport, const std::string &layout = "");

	const std::string &GetLayout() const;

	RenderGraphDesc &GetDesc();

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum