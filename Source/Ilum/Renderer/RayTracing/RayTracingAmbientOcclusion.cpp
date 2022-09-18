#include "RayTracingAmbientOcclusion.hpp"

namespace Ilum
{
RenderPassDesc RayTracingAmbientOcclusion::CreateDesc()
{
	RenderPassDesc desc = {};

	desc.name = "RayTracingAmbientOcclusion";
	desc
	    .SetName("RayTracingAmbientOcclusion")
	    .SetBindPoint(BindPoint::RayTracing)
	    .Write("VisibilityBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::RenderTarget)
	    .Write("DepthBuffer", RenderResourceDesc::Type::Texture, RHIResourceState::DepthWrite);

	return desc;
}

RenderGraph::RenderTask RayTracingAmbientOcclusion::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return RenderGraph::RenderTask{};
}
}        // namespace Ilum