#include "RawCUDACompute.hpp"
#include "RawCUDACompute.cuh"

#include <RHI/../../Private/Backend/CUDA/Command.hpp>
#include <RHI/../../Private/Backend/CUDA/Texture.hpp>
#include <RHI/RHIDevice.hpp>

namespace Ilum
{
RenderPassDesc RawCUDACompute::CreateDesc()
{
	RenderPassDesc desc;

	return desc.SetName<RawCUDACompute>()
	    .SetBindPoint(BindPoint::CUDA)
	    .Write("Result", RenderResourceDesc::Type::Texture, RHIResourceState::UnorderedAccess);
}

RenderGraph::RenderTask RawCUDACompute::Create(const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
{
	return [=](RenderGraph &render_graph, RHICommand *cmd_buffer, rttr::variant &config) {
		auto *result = render_graph.GetCUDATexture(desc.resources.at("Result").handle);

		auto cuda_texture = static_cast<CUDA::Texture *>(result);

		static_cast<CUDA::Command *>(cmd_buffer)->Execute([=]() {
			ExecuteRawCUDAComputeKernal(cuda_texture->GetSurfaceHostHandle()[0], result->GetDesc().width, result->GetDesc().height, 1, 8, 8, 1);
		});
	};
}
}        // namespace Ilum