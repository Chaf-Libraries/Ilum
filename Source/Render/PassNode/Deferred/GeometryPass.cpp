#include "GeometryPass.hpp"
#include "RenderGraph/RenderNode.hpp"

namespace Ilum::Render
{
GeometryPass::GeometryPass(RenderGraph &render_graph) :
    IPassNode("Geometry Pass", render_graph)
{
	BindBuffer(0, 0, "Vertex Buffer", VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, AccessMode::Read);
	BindBuffer(0, 1, "Index Buffer", VK_BUFFER_USAGE_INDEX_BUFFER_BIT, AccessMode::Read);
	BindImage(1, 0, "Hi-Z", VK_IMAGE_USAGE_STORAGE_BIT, AccessMode::Write);
	BindImage(0, 2, "Textures", VK_IMAGE_USAGE_SAMPLED_BIT, AccessMode::Read);
	BindSampler(0, 3, "Sampler");

	AddDependency("Indirect Draw Buffer", VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, AccessMode::Read);
}

void GeometryPass::OnExecute(Graphics::CommandBuffer &cmd_buffer)
{
}
}        // namespace Ilum::Render