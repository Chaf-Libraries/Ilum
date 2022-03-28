#pragma once

#include "CopyFrame.hpp"

#include "Renderer/Renderer.hpp"
#include "Renderer/RenderGraph/RenderGraph.hpp"

namespace Ilum::pass
{
CopyFrame::CopyFrame(const std::string &from, const std::string &to) :
    m_from(from), m_to(to)
{
}

void CopyFrame::setupPipeline(PipelineState &state)
{
	state.addDependency(m_from, VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
	state.addDependency(m_to, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

void CopyFrame::resolveResources(ResolveState &resolve)
{
}

void CopyFrame::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	cmd_buffer.copyImage(
	    ImageInfo{state.graph.getAttachment(m_from), VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
	    ImageInfo{state.graph.getAttachment(m_to), VK_IMAGE_USAGE_TRANSFER_DST_BIT});
}
}        // namespace Ilum::pass