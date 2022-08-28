#include "RenderGraph.hpp"

namespace Ilum
{
RGHandle::RGHandle(size_t handle) :
    handle(handle)
{
}

bool RGHandle::IsValid()
{
	return handle != ~0U;
}

bool RGHandle::operator<(const RGHandle &rhs) const
{
	return handle < rhs.handle;
}

bool RGHandle::operator==(const RGHandle &rhs) const
{
	return handle == rhs.handle;
}

size_t RGHandle::GetHandle() const
{
	return handle;
}

RenderGraph::RenderGraph(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

RenderGraph::~RenderGraph()
{
}

RHITexture *RenderGraph::GetTexture(RGHandle handle)
{
	return m_texture_lookup.at(handle);
}

RHIBuffer *RenderGraph::GetBuffer(RGHandle handle)
{
	return m_buffer_lookup.at(handle);
}

void RenderGraph::Execute()
{
	for (auto &pass : m_render_passes)
	{
		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BeginMarker(pass.name);

		if (!m_init)
		{
			m_initialize_barrier(*this, cmd_buffer);
			m_init = true;
		}

		pass.barrier(*this, cmd_buffer);
		pass.execute(*this, cmd_buffer);

		cmd_buffer->EndMarker();
		cmd_buffer->End();
	}
}

RenderGraph &RenderGraph::AddPass(const std::string &name, std::function<void(RenderGraph &, RHICommand *)> &&task, std::function<void(RenderGraph &, RHICommand *)> &&barrier)
{
	m_render_passes.emplace_back(RenderPassInfo{name, std::move(task), std::move(barrier)});
	return *this;
}

RenderGraph &RenderGraph::AddInitializeBarrier(RenderTask &&barrier)
{
	m_initialize_barrier = std::move(barrier);
	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const TextureCreateInfo &create_info)
{
	auto &texture = m_textures.emplace_back(p_rhi_context->CreateTexture(create_info.desc));
	m_texture_lookup.emplace(create_info.handle, texture.get());

	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const std::vector<TextureCreateInfo> &create_infos)
{
	TextureDesc pool_desc = {};
	pool_desc.name        = "Pool Texture " + std::to_string(m_textures.size());
	pool_desc.width       = 0;
	pool_desc.width       = 0;
	pool_desc.height      = 0;
	pool_desc.depth       = 0;
	pool_desc.mips        = 0;
	pool_desc.layers      = 0;
	pool_desc.samples     = 0;
	pool_desc.format      = (RHIFormat) 0;
	pool_desc.usage       = (RHITextureUsage) 0;

	// Memory alias
	for (auto &info : create_infos)
	{
		pool_desc.width   = std::max(pool_desc.width, info.desc.width);
		pool_desc.height  = std::max(pool_desc.height, info.desc.height);
		pool_desc.depth   = std::max(pool_desc.depth, info.desc.depth);
		pool_desc.mips    = std::max(pool_desc.mips, info.desc.mips);
		pool_desc.layers  = std::max(pool_desc.layers, info.desc.layers);
		pool_desc.samples = std::max(pool_desc.samples, info.desc.samples);
		pool_desc.format  = (RHIFormat) std::max((uint64_t) pool_desc.format, (uint64_t) info.desc.format);
		pool_desc.usage   = pool_desc.usage | info.desc.usage;
	}

	m_textures.emplace_back(p_rhi_context->CreateTexture(pool_desc));
	auto pool_texture = m_textures.back().get();

	for (auto &info : create_infos)
	{
		auto &texture = m_textures.emplace_back(pool_texture->Alias(info.desc));
		m_texture_lookup.emplace(info.handle, texture.get());
	}

	return *this;
}

RenderGraph &RenderGraph::RegisterBuffer(const BufferCreateInfo &create_info)
{
	auto &buffer = m_buffers.emplace_back(p_rhi_context->CreateBuffer(create_info.desc));
	m_buffer_lookup.emplace(create_info.handle, buffer.get());

	return *this;
}
}        // namespace Ilum