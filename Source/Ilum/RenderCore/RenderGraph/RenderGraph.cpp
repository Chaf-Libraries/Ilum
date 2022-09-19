#include "RenderGraph.hpp"

namespace Ilum
{
RGHandle::RGHandle() :
    handle(~0U)
{
}

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
	p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Execute();
	p_rhi_context->GetQueue(RHIQueueFamily::Compute)->Execute();
	p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Wait();
	p_rhi_context->GetQueue(RHIQueueFamily::Compute)->Wait();

	p_rhi_context->WaitIdle();
}

RHITexture *RenderGraph::GetTexture(RGHandle handle)
{
	auto iter = m_texture_lookup.find(handle);
	return iter == m_texture_lookup.end() ? nullptr : iter->second;
}

RHIBuffer *RenderGraph::GetBuffer(RGHandle handle)
{
	auto iter = m_buffer_lookup.find(handle);
	return iter == m_buffer_lookup.end() ? nullptr : iter->second;
}

void RenderGraph::Execute()
{
	if (!m_init)
	{
		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BeginMarker("Initialize");
		m_initialize_barrier(*this, cmd_buffer);
		m_init = true;
		cmd_buffer->EndMarker();
		cmd_buffer->End();
		p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Submit({cmd_buffer});
	}

	std::vector<RHICommand *> cmd_buffers;
	cmd_buffers.reserve(m_render_passes.size());
	for (auto &pass : m_render_passes)
	{
		auto *cmd_buffer = p_rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BeginMarker(pass.name);
		pass.profiler->Begin(cmd_buffer, p_rhi_context->GetSwapchain()->GetCurrentFrameIndex());
		pass.barrier(*this, cmd_buffer);
		pass.execute(*this, cmd_buffer, pass.config);
		pass.profiler->End();
		cmd_buffer->EndMarker();
		cmd_buffer->End();
		cmd_buffers.push_back(cmd_buffer);
	}
	p_rhi_context->GetQueue(RHIQueueFamily::Graphics)->Submit(cmd_buffers);
}

const std::vector<RenderGraph::RenderPassInfo> &RenderGraph::GetRenderPasses() const
{
	return m_render_passes;
}

RenderGraph &RenderGraph::AddPass(const std::string &name, const rttr::variant &config, RenderTask &&task, BarrierTask &&barrier)
{
	m_render_passes.emplace_back(RenderPassInfo{name, config, std::move(task), std::move(barrier), p_rhi_context->CreateProfiler()});
	return *this;
}

RenderGraph &RenderGraph::AddInitializeBarrier(BarrierTask &&barrier)
{
	m_initialize_barrier = std::move(barrier);
	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const TextureCreateInfo &create_info)
{
	TextureDesc desc    = create_info.desc;
	auto       &texture = m_textures.emplace_back(p_rhi_context->CreateTexture(desc));
	m_texture_lookup.emplace(create_info.handle, texture.get());

	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const std::vector<TextureCreateInfo> &create_infos)
{
	if (create_infos.size() == 1)
	{
		return RegisterTexture(create_infos[0]);
	}

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
	pool_desc.usage       = RHITextureUsage::Transfer | RHITextureUsage::ShaderResource;

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
	}

	m_textures.emplace_back(p_rhi_context->CreateTexture(pool_desc));
	auto pool_texture = m_textures.back().get();

	for (auto &info : create_infos)
	{
		TextureDesc desc = info.desc;
		desc.usage |= RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;
		auto &texture = m_textures.emplace_back(pool_texture->Alias(desc));
		m_texture_lookup.emplace(info.handle, texture.get());
	}

	return *this;
}

RenderGraph &RenderGraph::RegisterBuffer(const BufferCreateInfo &create_info)
{
	BufferDesc desc = create_info.desc;
	desc.usage      = desc.usage | RHIBufferUsage::Transfer | RHIBufferUsage::ConstantBuffer;
	auto &buffer    = m_buffers.emplace_back(p_rhi_context->CreateBuffer(desc));
	m_buffer_lookup.emplace(create_info.handle, buffer.get());

	return *this;
}
}        // namespace Ilum