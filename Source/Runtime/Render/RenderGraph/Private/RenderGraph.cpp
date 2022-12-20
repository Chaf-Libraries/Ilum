#include "RenderGraph/RenderGraph.hpp"

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

template <typename Archive>
void RGHandle::serialize(Archive &archive)
{
	archive(handle);
}

template EXPORT_API void RGHandle::serialize<InputArchive>(InputArchive &archive);
template EXPORT_API void RGHandle::serialize<OutputArchive>(OutputArchive &archive);

struct RenderGraph::Impl
{
	RHIContext *rhi_context = nullptr;

	BarrierTask initialize_barrier;

	std::vector<RenderPassInfo> render_passes;

	std::vector<std::unique_ptr<RHITexture>> textures;
	std::map<RGHandle, RHITexture *>         texture_lookup;
	std::map<RGHandle, RHITexture *>         cuda_textures;

	std::vector<std::unique_ptr<RHIBuffer>> buffers;
	std::map<RGHandle, RHIBuffer *>         buffer_lookup;

	std::map<RHISemaphore *, std::unique_ptr<RHISemaphore>> cuda_semaphore_map;

	bool init = false;
};

RenderGraph::RenderGraph(RHIContext *rhi_context)
{
	m_impl              = new Impl;
	m_impl->rhi_context = rhi_context;
}

RenderGraph::~RenderGraph()
{
	m_impl->rhi_context->Reset();
	m_impl->rhi_context->WaitIdle();

	delete m_impl;
}

RHITexture *RenderGraph::GetTexture(RGHandle handle)
{
	auto iter = m_impl->texture_lookup.find(handle);
	return iter == m_impl->texture_lookup.end() ? nullptr : iter->second;
}

RHIBuffer *RenderGraph::GetBuffer(RGHandle handle)
{
	auto iter = m_impl->buffer_lookup.find(handle);
	return iter == m_impl->buffer_lookup.end() ? nullptr : iter->second;
}

RHITexture *RenderGraph::GetCUDATexture(RGHandle handle)
{
	if (m_impl->cuda_textures.find(handle) == m_impl->cuda_textures.end())
	{
		m_impl->textures.emplace_back(m_impl->rhi_context->MapToCUDATexture(m_impl->texture_lookup.at(handle)));
		m_impl->cuda_textures.emplace(handle, m_impl->textures.back().get());
	}
	return m_impl->cuda_textures.at(handle);
}

void RenderGraph::Execute(RenderGraphBlackboard &black_board)
{
	if (m_impl->render_passes.empty())
	{
		return;
	}

	if (!m_impl->init)
	{
		auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		cmd_buffer->Begin();
		cmd_buffer->BeginMarker("Initialize");
		m_impl->initialize_barrier(*this, cmd_buffer);
		m_impl->init = true;
		cmd_buffer->EndMarker();
		cmd_buffer->End();
		m_impl->rhi_context->Submit({cmd_buffer});
	}

	BindPoint      last_bind_point = m_impl->render_passes[0].bind_point;
	RHIQueueFamily family          = RHIQueueFamily::Graphics;

	std::vector<RHICommand *> cmd_buffers;
	cmd_buffers.reserve(m_impl->render_passes.size());
	RHISemaphore *last_semaphore = nullptr;
	for (auto &pass : m_impl->render_passes)
	{
		if (pass.bind_point != last_bind_point && pass.bind_point != BindPoint::None && !cmd_buffers.empty())
		{
			RHISemaphore *pass_semaphore   = m_impl->rhi_context->CreateFrameSemaphore();
			RHISemaphore *signal_semaphore = last_bind_point == BindPoint::CUDA ? MapToCUDASemaphore(pass_semaphore) : pass_semaphore;
			RHISemaphore *wait_semaphore   = last_semaphore ? pass.bind_point == BindPoint::CUDA ? MapToCUDASemaphore(last_semaphore) : last_semaphore : nullptr;

			m_impl->rhi_context->Submit(std::move(cmd_buffers), wait_semaphore ? std::vector<RHISemaphore *>{wait_semaphore} : std::vector<RHISemaphore *>{}, {signal_semaphore});
			last_semaphore = pass_semaphore;
			cmd_buffers.clear();
			last_bind_point = pass.bind_point;
		}

		if (pass.bind_point == BindPoint::CUDA)
		{
			auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute, true);
			cmd_buffer->Begin();
			pass.profiler->Begin(cmd_buffer, m_impl->rhi_context->GetSwapchain()->GetCurrentFrameIndex());
			pass.execute(*this, cmd_buffer, pass.config, black_board);
			pass.profiler->End(cmd_buffer);
			cmd_buffer->End();
			cmd_buffers.push_back(cmd_buffer);
		}
		else
		{
			if (pass.bind_point == BindPoint::Compute ||
			    pass.bind_point == BindPoint::RayTracing)
			{
				family = RHIQueueFamily::Compute;
			}
			else if (pass.bind_point==BindPoint::Rasterization)
			{
				family = RHIQueueFamily::Graphics;
			}

			auto *cmd_buffer = m_impl->rhi_context->CreateCommand(family);
			cmd_buffer->SetName(pass.name);
			cmd_buffer->Begin();
			cmd_buffer->BeginMarker(pass.name);
			pass.profiler->Begin(cmd_buffer, m_impl->rhi_context->GetSwapchain()->GetCurrentFrameIndex());
			pass.barrier(*this, cmd_buffer);
			pass.execute(*this, cmd_buffer, pass.config, black_board);
			pass.profiler->End(cmd_buffer);
			cmd_buffer->EndMarker();
			cmd_buffer->End();
			cmd_buffers.push_back(cmd_buffer);
		}
	}

	if (!cmd_buffers.empty())
	{
		m_impl->rhi_context->Submit(std::move(cmd_buffers), last_semaphore ? std::vector<RHISemaphore *>{last_semaphore} : std::vector<RHISemaphore *>{}, {});
	}
}

const std::vector<RenderGraph::RenderPassInfo> &RenderGraph::GetRenderPasses() const
{
	return m_impl->render_passes;
}

RenderGraph &RenderGraph::AddPass(
    const std::string &name,
    BindPoint          bind_point,
    const Variant     &config,
    RenderTask       &&task,
    BarrierTask      &&barrier)
{
	m_impl->render_passes.emplace_back(RenderPassInfo{
	    name,
	    bind_point,
	    config,
	    std::move(task),
	    std::move(barrier),
	    m_impl->rhi_context->CreateProfiler(bind_point == BindPoint::CUDA)});
	return *this;
}

RenderGraph &RenderGraph::AddInitializeBarrier(BarrierTask &&barrier)
{
	m_impl->initialize_barrier = std::move(barrier);
	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const TextureCreateInfo &create_info)
{
	TextureDesc desc    = create_info.desc;
	auto       &texture = m_impl->textures.emplace_back(m_impl->rhi_context->CreateTexture(desc));
	m_impl->texture_lookup.emplace(create_info.handle, texture.get());

	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const std::vector<TextureCreateInfo> &create_infos)
{
	if (create_infos.size() == 1)
	{
		return RegisterTexture(create_infos[0]);
	}

	TextureDesc pool_desc = {};
	pool_desc.name        = "Pool Texture " + std::to_string(m_impl->textures.size());
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

	m_impl->textures.emplace_back(m_impl->rhi_context->CreateTexture(pool_desc));
	auto pool_texture = m_impl->textures.back().get();

	for (auto &info : create_infos)
	{
		TextureDesc desc = info.desc;
		desc.usage |= RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;
		auto &texture = m_impl->textures.emplace_back(pool_texture->Alias(desc));
		m_impl->texture_lookup.emplace(info.handle, texture.get());
	}

	return *this;
}

RenderGraph &RenderGraph::RegisterBuffer(const BufferCreateInfo &create_info)
{
	BufferDesc desc = create_info.desc;
	desc.usage      = desc.usage | RHIBufferUsage::Transfer | RHIBufferUsage::ConstantBuffer;
	auto &buffer    = m_impl->buffers.emplace_back(m_impl->rhi_context->CreateBuffer(desc));
	m_impl->buffer_lookup.emplace(create_info.handle, buffer.get());
	return *this;
}

RHISemaphore *RenderGraph::MapToCUDASemaphore(RHISemaphore *semaphore)
{
	if (m_impl->cuda_semaphore_map.find(semaphore) != m_impl->cuda_semaphore_map.end())
	{
		return m_impl->cuda_semaphore_map.at(semaphore).get();
	}
	m_impl->cuda_semaphore_map.emplace(semaphore, m_impl->rhi_context->MapToCUDASemaphore(semaphore));
	return m_impl->cuda_semaphore_map.at(semaphore).get();
}
}        // namespace Ilum