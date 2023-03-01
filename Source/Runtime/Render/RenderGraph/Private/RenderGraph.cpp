#include "RenderGraph.hpp"

#include <Core/JobSystem.hpp>

namespace Ilum
{
RenderGraphDesc &RenderGraphDesc::SetName(const std::string &name)
{
	m_name = name;
	return *this;
}

RenderGraphDesc &RenderGraphDesc::AddPass(size_t handle, RenderPassDesc &&desc)
{
	for (auto &[pin_handle, pin] : desc.GetPins())
	{
		m_pass_lookup[pin_handle] = handle;
	}

	m_pass_lookup[handle] = handle;
	m_pass_lookup[handle] = handle;
	m_pass_lookup[handle] = handle;
	desc.SetHandle(handle);
	m_passes.emplace(handle, std::move(desc));
	return *this;
}

void RenderGraphDesc::ErasePass(size_t handle)
{
	auto &desc = m_passes.at(handle);
	auto &pins = desc.GetPins();

	for (auto iter = m_edges.begin(); iter != m_edges.end();)
	{
		if (pins.find(iter->first) != pins.end() ||
		    pins.find(iter->second) != pins.end() ||
		    iter->first == handle ||
		    iter->second == handle)
		{
			iter = m_edges.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	for (auto &[handle, name] : pins)
	{
		m_pass_lookup.erase(handle);
	}

	m_passes.erase(handle);
}

void RenderGraphDesc::EraseLink(size_t source, size_t target)
{
	if (m_edges.find(target) != m_edges.end() &&
	    m_edges.at(target) == source)
	{
		m_edges.erase(target);
	}
}

RenderGraphDesc &RenderGraphDesc::Link(size_t source, size_t target)
{
	const auto &src_node = m_passes.at(m_pass_lookup.at(source));
	const auto &dst_node = m_passes.at(m_pass_lookup.at(target));

	if ((source == src_node.GetHandle()) &&
	    (target == dst_node.GetHandle()))
	{
		// Link Node
		m_edges[target] = source;
	}
	else if ((source != src_node.GetHandle()) &&
	         (target != dst_node.GetHandle()))
	{
		// Link Pin
		const auto &src_pin = src_node.GetPin(source);
		const auto &dst_pin = dst_node.GetPin(target);

		if ((src_pin.type == dst_pin.type) &&
		    src_pin.attribute != dst_pin.attribute)
		{
			m_edges[target] = source;
		}
	}

	return *this;
}

bool RenderGraphDesc::HasLink(size_t target) const
{
	return m_edges.find(target) != m_edges.end();
}

size_t RenderGraphDesc::LinkFrom(size_t target) const
{
	return m_edges.at(target);
}

std::set<size_t> RenderGraphDesc::LinkTo(size_t source) const
{
	std::set<size_t> result;
	for (auto &[dst, src] : m_edges)
	{
		if (src == source)
		{
			result.insert(dst);
		}
	}
	return result;
}

bool RenderGraphDesc::HasPass(size_t handle) const
{
	return m_passes.find(handle) != m_passes.end();
}

RenderPassDesc &RenderGraphDesc::GetPass(size_t handle)
{
	return m_passes.at(m_pass_lookup.at(handle));
}

const std::string &RenderGraphDesc::GetName() const
{
	return m_name;
}

std::map<size_t, RenderPassDesc> &RenderGraphDesc::GetPasses()
{
	return m_passes;
}

const std::map<size_t, size_t> &RenderGraphDesc::GetEdges() const
{
	return m_edges;
}

void RenderGraphDesc::Clear()
{
	m_passes.clear();
	m_edges.clear();
	m_pass_lookup.clear();
}

struct RenderGraph::Impl
{
	RHIContext *rhi_context = nullptr;

	InitializeBarrierTask initialize_barrier;

	std::vector<RenderPassInfo> render_passes;

	std::vector<std::unique_ptr<RHITexture>> textures;
	std::map<size_t, RHITexture *>           texture_lookup;
	std::map<size_t, RHITexture *>           cuda_textures;

	std::vector<std::unique_ptr<RHIBuffer>> buffers;
	std::map<size_t, RHIBuffer *>           buffer_lookup;

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

RHITexture *RenderGraph::GetTexture(size_t handle)
{
	auto iter = m_impl->texture_lookup.find(handle);
	return iter == m_impl->texture_lookup.end() ? nullptr : iter->second;
}

RHIBuffer *RenderGraph::GetBuffer(size_t handle)
{
	auto iter = m_impl->buffer_lookup.find(handle);
	return iter == m_impl->buffer_lookup.end() ? nullptr : iter->second;
}

RHITexture *RenderGraph::GetCUDATexture(size_t handle)
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
		auto *graphics_cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Graphics);
		auto *compute_cmd_buffer  = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute);
		graphics_cmd_buffer->Begin();
		compute_cmd_buffer->Begin();
		graphics_cmd_buffer->BeginMarker("Initialize - Graphics Queue");
		compute_cmd_buffer->BeginMarker("Initialize - Compute Queue");
		m_impl->initialize_barrier(*this, graphics_cmd_buffer, compute_cmd_buffer);
		m_impl->init = true;
		graphics_cmd_buffer->EndMarker();
		compute_cmd_buffer->EndMarker();
		graphics_cmd_buffer->End();
		compute_cmd_buffer->End();
		m_impl->rhi_context->Execute({graphics_cmd_buffer});
		m_impl->rhi_context->Execute({compute_cmd_buffer});
	}

	// Collect cmd buffers
	/*std::vector<std::future<RHICommand *>> cmd_buffer_futures;
	for (auto &pass : m_impl->render_passes)
	{
	    auto future = JobSystem::GetInstance().ExecuteAsync([&]() {
	        if (pass.bind_point == BindPoint::CUDA)
	        {
	            auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute, true);
	            cmd_buffer->Begin();
	            pass.profiler->Begin(cmd_buffer, m_impl->rhi_context->GetSwapchain()->GetCurrentFrameIndex());
	            pass.execute(*this, cmd_buffer, pass.config, black_board);
	            pass.profiler->End(cmd_buffer);
	            cmd_buffer->End();
	            return cmd_buffer;
	        }
	        else
	        {
	            RHIQueueFamily family = RHIQueueFamily::Graphics;
	            if (pass.bind_point == BindPoint::Rasterization)
	            {
	                family = RHIQueueFamily::Graphics;
	            }
	            else
	            {
	                family = RHIQueueFamily::Compute;
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
	            return cmd_buffer;
	        }
	    });
	    cmd_buffer_futures.emplace_back(std::move(future));
	}

	if (!cmd_buffer_futures.empty())
	{
	    std::vector<RHICommand *> cmd_buffers(cmd_buffer_futures.size());
	    std::transform(cmd_buffer_futures.begin(), cmd_buffer_futures.end(), cmd_buffers.begin(), [](std::future<RHICommand *> &iter) { return iter.get(); });

	    RHIQueueFamily            last_queue_family = cmd_buffers[0]->GetQueueFamily();
	    std::string               last_backend      = cmd_buffers[0]->GetBackend();
	    RHISemaphore             *last_semaphore    = nullptr;
	    std::vector<RHICommand *> submit_cmd_buffers;
	    for (auto &cmd_buffer : cmd_buffers)
	    {
	        if (last_queue_family != cmd_buffer->GetQueueFamily() ||
	            cmd_buffer->GetBackend() != last_backend &&
	                !submit_cmd_buffers.empty())
	        {
	            RHISemaphore *pass_semaphore   = m_impl->rhi_context->CreateFrameSemaphore();
	            RHISemaphore *signal_semaphore = last_backend == "CUDA" ? MapToCUDASemaphore(pass_semaphore) : pass_semaphore;
	            RHISemaphore *wait_semaphore   = last_semaphore ? last_backend == "CUDA" ? MapToCUDASemaphore(last_semaphore) : last_semaphore : nullptr;

	            m_impl->rhi_context->Submit(std::move(submit_cmd_buffers), wait_semaphore ? std::vector<RHISemaphore *>{wait_semaphore} : std::vector<RHISemaphore *>{}, {signal_semaphore});
	            submit_cmd_buffers.clear();
	            last_semaphore    = pass_semaphore;
	            last_queue_family = cmd_buffer->GetQueueFamily();
	        }

	        submit_cmd_buffers.push_back(cmd_buffer);
	    }
	    if (!submit_cmd_buffers.empty())
	    {
	        m_impl->rhi_context->Submit(std::move(submit_cmd_buffers), last_semaphore ? std::vector<RHISemaphore *>{last_semaphore} : std::vector<RHISemaphore *>{}, {});
	    }
	}*/

	std::vector<RHICommand *> cmd_buffers;
	for (auto &pass : m_impl->render_passes)
	{
		if (pass.bind_point == BindPoint::CUDA)
		{
			auto *cmd_buffer = m_impl->rhi_context->CreateCommand(RHIQueueFamily::Compute, true);
			cmd_buffer->Begin();
			pass.profiler->Begin(cmd_buffer, m_impl->rhi_context->GetSwapchain()->GetCurrentFrameIndex());
			pass.execute(*this, cmd_buffer, pass.config, black_board);
			pass.profiler->End(cmd_buffer);
			cmd_buffer->End();
			cmd_buffers.emplace_back(cmd_buffer);
		}
		else
		{
			RHIQueueFamily family = RHIQueueFamily::Graphics;
			if (pass.bind_point == BindPoint::Rasterization)
			{
				family = RHIQueueFamily::Graphics;
			}
			else
			{
				family = RHIQueueFamily::Compute;
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
			cmd_buffers.emplace_back(cmd_buffer);
		}
	}

	if (!cmd_buffers.empty())
	{
		RHIQueueFamily            last_queue_family = cmd_buffers[0]->GetQueueFamily();
		std::string               last_backend      = cmd_buffers[0]->GetBackend();
		RHISemaphore             *last_semaphore    = nullptr;
		std::vector<RHICommand *> submit_cmd_buffers;
		for (auto &cmd_buffer : cmd_buffers)
		{
			if (last_queue_family != cmd_buffer->GetQueueFamily() ||
			    cmd_buffer->GetBackend() != last_backend &&
			        !submit_cmd_buffers.empty())
			{
				RHISemaphore *pass_semaphore   = m_impl->rhi_context->CreateFrameSemaphore();
				RHISemaphore *signal_semaphore = last_backend == "CUDA" ? MapToCUDASemaphore(pass_semaphore) : pass_semaphore;
				RHISemaphore *wait_semaphore   = last_semaphore ? last_backend == "CUDA" ? MapToCUDASemaphore(last_semaphore) : last_semaphore : nullptr;

				m_impl->rhi_context->Submit(std::move(submit_cmd_buffers), wait_semaphore ? std::vector<RHISemaphore *>{wait_semaphore} : std::vector<RHISemaphore *>{}, {signal_semaphore});
				submit_cmd_buffers.clear();
				last_semaphore    = pass_semaphore;
				last_queue_family = cmd_buffer->GetQueueFamily();
			}

			submit_cmd_buffers.push_back(cmd_buffer);
		}
		if (!submit_cmd_buffers.empty())
		{
			m_impl->rhi_context->Submit(std::move(submit_cmd_buffers), last_semaphore ? std::vector<RHISemaphore *>{last_semaphore} : std::vector<RHISemaphore *>{}, {});
		}
	}
}

const std::vector<RenderGraph::RenderPassInfo> &RenderGraph::GetRenderPasses() const
{
	return m_impl->render_passes;
}

RenderGraph &RenderGraph::AddPass(
    const std::string &name,
    const std::string &category,
    BindPoint          bind_point,
    const Variant     &config,
    RenderTask       &&task,
    BarrierTask      &&barrier)
{
	m_impl->render_passes.emplace_back(RenderPassInfo{
	    name,
	    category,
	    bind_point,
	    config,
	    std::move(task),
	    std::move(barrier),
	    m_impl->rhi_context->CreateProfiler(bind_point == BindPoint::CUDA)});
	return *this;
}

RenderGraph &RenderGraph::AddInitializeBarrier(InitializeBarrierTask &&barrier)
{
	m_impl->initialize_barrier = std::move(barrier);
	return *this;
}

RenderGraph &RenderGraph::RegisterTexture(const TextureCreateInfo &create_info)
{
	TextureDesc desc    = create_info.desc;
	auto       &texture = m_impl->textures.emplace_back(m_impl->rhi_context->CreateTexture(desc));
	for (auto &handle : create_info.handles)
	{
		m_impl->texture_lookup.emplace(handle, texture.get());
	}
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
		for (auto &handle : info.handles)
		{
			m_impl->texture_lookup.emplace(handle, texture.get());
		}
	}

	return *this;
}

RenderGraph &RenderGraph::RegisterBuffer(const BufferCreateInfo &create_info)
{
	BufferDesc desc = create_info.desc;
	desc.usage      = desc.usage | RHIBufferUsage::Transfer | RHIBufferUsage::ConstantBuffer;
	auto &buffer    = m_impl->buffers.emplace_back(m_impl->rhi_context->CreateBuffer(desc));
	for (auto &handle : create_info.handles)
	{
		m_impl->buffer_lookup.emplace(handle, buffer.get());
	}
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