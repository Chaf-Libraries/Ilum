#include "RenderGraph.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

namespace Ilum
{
RenderGraph::RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Image> &&attachments, const std::string &output_name, const std::string &view_name, PresentCallback on_present, CreateCallback on_create) :
    m_nodes(std::move(nodes)), m_attachments(std::move(attachments)), m_output(output_name), m_view(view_name), onPresent(on_present), onCreate(on_create)
{
}

RenderGraph::~RenderGraph()
{
	GraphicsContext::instance()->getQueueSystem().waitAll();

	for (auto &node : m_nodes)
	{
		if (node.pass_native.frame_buffer)
		{
			vkDestroyFramebuffer(GraphicsContext::instance()->getLogicalDevice(), node.pass_native.frame_buffer, nullptr);
		}
		if (node.pass_native.pipeline)
		{
			vkDestroyPipeline(GraphicsContext::instance()->getLogicalDevice(), node.pass_native.pipeline, nullptr);
		}
		if (node.pass_native.pipeline_layout)
		{
			vkDestroyPipelineLayout(GraphicsContext::instance()->getLogicalDevice(), node.pass_native.pipeline_layout, nullptr);
		}
		if (node.pass_native.render_pass)
		{
			vkDestroyRenderPass(GraphicsContext::instance()->getLogicalDevice(), node.pass_native.render_pass, nullptr);
		}
	}

	reset();
}

bool RenderGraph::empty() const
{
	return m_nodes.empty();
}

void RenderGraph::execute(const CommandBuffer &command_buffer)
{
	initialize();

	ResolveInfo resolve;
	for (const auto &[name, attachment] : m_attachments)
	{
		resolve.resolve(name, attachment);
	}

	for (auto &node : m_nodes)
	{
		executeNode(node, command_buffer, resolve);
	}

	// TODO: Multi-threading not working
	/*std::vector<std::future<VkSubmitInfo>> futures;
	std::vector<VkSubmitInfo> submit_infos;

	for (auto &node : m_nodes)
	{
		futures.push_back(ThreadPool::instance()->addTask([this, &node](size_t) { 
			ResolveInfo resolve;
			for (const auto &[name, attachment] : m_attachments)
			{
				resolve.resolve(name, attachment);
			}
			return executeNode(node, resolve); }));
	}

	for (auto& queue : m_queues)
	{
		queue->waitIdle();
	}
	m_queues.clear();
	for (auto& future : futures)
	{
		m_queues.push_back(GraphicsContext::instance()->getQueueSystem().acquire(QueueUsage::Graphics));
		vkQueueSubmit(*m_queues.back(), 1, &future.get(), VK_NULL_HANDLE);
	}*/
}

void RenderGraph::present(const CommandBuffer &command_buffer, const Image &present_image)
{
	onPresent(command_buffer, m_attachments.at(m_output), present_image);
}

const std::vector<RenderGraphNode> &RenderGraph::getNodes() const
{
	return m_nodes;
}

std::vector<RenderGraphNode> &RenderGraph::getNodes()
{
	return m_nodes;
}

const RenderGraphNode &RenderGraph::getNode(const std::string &name) const
{
	auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [&name](const RenderGraphNode &node) { return node.name == name; });
	ASSERT(iter != m_nodes.end());
	return *iter;
}

RenderGraphNode &RenderGraph::getNode(const std::string &name)
{
	auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [&name](const RenderGraphNode &node) { return node.name == name; });
	ASSERT(iter != m_nodes.end());
	return *iter;
}

const Image &RenderGraph::getAttachment(const std::string &name) const
{
	return m_attachments.at(name);
}

const std::unordered_map<std::string, Image> &RenderGraph::getAttachments() const
{
	return m_attachments;
}

bool RenderGraph::hasAttachment(const std::string &name) const
{
	return m_attachments.find(name) != m_attachments.end();
}

bool RenderGraph::hasRenderPass(const std::string &name) const
{
	auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [&name](const RenderGraphNode &node) { return node.name == name; });
	return iter != m_nodes.end();
}

void RenderGraph::reset()
{
	m_nodes.clear();
	m_attachments.clear();
	m_output      = "";
	m_initialized = false;
	onPresent     = {};
	onCreate      = {};
}

const std::string &RenderGraph::output() const
{
	return m_output;
}

const std::string &RenderGraph::view() const
{
	return m_view;
}

void RenderGraph::initialize()
{
	if (!m_initialized)
	{
		CommandBuffer command_buffer;
		command_buffer.begin();
		onCreate(command_buffer);
		command_buffer.end();
		command_buffer.submitIdle();
		m_initialized = true;
	}
}

void RenderGraph::executeNode(RenderGraphNode &node, const CommandBuffer &command_buffer, ResolveInfo &resolve)
{
	RenderPassState state{*this, command_buffer, node.pass_native};

	node.pass->resolveResources(resolve);
	node.descriptors.resolve(resolve);
	node.descriptors.write(node.pass_native.descriptor_sets);

	// Insert pipeline barrier
	node.pipeline_barrier_callback(command_buffer, resolve);

	if (command_buffer.beginRenderPass(state.pass))
	{
		node.pass->render(state);
		command_buffer.endRenderPass();
	}
}

VkSubmitInfo RenderGraph::executeNode(RenderGraphNode &node, ResolveInfo &resolve)
{
	// TODO: Multi thread rendering
	//vkWaitForFences(GraphicsContext::instance()->getLogicalDevice(), 1, &node.submit_info.fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
	//vkResetFences(GraphicsContext::instance()->getLogicalDevice(), 1, &node.submit_info.fence);

	auto &command_buffer = GraphicsContext::instance()->acquireCommandBuffer(node.pass_native.bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? QueueUsage::Compute : QueueUsage::Graphics);
	command_buffer.begin();
	VK_Debugger::setName(command_buffer, ("Command Buffer - " + node.name).c_str());
	if (node.submit_info.signal_semaphore)
	{
		VK_Debugger::setName(node.submit_info.signal_semaphore, ("Semaphore - " + node.name).c_str());
	}
	RenderPassState state{*this, command_buffer, node.pass_native};

	node.pass->resolveResources(resolve);
	node.descriptors.resolve(resolve);
	node.descriptors.write(node.pass_native.descriptor_sets);

	// Insert pipeline barrier
	node.pipeline_barrier_callback(command_buffer, resolve);

	if (command_buffer.beginRenderPass(state.pass))
	{
		node.pass->render(state);
		command_buffer.endRenderPass();
	}

	command_buffer.end();

	VkSubmitInfo submit_info         = {};
	submit_info.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &command_buffer.getCommandBuffer();
	submit_info.signalSemaphoreCount = node.submit_info.signal_semaphore ? 1 : 0;
	submit_info.pSignalSemaphores    = &node.submit_info.signal_semaphore;
	submit_info.waitSemaphoreCount   = static_cast<uint32_t>(node.submit_info.wait_semaphores.size());
	submit_info.pWaitSemaphores      = node.submit_info.wait_semaphores.empty() ? nullptr : node.submit_info.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = node.submit_info.wait_stages.empty() ? nullptr : node.submit_info.wait_stages.data();

	auto *queue = GraphicsContext::instance()->getQueueSystem().acquire(QueueUsage::Graphics);

	//vkQueueSubmit(*queue, 1, &submit_info, VK_NULL_HANDLE);

	return submit_info;
}
}        // namespace Ilum