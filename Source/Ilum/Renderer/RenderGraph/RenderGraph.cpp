#include "RenderGraph.hpp"

#include "Device/LogicalDevice.hpp"

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

	std::unordered_map<std::string, ResolveInfo> resolves;
	for (const auto &[name, attachment] : m_attachments)
	{
		for (auto &node : m_nodes)
		{
			resolves[node.name].resolve(name, attachment);
		}
	}

	std::vector<std::future<void>> futures;

	for (auto &node : m_nodes)
	{
		futures.push_back(ThreadPool::instance()->addTask([this, &node, &resolves, &command_buffer](size_t) {
			executeNode(node, command_buffer, resolves[node.name]);
		}));
	}

	for (auto &future : futures)
	{
		future.get();
	}
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
	//RenderPassState state{*this, command_buffer, node.pass_native};

	//node.pass->resolveResources(resolve);
	//node.descriptors.resolve(resolve);
	//node.descriptors.write(node.pass_native.descriptor_sets);

	// Insert pipeline barrier

	QueueUsage usage = node.pass_native.bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? QueueUsage::Compute : QueueUsage::Graphics;

	auto &          cmd_buffer = GraphicsContext::instance()->acquireCommandBuffer(usage);
	VK_Debugger::setName(cmd_buffer, node.name.c_str());
	RenderPassState state{*this, cmd_buffer, node.pass_native};

	node.pass->resolveResources(resolve);
	node.descriptors.resolve(resolve);
	node.descriptors.write(node.pass_native.descriptor_sets);

	cmd_buffer.begin();
	node.pipeline_barrier_callback(cmd_buffer, resolve);
	if (cmd_buffer.beginRenderPass(state.pass))
	{
		node.pass->render(state);
		cmd_buffer.endRenderPass();
	}
	cmd_buffer.end();
	GraphicsContext::instance()->getQueueSystem().acquire(usage)->submit(cmd_buffer, node.submit_info);

	//node.pipeline_barrier_callback(command_buffer, resolve);

	//if (command_buffer.beginRenderPass(state.pass))
	//{
	//	node.pass->render(state);
	//	command_buffer.endRenderPass();
	//}
}
}        // namespace Ilum