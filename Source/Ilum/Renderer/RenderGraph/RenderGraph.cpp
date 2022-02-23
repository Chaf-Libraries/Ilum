#include "RenderGraph.hpp"

#include "Device/LogicalDevice.hpp"
#include "Device/Swapchain.hpp"

#include "Threading/ThreadPool.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"
#include "Graphics/RenderFrame.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include <imgui.h>

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
		for (auto &query_pool : node.pass_native.query_pools)
		{
			vkDestroyQueryPool(GraphicsContext::instance()->getLogicalDevice(), query_pool, nullptr);
		}
	}

	reset();
}

bool RenderGraph::empty() const
{
	return m_nodes.empty();
}

void RenderGraph::execute()
{
	initialize();

	for (auto &node : m_nodes)
	{
		node.pass->onUpdate();
	}

	if (m_multi_threading)
	{
		std::vector<std::future<VkCommandBuffer>> cmd_buffers;
		for (auto &node : m_nodes)
		{
			cmd_buffers.push_back(ThreadPool::instance()->addTask([&node, this](size_t) {
				auto &cmd_buffer = GraphicsContext::instance()->getFrame().requestCommandBuffer();
				cmd_buffer.begin();
				executeNode(node, cmd_buffer, m_resolve_info);
				cmd_buffer.end();
				return cmd_buffer.getCommandBuffer();
			}));
		}

		for (auto &future : cmd_buffers)
		{
			GraphicsContext::instance()->submitCommandBuffer(future.get());
		}
	}
	else
	{
		for (auto &node : m_nodes)
		{
			auto &cmd_buffer = GraphicsContext::instance()->getFrame().requestCommandBuffer();
			cmd_buffer.begin();
			executeNode(node, cmd_buffer, m_resolve_info);
			cmd_buffer.end();

			GraphicsContext::instance()->submitCommandBuffer(cmd_buffer);
		}
	}
}

void RenderGraph::present(const CommandBuffer &command_buffer, const Image &present_image)
{
	onPresent(command_buffer, m_attachments.at(m_output), present_image);
}

void RenderGraph::onImGui()
{
	ImGui::Checkbox("Enable Multi-Threading", &m_multi_threading);
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

		for (const auto &[name, attachment] : m_attachments)
		{
			m_resolve_info.resolve(name, attachment);
		}

		for (auto &node : m_nodes)
		{
			node.pass->resolveResources(m_resolve_info);
		}
	}
}

void RenderGraph::executeNode(RenderGraphNode &node, const CommandBuffer &command_buffer, ResolveInfo &resolve)
{
	RenderPassState state{*this, command_buffer, node.pass_native};

	/*node.pass->resolveResources(resolve);*/

	if (node.descriptors.getOption() != ResolveOption::None)
	{
		node.descriptors.resolve(resolve);
		node.descriptors.write(node.pass_native.descriptor_sets);
	}

	// Insert pipeline barrier
	node.pipeline_barrier_callback(command_buffer, resolve);

	node.pass->beginProfile(state);
	node.pass->render(state);
	node.pass->endProfile(state);
}
}        // namespace Ilum