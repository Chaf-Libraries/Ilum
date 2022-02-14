#include "RenderGraph.hpp"

#include <Graphics/Device/Device.hpp>
#include "Device/Swapchain.hpp"

#include <Core/JobSystem/JobSystem.hpp>

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include <Graphics/Vulkan.hpp>
#include <Graphics/RenderContext.hpp>

namespace Ilum
{
RenderGraph::RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Graphics::Image> &&attachments, const std::string &output_name, const std::string &view_name, PresentCallback on_present, CreateCallback on_create) :
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
			vkDestroyFramebuffer(Graphics::RenderContext::GetDevice(), node.pass_native.frame_buffer, nullptr);
		}
		if (node.pass_native.pipeline)
		{
			vkDestroyPipeline(Graphics::RenderContext::GetDevice(), node.pass_native.pipeline, nullptr);
		}
		if (node.pass_native.pipeline_layout)
		{
			vkDestroyPipelineLayout(Graphics::RenderContext::GetDevice(), node.pass_native.pipeline_layout, nullptr);
		}
		if (node.pass_native.render_pass)
		{
			vkDestroyRenderPass(Graphics::RenderContext::GetDevice(), node.pass_native.render_pass, nullptr);
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
		GraphicsContext::instance()->getProfiler().beginSample(node.name, command_buffer);
		executeNode(node, command_buffer, resolve);
		GraphicsContext::instance()->getProfiler().endSample(node.name, command_buffer);
	}
}

void RenderGraph::present(const CommandBuffer &command_buffer, const Graphics::Image &present_image)
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

const Graphics::Image &RenderGraph::getAttachment(const std::string &name) const
{
	return m_attachments.at(name);
}

const std::unordered_map<std::string, Graphics::Image> &RenderGraph::getAttachments() const
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
	if (node.descriptors.getOption() != ResolveOption::None)
	{
		node.descriptors.resolve(resolve);
		node.descriptors.write(node.pass_native.descriptor_sets);
	}

	// Insert pipeline barrier
	node.pipeline_barrier_callback(command_buffer, resolve);

	node.pass->render(state);
}
}        // namespace Ilum