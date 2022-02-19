#pragma once

#include "Utils/PCH.hpp"

#include "RenderPass.hpp"

#include "Graphics/Command/CommandBuffer.hpp"


namespace Ilum
{
class Graphics::Image;

struct RenderGraphNode
{
	std::string                                                     name;
	PassNative                                                      pass_native;
	scope<RenderPass>                                               pass;
	std::vector<std::string>                                        attachments;
	std::function<void(const Graphics::CommandBuffer &, const ResolveInfo &)> pipeline_barrier_callback;
	DescriptorBinding                                               descriptors;
};

class RenderGraph
{
  private:
	using PresentCallback = std::function<void(const Graphics::CommandBuffer &, const Graphics::Image &, const Graphics::Image &)>;
	using CreateCallback  = std::function<void(const Graphics::CommandBuffer &)>;

  public:
	RenderGraph() = default;

	RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Graphics::Image> &&attachments, const std::string &output_name, const std::string &view_name, PresentCallback on_present, CreateCallback on_create);

	~RenderGraph();

	RenderGraph(RenderGraph &&) = default;

	bool empty() const;

	void execute(const Graphics::CommandBuffer &command_buffer);

	void present(const Graphics::CommandBuffer &command_buffer, const Graphics::Image &present_image);

	template <typename T>
	const RenderGraphNode &getNode() const
	{
		auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [](const RenderGraphNode &node) { return node.pass->type() == typeid(T); });
		ASSERT(iter != m_nodes.end());
		return *iter;
	}

	template <typename T>
	RenderGraphNode &getNode()
	{
		auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [](const RenderGraphNode &node) { return node.pass->type() == typeid(T); });
		ASSERT(iter != m_nodes.end());
		return *iter;
	}

	const std::vector<RenderGraphNode> &getNodes() const;

	std::vector<RenderGraphNode> &getNodes();

	const RenderGraphNode &getNode(const std::string &name) const;

	RenderGraphNode &getNode(const std::string &name);

	const Graphics::Image &getAttachment(const std::string &name) const;

	const std::unordered_map<std::string, Graphics::Image> &getAttachments() const;

	bool hasAttachment(const std::string &name) const;

	bool hasRenderPass(const std::string &name) const;

	template <typename T>
	bool hasRenderPass() const
	{
		auto iter = std::find_if(m_nodes.begin(), m_nodes.end(), [](const RenderGraphNode &node) { return node.pass->type() == typeid(T); });
		return iter != m_nodes.end();
	}

	template <typename T>
	T &getRenderPass(const std::string &name)
	{
		auto &node = getNode(name);
		ASSERT(node.pass.get() != nullptr);
		return *static_cast<T *>(*node.pass.get());
	}

	template <typename T>
	const T &getRenderPass(const std::string &name) const
	{
		const auto &node = getNode(name);
		ASSERT(node.pass.get() != nullptr);
		return *static_cast<T *>(*node.pass.get());
	}

	void reset();

	const std::string &output() const;

	const std::string &view() const;

  private:
	void initialize();

	void executeNode(RenderGraphNode &node, const Graphics::CommandBuffer &command_buffer, ResolveInfo &resolve);

  private:
	std::vector<RenderGraphNode>           m_nodes;
	std::unordered_map<std::string, Graphics::Image> m_attachments;
	std::string                            m_output      = "output";
	std::string                            m_view        = "view";
	bool                                   m_initialized = false;
	PresentCallback                        onPresent;
	CreateCallback                         onCreate;
};
}        // namespace Ilum