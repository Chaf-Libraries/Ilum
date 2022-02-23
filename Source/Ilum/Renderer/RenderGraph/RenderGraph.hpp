#pragma once

#include "Utils/PCH.hpp"

#include "RenderPass.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Synchronization/Queue.hpp"

namespace Ilum
{
class Image;

struct RenderGraphNode
{
	std::string                                                     name;
	PassNative                                                      pass_native;
	scope<RenderPass>                                               pass;
	std::vector<std::string>                                        attachments;
	std::function<void(const CommandBuffer &, const ResolveInfo &)> pipeline_barrier_callback;
	DescriptorBinding                                               descriptors;
	SubmitInfo                                                      submit_info;
};

class RenderGraph
{
  private:
	using PresentCallback = std::function<void(const CommandBuffer &, const Image &, const Image &)>;
	using CreateCallback  = std::function<void(const CommandBuffer &)>;

  public:
	RenderGraph() = default;

	RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Image> &&attachments, const std::string &output_name, const std::string &view_name, PresentCallback on_present, CreateCallback on_create);

	~RenderGraph();

	RenderGraph(RenderGraph &&) = default;

	bool empty() const;

	void execute();

	void present(const CommandBuffer &command_buffer, const Image &present_image);

	void onImGui();

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

	const Image &getAttachment(const std::string &name) const;

	const std::unordered_map<std::string, Image> &getAttachments() const;

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

	void executeNode(RenderGraphNode &node, const CommandBuffer &command_buffer, ResolveInfo &resolve);

  private:
	std::vector<RenderGraphNode>           m_nodes;
	std::unordered_map<std::string, Image> m_attachments;
	std::string                            m_output      = "output";
	std::string                            m_view        = "view";
	bool                                   m_initialized = false;
	PresentCallback                        onPresent;
	CreateCallback                         onCreate;
	std::vector<Queue *>                   m_queues;
	ResolveInfo                            m_resolve_info;
	bool                                   m_multi_threading = true;
};
}        // namespace Ilum