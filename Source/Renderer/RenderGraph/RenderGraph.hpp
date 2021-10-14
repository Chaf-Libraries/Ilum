#pragma once

#include "Utils/PCH.hpp"

#include "RenderPass.hpp"

namespace Ilum
{
class CommandBuffer;
class Image;

struct RenderGraphNode
{
	std::string                                                     name;
	PassNative                                                      pass_native;
	scope<RenderPass>                                               pass;
	std::vector<std::string>                                        attachments;
	std::function<void(const CommandBuffer &, const ResolveInfo &)> pipeline_barrier_callback;
	DescriptorBinding                                               descriptors;
};

class RenderGraph
{
  private:
	using PresentCallback = std::function<void(const CommandBuffer &, const Image &, const Image &)>;
	using CreateCallback  = std::function<void(const CommandBuffer &)>;

  public:
	RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Image> &&attachments, const std::string &output_name, PresentCallback on_present, CreateCallback on_create);

	~RenderGraph();

	RenderGraph(RenderGraph &&) = default;

	void execute(const CommandBuffer &command_buffer);

	void present(const CommandBuffer &command_buffer, const Image &present_image);

	const RenderGraphNode &getNode(const std::string &name) const;

	RenderGraphNode &getNode(const std::string &name);

	const Image &getAttachment(const std::string &name) const;

	bool hasRenderPass(const std::string &name) const;

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

  private:
	void initialize(const CommandBuffer &command_buffer);

	void executeNode(RenderGraphNode &node, const CommandBuffer &command_buffer, ResolveInfo &resolve);

  private:
	std::vector<RenderGraphNode>           m_nodes;
	std::unordered_map<std::string, Image> m_attachments;
	std::string                            m_output;
	bool                                   m_initialized = false;
	PresentCallback                        onPresent;
	CreateCallback                         onCreate;
};
}        // namespace Ilum