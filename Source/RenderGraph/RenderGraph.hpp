#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
class CommandBuffer;
class Image;

struct RenderGraphNode
{
	std::string name;
};

class RenderGraph
{
  public:
	using PresentCallback = std::function<void(const CommandBuffer &, const Image *, const Image *)>;
	using CreateCallback  = std::function<void(CommandBuffer &)>;

  public:
	RenderGraph(std::vector<RenderGraphNode> &&nodes, std::unordered_map<std::string, Image> &&attachments, const std::string &output_name, PresentCallback on_present, CreateCallback on_create);

	~RenderGraph();

	RenderGraph(RenderGraph &&) = default;

  private:
	void executeNode(RenderGraphNode &node, const CommandBuffer &command_buffer);

  private:
	std::vector<RenderGraphNode>           m_nodes;
	std::unordered_map<std::string, Image> attachments;
	std::string                            m_output_name;
};
}        // namespace Ilum