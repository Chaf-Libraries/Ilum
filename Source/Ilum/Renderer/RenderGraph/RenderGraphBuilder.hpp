#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Synchronization/Queue.hpp"

namespace Ilum
{
namespace Graphics
{
class Image;
class Buffer;
};        // namespace Graphics

class RenderPass;
class RenderGraph;
struct RenderGraphNode;
class PipelineState;
class CommandBuffer;
class ResolveInfo;
struct PassNative;

struct ImageTransition
{
	VkImageUsageFlagBits initial_usage;
	VkImageUsageFlagBits final_usage;
};

struct BufferTransition
{
	VkBufferUsageFlagBits initial_usage;
	VkBufferUsageFlagBits final_usage;
};

class RenderGraphBuilder
{
  public:
	RenderGraphBuilder() = default;

	~RenderGraphBuilder() = default;

	RenderGraphBuilder &addRenderPass(const std::string &name, std::unique_ptr<RenderPass> render_pass);

	RenderGraphBuilder &setOutput(const std::string &name);

	RenderGraphBuilder &setView(const std::string &name);

	scope<RenderGraph> build();

	const std::string &output() const;

	const std::string &view() const;

	void reset();

	bool empty() const;

  private:
	struct RenderPassReference
	{
		std::string                 name;
		std::unique_ptr<RenderPass> pass;
	};

	template <typename ResourceType, typename TransitionType>
	struct ResourceTypeTransition
	{
		// Pass name - resource type - transition type
		std::unordered_map<std::string, std::unordered_map<ResourceType, TransitionType>> transitions;
		std::unordered_map<ResourceType, uint32_t>                                        total_usages;
		std::unordered_map<ResourceType, std::string>                                     first_usages;
		std::unordered_map<ResourceType, std::string>                                     last_usages;
	};

	struct ResourceTransitions
	{
		ResourceTypeTransition<std::string, BufferTransition> buffers;
		ResourceTypeTransition<std::string, ImageTransition>  images;
	};

	// TODO: Semaphore?

	using SynchronizeMap          = std::unordered_map<std::string, SubmitInfo>;
	using AttachmentMap           = std::unordered_map<std::string, Graphics::Image>;
	using PipelineMap             = std::unordered_map<std::string, PipelineState>;
	using PipelineBarrierCallback = std::function<void(const CommandBuffer &, const ResolveInfo &)>;
	using PresentCallback         = std::function<void(const CommandBuffer &, const Graphics::Image &, const Graphics::Image &)>;
	using CreateCallback          = std::function<void(const CommandBuffer &)>;

  private:
	PipelineMap createPipelineStates();

	ResourceTransitions resolveResourceTransitions(const PipelineMap &pipeline_states);

	SynchronizeMap createSynchronizeDependency(const PipelineMap &pipeline_states);

	void setOutputImage(ResourceTransitions &resource_transitions, const std ::string &name);

	AttachmentMap allocateAttachments(const PipelineMap &pipeline_states, const ResourceTransitions &resource_transitions);

	PassNative buildRenderPass(const RenderPassReference &render_pass_reference, const PipelineMap &pipeline_states, const AttachmentMap &attachments, const ResourceTransitions &resource_transitions);

	std::vector<std::string> getRenderPassAttachmentNames(const std::string &render_pass_name, const PipelineMap &pipeline_states);

	PipelineBarrierCallback createPipelineBarrierCallback(const std::string &render_pass_name, const PipelineState &pipeline_state, const ResourceTransitions &resource_transitions);

	CreateCallback createOnCreateCallback(const PipelineMap &pipeline_states, const ResourceTransitions &resource_transitions, const AttachmentMap &attachments);

	PresentCallback createOnPresentCallback(const std::string &output, const ResourceTransitions &resource_transitions);

  private:
	std::vector<RenderPassReference> m_render_pass_references;
	std::string                      m_output = "output";
	std::string                      m_view   = "view";
};
}        // namespace Ilum