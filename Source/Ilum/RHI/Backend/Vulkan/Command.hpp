#pragma once

#include "RHI/RHICommand.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Command : public RHICommand
{
  public:
	Command(RHIDevice *device, uint32_t frame_index, RHIQueueFamily family);
	virtual ~Command() override;

	void SetState(CommandState state);

	static void ResetCommandPool(RHIDevice *device, uint32_t frame_index);

	VkCommandBuffer GetHandle() const;

	virtual void Begin() override;
	virtual void End() override;

	virtual void BeginPass() override;
	virtual void EndPass() override;

	virtual void BindVertexBuffer() override;
	virtual void BindIndexBuffer() override;

	virtual void BindPipeline(RHIPipelineState *pipeline_state, RHIDescriptor *descriptor) override;

	virtual void Dispatch(uint32_t group_x, uint32_t group_y, uint32_t group_z) override;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) override;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance) override;

	virtual void ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions) override;

  private:
	VkCommandBuffer m_handle = VK_NULL_HANDLE;
	VkCommandPool   m_pool   = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan