#pragma once

#include "RHI/RHICommand.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Descriptor;
class RenderTarget;
class PipelineState;

class Command : public RHICommand
{
  public:
	Command(RHIDevice *device, RHIQueueFamily family);

	Command(RHIDevice *device, VkCommandPool pool, RHIQueueFamily family);

	virtual ~Command() override;

	void SetState(CommandState state);

	VkCommandBuffer GetHandle() const;

	virtual void Begin() override;
	virtual void End() override;

	virtual void BeginMarker(const std::string &name, float r, float g, float b, float a) override;
	virtual void EndMarker() override;

	virtual void BeginRenderPass(RHIRenderTarget *render_target) override;
	virtual void EndRenderPass() override;

	virtual void BindVertexBuffer(RHIBuffer *vertex_buffer) override;
	virtual void BindIndexBuffer(RHIBuffer *index_buffer, bool is_short = false) override;

	virtual void BindDescriptor(RHIDescriptor *descriptor) override;
	virtual void BindPipelineState(RHIPipelineState *pipeline_state) override;

	virtual void SetViewport(float width, float height, float x = 0.f, float y = 0.f) override;
	virtual void SetScissor(uint32_t width, uint32_t height, int32_t offset_x = 0, int32_t offset_y = 0) override;

	virtual void Dispatch(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) override;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) override;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance) override;
	virtual void DrawMeshTask(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) override;
	virtual void DrawMeshTask(uint32_t task_count, uint32_t task_offset) override;
	virtual void TraceRay(uint32_t width, uint32_t height, uint32_t depth) override;

	virtual void CopyBufferToTexture(RHIBuffer *src_buffer, RHITexture *dst_texture, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count) override;
	virtual void CopyTextureToBuffer(RHITexture *src_texture, RHIBuffer *dst_buffer, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count) override;
	virtual void CopyBufferToBuffer(RHIBuffer *src_buffer, RHIBuffer *dst_buffer, size_t size, size_t src_offset, size_t dst_offset) override;

	virtual void GenerateMipmaps(RHITexture *texture, RHIResourceState initial_state, RHIFilter filter) override;
	virtual void BlitTexture(RHITexture *src_texture, const TextureRange &src_range, const RHIResourceState &src_state, RHITexture *dst_texture, const TextureRange &dst_range, const RHIResourceState &dst_state, RHIFilter filter) override;

	virtual void ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions) override;

  private:
	VkCommandBuffer m_handle = VK_NULL_HANDLE;
	VkCommandPool   m_pool   = VK_NULL_HANDLE;

	Descriptor    *p_descriptor     = nullptr;
	PipelineState *p_pipeline_state = nullptr;
	RenderTarget  *p_render_target  = nullptr;
};
}        // namespace Ilum::Vulkan