#pragma once

#include "Fwd.hpp"

namespace Ilum::CUDA
{
class Command : public RHICommand
{
  public:
	Command(RHIDevice *device, RHIQueueFamily family);

	virtual ~Command();

	virtual void SetName(const std::string &name);

	virtual void Begin() override;
	virtual void End() override;

	virtual void BeginMarker(const std::string &name, float r = 1.f, float g = 1.f, float b = 1.f, float a = 1.f) override;
	virtual void EndMarker() override;

	virtual void BeginRenderPass(RHIRenderTarget *render_target) override;
	virtual void EndRenderPass() override;

	virtual void BindVertexBuffer(uint32_t binding, RHIBuffer *vertex_buffer) override;
	virtual void BindIndexBuffer(RHIBuffer *index_buffer, bool is_short = false) override;

	virtual void BindDescriptor(RHIDescriptor *descriptor) override;
	virtual void BindPipelineState(RHIPipelineState *pipeline_state) override;

	virtual void SetViewport(float width, float height, float x = 0.f, float y = 0.f) override;
	virtual void SetScissor(uint32_t width, uint32_t height, int32_t offset_x = 0, int32_t offset_y = 0) override;

	// Drawcall
	virtual void Dispatch(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) override;
	virtual void DispatchIndirect(RHIBuffer *buffer, size_t offset) override;

	virtual void Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) override;
	virtual void DrawIndirect(RHIBuffer *buffer, size_t offset, uint32_t draw_count, uint32_t stride) override;
	virtual void DrawIndirectCount(RHIBuffer *buffer, size_t offset, RHIBuffer *count_buffer, size_t count_buffer_offset, uint32_t max_draw_count, uint32_t stride) override;

	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance) override;
	virtual void DrawIndexedIndirect(RHIBuffer *buffer, size_t offset, uint32_t draw_count, uint32_t stride) override;
	virtual void DrawIndexedIndirectCount(RHIBuffer *buffer, size_t offset, RHIBuffer *count_buffer, size_t count_buffer_offset, uint32_t max_draw_count, uint32_t stride) override;

	virtual void DrawMeshTask(uint32_t thread_x, uint32_t thread_y, uint32_t thread_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) override;
	virtual void DrawMeshTasksIndirect(RHIBuffer *buffer, size_t offset, uint32_t draw_count, uint32_t stride) override;
	virtual void DrawMeshTasksIndirectCount(RHIBuffer *buffer, size_t offset, RHIBuffer *count_buffer, size_t count_buffer_offset, uint32_t max_draw_count, uint32_t stride) override;

	virtual void TraceRay(uint32_t width, uint32_t height, uint32_t depth) override;

	// Resource Copy
	virtual void CopyBufferToTexture(RHIBuffer *src_buffer, RHITexture *dst_texture, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count) override;
	virtual void CopyTextureToBuffer(RHITexture *src_texture, RHIBuffer *dst_buffer, uint32_t mip_level, uint32_t base_layer, uint32_t layer_count) override;
	virtual void CopyBufferToBuffer(RHIBuffer *src_buffer, RHIBuffer *dst_buffer, size_t size, size_t src_offset, size_t dst_offset) override;

	virtual void GenerateMipmaps(RHITexture *texture, RHIResourceState initial_state, RHIFilter filter) override;
	virtual void BlitTexture(RHITexture *src_texture, const TextureRange &src_range, const RHIResourceState &src_state, RHITexture *dst_texture, const TextureRange &dst_range, const RHIResourceState &dst_state, RHIFilter filter = RHIFilter::Linear) override;

	virtual void FillBuffer(RHIBuffer *buffer, RHIResourceState state, size_t size, size_t offset, uint32_t data) override;
	virtual void FillTexture(RHITexture *texture, RHIResourceState state, const TextureRange &range, const glm::vec4 &color) override;
	virtual void FillTexture(RHITexture *texture, RHIResourceState state, const TextureRange &range, float depth) override;

	// Resource Barrier
	virtual void ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions) override;

	void Execute();

	void Execute(std::function<void(void)> &&task);

	void EventRecord(cudaEvent_t &cuda_event);

	void EventElapsedTime(cudaEvent_t begin, cudaEvent_t end, float &time);

  private:
	RHIDescriptor    *p_descriptor     = nullptr;
	RHIPipelineState *p_pipeline_state = nullptr;

	std::vector<std::function<void()>> m_calls;
};
}        // namespace Ilum::CUDA