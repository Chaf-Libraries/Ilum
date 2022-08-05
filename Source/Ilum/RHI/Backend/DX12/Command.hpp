#pragma once

#include "RHI/RHICommand.hpp"

#include <vector>

#include <directx/d3d12.h>
#include <directx/dxgicommon.h>
#include <wrl.h>

using Microsoft::WRL::ComPtr;

namespace Ilum::DX12
{
class Command : public RHICommand
{
  public:
	Command(RHIDevice *device, RHIQueueFamily family);

	virtual ~Command() override;

	virtual void Begin() override;
	virtual void End() override;

	virtual void BeginPass() override;
	virtual void EndPass() override;

	virtual void BindVertexBuffer() override;
	virtual void BindIndexBuffer() override;

	virtual void BindPipeline(RHIPipelineState *pipeline_state, RHIDescriptor *descriptor) override;

	// Drawcall
	virtual void Dispatch(uint32_t group_x, uint32_t group_y, uint32_t group_z) override;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance) override;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance) override;

	// Resource
	virtual void ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions) override;

  private:
	ComPtr<ID3D12GraphicsCommandList6> m_handle = nullptr;
};
}        // namespace Ilum::DX12