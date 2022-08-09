#pragma once

#include "RHIBuffer.hpp"
#include "RHITexture.hpp"

namespace Ilum
{
class RHIDevice;
class RHIBuffer;
class RHIPipelineState;
class RHIDescriptor;

enum class CommandState
{
	Available,
	Initial,
	Recording,
	Executable,
	Pending
};

class RHICommand
{
  public:
	RHICommand(RHIDevice *device, RHIQueueFamily family);
	virtual ~RHICommand() = default;

	CommandState GetState() const;

	void Init();

	static std::unique_ptr<RHICommand> Create(RHIDevice *device, RHIQueueFamily family);

	virtual void Begin() = 0;
	virtual void End()   = 0;

	static void Reset(RHIDevice *device, uint32_t frame_index);

	virtual void BeginPass() = 0;
	virtual void EndPass()   = 0;

	virtual void BindVertexBuffer() = 0;
	virtual void BindIndexBuffer()  = 0;

	// Pipeline & Resource Binding
	virtual void BindPipeline(RHIPipelineState *pipeline_state, RHIDescriptor *descriptor) = 0;

	// Drawcall
	virtual void Dispatch(uint32_t group_x = 1, uint32_t group_y = 1, uint32_t group_z = 1)                                                                        = 0;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0)                                  = 0;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) = 0;

	// Resource Barrier
	virtual void ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions) = 0;

  protected:
	RHIDevice     *p_device = nullptr;
	RHIQueueFamily m_family;
	CommandState   m_state = CommandState::Available;
};
}        // namespace Ilum