#include "Command.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"

namespace Ilum::DX12
{
Command::Command(RHIDevice *device, RHIQueueFamily family) :
    RHICommand(device, family)
{
}

Command::~Command()
{
}

void Command::Begin()
{
	m_state = CommandState::Recording;
}

void Command::End()
{
	m_handle->Close();
	m_state = CommandState::Executable;
}

void Command::BeginRenderPass(RHIRenderTarget* render_target)
{

}

void Command::EndRenderPass()
{

}

void Command::BindVertexBuffer()
{

}

void Command::BindIndexBuffer()
{

}

void Command::BindDescriptor(RHIDescriptor* descriptor)
{

}

void Command::BindPipelineState(RHIPipelineState* pipeline_state)
{

}

void Command::SetViewport(float width, float height, float x, float y)
{

}

void Command::SetScissor(uint32_t width, uint32_t height, int32_t offset_x, int32_t offset_y)
{

}

void Command::Dispatch(uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
}

void Command::Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
}

void Command::DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
}

void Command::ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions)
{
	std::vector<D3D12_RESOURCE_BARRIER> resource_barriers;

	resource_barriers.reserve(texture_transitions.size() + buffer_transitions.size());

	for (auto &texture_transition : texture_transitions)
	{
		if (texture_transition.src != texture_transition.dst)
		{
			D3D12_RESOURCE_BARRIER barrier = {};
			barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			barrier.Transition.pResource   = static_cast<Texture *>(texture_transition.texture)->GetHandle();
			barrier.Transition.StateBefore = TextureState::Create(texture_transition.src).state;
			barrier.Transition.StateAfter  = TextureState::Create(texture_transition.dst).state;
			// barrier.Transition.Subresource = MipSlice + (ArraySlice * MipLevels) + (PlaneSlice * MipLevels * ArraySize);
			resource_barriers.push_back(barrier);
		}
	}
	for (auto &buffer_transition : buffer_transitions)
	{
		if (buffer_transition.src != buffer_transition.dst)
		{
			D3D12_RESOURCE_BARRIER barrier = {};
			barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
			barrier.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
			barrier.Transition.pResource   = static_cast<Buffer *>(buffer_transition.buffer)->GetHandle();
			barrier.Transition.StateBefore = BufferState::Create(buffer_transition.src).state;
			barrier.Transition.StateAfter  = BufferState::Create(buffer_transition.dst).state;
			barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
			resource_barriers.push_back(barrier);
		}
	}

	m_handle->ResourceBarrier(static_cast<uint32_t>(resource_barriers.size()), resource_barriers.data());
}
}        // namespace Ilum::DX12