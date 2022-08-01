#pragma once

#include "RHIDefinitions.hpp"

namespace Ilum
{
class RHIDevice;
class RHIBuffer;

class RHICommond
{
  public:
	RHICommond(RHIDevice *device, RHIQueueFamily family);
	virtual ~RHICommond() = default;

	std::unique_ptr<RHICommond> Create(RHIDevice *device, RHIQueueFamily family);

	virtual void Begin() = 0;
	virtual void End()   = 0;

	virtual void BeginPass() = 0;
	virtual void EndPass()   = 0;

	virtual void BindVertexBuffer() = 0;
	virtual void BindIndexBuffer()  = 0;

	// Drawcall
	virtual void Dispatch(uint32_t group_x = 1, uint32_t group_y = 1, uint32_t group_z = 1)                                       = 0;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0) = 0;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) = 0;

  protected:
	RHIDevice     *p_device = nullptr;
	RHIQueueFamily m_family;
};
}        // namespace Ilum