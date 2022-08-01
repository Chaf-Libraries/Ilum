#pragma once

#include "RHI/RHICommond.hpp"

#include <volk.h>

namespace Ilum::Vulkan
{
class Commond : public RHICommond
{
  public:
	Commond(RHIDevice *device, RHIQueueFamily family);
	virtual ~Commond() override;

	VkCommandBuffer GetHandle() const;

	virtual void Begin() override;
	virtual void End()   override;

	virtual void BeginPass() override;
	virtual void EndPass()   override;

	virtual void BindVertexBuffer() override;
	virtual void BindIndexBuffer()  override;

	// Drawcall
	virtual void Dispatch(uint32_t group_x = 1, uint32_t group_y = 1, uint32_t group_z = 1)                                                                        override;
	virtual void Draw(uint32_t vertex_count, uint32_t instance_count = 1, uint32_t first_vertex = 0, uint32_t first_instance = 0)                                  override;
	virtual void DrawIndexed(uint32_t index_count, uint32_t instance_count = 1, uint32_t first_index = 0, uint32_t vertex_offset = 0, uint32_t first_instance = 0) override;

  private:
	VkCommandBuffer m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan