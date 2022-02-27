#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/Descriptor/DescriptorBinding.hpp"

#include "Graphics/RTX/ShaderBindingTable.hpp"

namespace Ilum
{
class PipelineState;
class RenderGraph;

struct PassNative
{
	VkRenderPass               render_pass = VK_NULL_HANDLE;
	std::vector<DescriptorSet> descriptor_sets;
	VkFramebuffer              frame_buffer    = VK_NULL_HANDLE;
	VkPipeline                 pipeline        = VK_NULL_HANDLE;
	VkPipelineLayout           pipeline_layout = VK_NULL_HANDLE;
	std::vector<VkQueryPool>   query_pools;
	VkPipelineBindPoint        bind_point;
	VkRect2D                   render_area;
	std::vector<VkClearValue>  clear_values;

	struct
	{
		std::unique_ptr<ShaderBindingTable> raygen = nullptr;
		std::unique_ptr<ShaderBindingTable> miss   = nullptr;
		std::unique_ptr<ShaderBindingTable> hit    = nullptr;
		std::unique_ptr<ShaderBindingTable> callable = nullptr;
	} shader_binding_table;
};

struct RenderPassState
{
	RenderGraph &        graph;
	const CommandBuffer &command_buffer;
	const PassNative &   pass;

	const Image &getAttachment(const std::string &name);
};

using ResolveState = ResolveInfo;

class RenderPass
{
  public:
	virtual ~RenderPass() = default;

	virtual void onUpdate(){};

	virtual void setupPipeline(PipelineState &state){};

	virtual void resolveResources(ResolveState &resolve){};

	virtual void render(RenderPassState &state){};

	virtual void onImGui(){};

	virtual std::type_index type() const = 0;

  public:
	void beginProfile(RenderPassState &state);
	void endProfile(RenderPassState &state);

	float getGPUTime();
	float getCPUTime();

	size_t getThreadID();

  protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_cpu_start;
	std::chrono::time_point<std::chrono::high_resolution_clock> m_cpu_end;

	uint64_t m_gpu_start = 0;
	uint64_t m_gpu_end   = 0;

	float m_cpu_time = 0.f;
	float m_gpu_time = 0.f;

	std::thread::id m_thread_id;
};

template <typename T>
class TRenderPass : public RenderPass
{
  public:
	virtual std::type_index type() const override
	{
		return typeid(T);
	}
};
}        // namespace Ilum