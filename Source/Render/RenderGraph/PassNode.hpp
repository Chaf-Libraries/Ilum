#pragma once

#include "RenderNode.hpp"

#include <Graphics/Command/CommandBuffer.hpp>
#include <Graphics/Pipeline/PipelineState.hpp>
#include <Graphics/RenderPass/RenderPass.hpp>
#include <Graphics/Resource/Buffer.hpp>
#include <Graphics/Resource/Image.hpp>
#include <Graphics/Resource/Sampler.hpp>

namespace Ilum::Render
{
enum class PassType
{
	Precompute,
	Persistent,
	AsyncCompute
};

struct PassData
{
	VkPipeline          pipeline        = VK_NULL_HANDLE;
	VkPipelineLayout    pipeline_layout = VK_NULL_HANDLE;
	VkFramebuffer       frame_buffer    = VK_NULL_HANDLE;
	VkDescriptorSet     descriptor_set  = VK_NULL_HANDLE;
	VkPipelineBindPoint bind_point;
};

struct AttachmentBindInfo
{
	// For resource creation and validation
	std::string name    = "";
	VkFormat    format  = VK_FORMAT_UNDEFINED;
	uint32_t    width   = 0;
	uint32_t    height  = 0;
	bool        mipmaps = false;
	uint32_t    layers  = 1;

	// Resource
	const Graphics::Image *image = nullptr;

	// Clear value
	VkClearColorValue        color_clear         = {};
	VkClearDepthStencilValue depth_stencil_clear = {};

	// Render pass value
	Graphics::Attachment attachment = {};

	// Color Blend
	Graphics::ColorBlendAttachmentState color_blend = {};
};

struct ImageBindInfo
{
	std::string name = "";

	std::vector<const Graphics::Image *> images;

	struct
	{
		VkImageUsageFlagBits initial_usage;
		VkImageUsageFlagBits final_usage;
	} transition;
};

struct BufferBindInfo
{
	std::string name = "";

	std::vector<const Graphics::Buffer *> buffers;

	struct
	{
		VkBufferUsageFlagBits initial_usage;
		VkBufferUsageFlagBits final_usage;
	} transition;
};

struct SamplerBindInfo
{
	std::string name = "";

	std::vector<const Graphics::Sampler *> samplers;
};

class IResourceNode;

class IPassNode : public RenderNode
{
  public:
	IPassNode(const std::string &name, PassType pass_type);
	~IPassNode() = default;

	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, Graphics::ImageReference image);
	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, const std::vector<Graphics::ImageReference> &images);
	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, Graphics::BufferReference image);
	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, const std::vector<Graphics::BufferReference> &buffers);
	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, Graphics::SamplerReference sampler);
	bool Bind(uint32_t set, uint32_t binding, IResourceNode *resource, const std::vector<Graphics::SamplerReference> &samplers);
	bool Output(uint32_t index, IResourceNode *resource, Graphics::ImageReference attachment);

	bool Unbind(IResourceNode *resource);

	bool Validate();

	void OnUpdate();
	void OnImGui();
	void OnImNode();
	void OnExecute(Graphics::CommandBuffer &cmd_buffer);

	PassType           GetPassType() const;
	const std::string &GetName() const;

  private:
	std::string m_name;

	std::map<uint32_t, std::map<uint32_t, ImageBindInfo>>   m_image_bind_infos;
	std::map<uint32_t, std::map<uint32_t, BufferBindInfo>>  m_buffer_bind_infos;
	std::map<uint32_t, std::map<uint32_t, SamplerBindInfo>> m_sampler_bind_infos;



	std::vector<AttachmentBindInfo>                         m_attachments;

	std::map<uint32_t, std::map<uint32_t, IResourceNode *>> m_resource_bind;
	std::map<uint32_t, IResourceNode *>                     m_resource_attachment;

	bool     m_dirty = false;
	PassType m_type;

	PassData m_pass_data;
};

}        // namespace Ilum::Render