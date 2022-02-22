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
struct PassData
{
	VkPipeline          pipeline        = VK_NULL_HANDLE;
	VkPipelineLayout    pipeline_layout = VK_NULL_HANDLE;
	VkFramebuffer       frame_buffer    = VK_NULL_HANDLE;
	VkDescriptorSet     descriptor_set  = VK_NULL_HANDLE;
	VkPipelineBindPoint bind_point;
};

enum class AccessMode
{
	Read,
	Write
};

struct AttachmentBindInfo
{
	// For resource creation and validation
	std::string name = "";

	// If width=0 or height =0, use global render area
	uint32_t width   = 0;
	uint32_t height  = 0;
	bool     mipmaps = false;
	uint32_t layers  = 1;

	// Resource
	std::vector<Graphics::ImageReference> images;

	// Clear value
	VkClearColorValue        color_clear         = {};
	VkClearDepthStencilValue depth_stencil_clear = {};

	// Render pass value
	VkFormat                format             = VK_FORMAT_UNDEFINED;
	VkSampleCountFlagBits   samples            = VK_SAMPLE_COUNT_1_BIT;
	Graphics::LoadStoreInfo load_store         = {};
	Graphics::LoadStoreInfo stencil_load_store = {};

	// Color Blend
	Graphics::ColorBlendAttachmentState color_blend = {};

	bool depth_stencil;
};

struct ImageBindInfo
{
	std::string name = "";

	std::vector<Graphics::ImageReference> images;

	VkImageUsageFlagBits usage = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;

	AccessMode access = AccessMode::Read;
};

struct BufferBindInfo
{
	std::string name = "";

	std::vector<Graphics::BufferReference> buffers;

	VkBufferUsageFlagBits usage = VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM;

	AccessMode access = AccessMode::Read;
};

struct SamplerBindInfo
{
	std::string name = "";

	std::vector<Graphics::SamplerReference> samplers;
};

class IResourceNode;
class RenderGraph;

class IPassNode : public RenderNode
{
  public:
	IPassNode(const std::string &name, RenderGraph &render_graph);
	~IPassNode() = default;

	bool Bind(int32_t pin, const Graphics::ImageReference &image, AccessMode access);
	bool Bind(int32_t pin, const std::vector<Graphics::ImageReference> &images, AccessMode access);
	bool Bind(int32_t pin, const Graphics::BufferReference &buffer, AccessMode access);
	bool Bind(int32_t pin, const std::vector<Graphics::BufferReference> &buffers, AccessMode access);
	bool Bind(int32_t pin, const Graphics::SamplerReference &sampler);
	bool Bind(int32_t pin, const std::vector<Graphics::SamplerReference> &samplers);

	void OnInitialize();

	virtual void OnUpdate() override;
	virtual void OnImGui() override;
	virtual void OnImNode() override;

	virtual void OnExecute(Graphics::CommandBuffer &cmd_buffer) = 0;

	bool IsValid() const;

  protected:
	void AddDependency(const std::string &name, VkImageUsageFlagBits usage, AccessMode access);
	void AddDependency(const std::string &name, VkBufferUsageFlagBits usage, AccessMode access);

	void BindImage(uint32_t set, uint32_t bind, const std::string &name, VkImageUsageFlagBits usage, AccessMode access);
	void BindBuffer(uint32_t set, uint32_t bind, const std::string &name, VkBufferUsageFlagBits usage, AccessMode access);
	void BindSampler(uint32_t set, uint32_t bind, const std::string &name);

	void AddColorAttachment(
	    const std::string &                 name,
	    VkFormat                            format,
	    Graphics::ColorBlendAttachmentState color_blend,
	    VkClearColorValue                   color_clear,
	    uint32_t                            width              = 0,
	    uint32_t                            height             = 0,
	    bool                                mipmaps            = false,
	    uint32_t                            layers             = 1,
	    VkSampleCountFlagBits               samples            = VK_SAMPLE_COUNT_1_BIT,
	    Graphics::LoadStoreInfo             load_store         = {},
	    Graphics::LoadStoreInfo             stencil_load_store = {});

	void AddDepthStencil(
	    const std::string &      name,
	    VkFormat                 format,
	    VkClearDepthStencilValue depth_stencil_clear,
	    uint32_t                 width              = 0,
	    uint32_t                 height             = 0,
	    bool                     mipmaps            = false,
	    uint32_t                 layers             = 1,
	    VkSampleCountFlagBits    samples            = VK_SAMPLE_COUNT_1_BIT,
	    Graphics::LoadStoreInfo  load_store         = {},
	    Graphics::LoadStoreInfo  stencil_load_store = {});

  private:
	std::vector<std::unique_ptr<ImageBindInfo>>      m_image_bind_infos;
	std::vector<std::unique_ptr<BufferBindInfo>>     m_buffer_bind_infos;
	std::vector<std::unique_ptr<SamplerBindInfo>>    m_sampler_bind_infos;
	std::vector<std::unique_ptr<AttachmentBindInfo>> m_attachment_infos;

	std::map<int32_t, ImageBindInfo *>      m_image_pin;
	std::map<int32_t, BufferBindInfo *>     m_buffer_pin;
	std::map<int32_t, SamplerBindInfo *>    m_sampler_pin;
	std::map<int32_t, AttachmentBindInfo *> m_attachment_pin;

	// Descriptors
	std::map<uint32_t, std::map<uint32_t, ImageBindInfo *>>   m_image_bind_descriptors;
	std::map<uint32_t, std::map<uint32_t, BufferBindInfo *>>  m_buffer_bind_descriptors;
	std::map<uint32_t, std::map<uint32_t, SamplerBindInfo *>> m_sampler_bind_descriptors;

	Graphics::PipelineState m_pso;

	bool m_dirty = true;
	bool m_valid = false;

	PassData m_pass_data;
};

}        // namespace Ilum::Render