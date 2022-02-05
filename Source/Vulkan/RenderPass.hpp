#pragma once

#include "Vulkan.hpp"

namespace Ilum::Vulkan
{
struct LoadStoreInfo
{
	VkAttachmentLoadOp  loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE;
};

struct Attachment
{
	VkFormat              format             = VK_FORMAT_UNDEFINED;
	VkSampleCountFlagBits samples            = VK_SAMPLE_COUNT_1_BIT;
	LoadStoreInfo         load_store         = {};
	LoadStoreInfo         stencil_load_store = {};
	VkImageLayout         initialLayout      = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImageLayout         finalLayout;
};

struct SubpassInfo
{
	std::string           subpass_name;
	std::vector<uint32_t> input_attachments;
	std::vector<uint32_t> output_attachments;
};

class RenderPass
{
  public:
	RenderPass(const std::vector<Attachment> &attachments, const std::vector<SubpassInfo> &subpass_infos);
	RenderPass(const std::vector<Attachment> &attachments);
	~RenderPass();

	RenderPass(const RenderPass &) = delete;
	RenderPass &operator=(const RenderPass &) = delete;
	RenderPass(RenderPass &&)                 = delete;
	RenderPass &operator=(RenderPass &&) = delete;

	operator const VkRenderPass &() const;

	const VkRenderPass &GetHandle() const;

	size_t GetHash() const;

  private:
	VkRenderPass m_handle = VK_NULL_HANDLE;
	size_t       m_hash   = 0;
};

class Framebuffer
{
  public:
	Framebuffer(const std::vector<VkImageView> &views, const RenderPass &render_pass, uint32_t width, uint32_t height, uint32_t layers = 1);
	~Framebuffer();

	Framebuffer(const Framebuffer &) = delete;
	Framebuffer &operator=(const Framebuffer &) = delete;
	Framebuffer(Framebuffer &&)                 = delete;
	Framebuffer &operator=(Framebuffer &&) = delete;

	operator const VkFramebuffer &() const;

	const VkFramebuffer &GetHandle() const;

  private:
	VkFramebuffer m_handle = VK_NULL_HANDLE;
};
}        // namespace Ilum::Vulkan