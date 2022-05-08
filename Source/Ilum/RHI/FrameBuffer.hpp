#pragma once

#include "Texture.hpp"

#include <optional>

struct ColorAttachmentInfo
{
	VkSampleCountFlagBits samples          = VK_SAMPLE_COUNT_1_BIT;
	VkAttachmentLoadOp    load_op          = VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkAttachmentStoreOp   store_op         = VK_ATTACHMENT_STORE_OP_STORE;
	VkAttachmentLoadOp    stencil_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	VkAttachmentStoreOp   stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	VkClearColorValue     clear_value      = {};
};

struct DepthStencilAttachmentInfo
{
	VkSampleCountFlagBits samples          = VK_SAMPLE_COUNT_1_BIT;
	VkAttachmentLoadOp    load_op          = VK_ATTACHMENT_LOAD_OP_CLEAR;
	VkAttachmentStoreOp   store_op         = VK_ATTACHMENT_STORE_OP_STORE;
	VkAttachmentLoadOp    stencil_load_op  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	VkAttachmentStoreOp   stencil_store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	VkClearDepthStencilValue clear_value      = {};
};

namespace Ilum
{
class FrameBuffer
{
	friend class RHIDevice;

  public:
	FrameBuffer()  = default;
	~FrameBuffer() = default;

	uint32_t GetWidth() const;
	uint32_t GetHeight() const;
	uint32_t GetLayer() const;

	const std::vector<VkClearValue> &GetClearValue() const;

	FrameBuffer &Bind(Texture *render_target, const TextureViewDesc &view_desc, const ColorAttachmentInfo &info);

	FrameBuffer &Bind(Texture *depth_stencil, const TextureViewDesc &view_desc, const DepthStencilAttachmentInfo &info);

	size_t Hash();

  private:
	uint32_t m_width  = 1;
	uint32_t m_height = 1;
	uint32_t m_layer  = 1;

	std::vector<VkClearValue>            m_clear_values                       = {};
	std::vector<VkImageView>             m_views                              = {};
	std::vector<VkAttachmentDescription> m_attachment_descriptions            = {};
	std::vector<VkAttachmentReference>   m_attachment_references              = {};
	std::optional<VkAttachmentReference> m_depth_stencil_attachment_reference = {};

	size_t m_hash  = 0;
	bool   m_dirty = false;
};
}        // namespace Ilum