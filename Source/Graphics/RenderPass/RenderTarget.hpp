#pragma once

#include "Utils/PCH.hpp"

#include "Math/Vector2.h"
#include "Math/Vector4.h"

namespace Ilum
{
class RenderPass;
class ImageDepth;
class Image2D;
class Framebuffer;

class Attachment
{
  public:
	enum class Type
	{
		Image,
		Depth,
		Swapchain
	};

  public:
	Attachment(uint32_t binding, const std::string &name, Type type, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, const Rgba &clear_color = {0, 0, 0, 0}, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

	Attachment() = default;

	uint32_t getBinding() const;

	const std::string &getName() const;

	Type getType() const;

	VkSampleCountFlagBits getSamples() const;

	VkFormat getFormat() const;

	const Rgba &getColor() const;

  private:
	uint32_t              m_binding     = 0;
	std::string           m_name        = "";
	Type                  m_type        = {};
	VkSampleCountFlagBits m_samples     = VK_SAMPLE_COUNT_1_BIT;
	VkFormat              m_format      = {};
	Rgba            m_clear_color = {};
};

class Subpass
{
  public:
	Subpass(uint32_t index, std::vector<uint32_t> &&output_attachments, std::vector<uint32_t> &&input_attachments = {});

	~Subpass() = default;

	uint32_t getIndex() const;

	const std::vector<uint32_t> &getInputAttachments() const;

	const std::vector<uint32_t> &getOutputAttachments() const;

  private:
	uint32_t              m_index;
	std::vector<uint32_t> m_input_attachments;
	std::vector<uint32_t> m_output_attachments;
};

class RenderTarget
{
  public:
	RenderTarget(std::vector<Attachment> &&attachments = {}, std::vector<Subpass> &&subpasses = {}, const VkRect2D &render_area = {});

	void resize(const VkRect2D &render_area);

	const VkRect2D &getRenderArea() const;

	const std::vector<Attachment> &getAttachments() const;

	std::optional<Attachment> getAttachment(uint32_t binding) const;

	std::optional<Attachment> getAttachment(const std::string &name) const;

	const std::vector<Subpass> &getSubpasses() const;

	const ImageDepth *getDepthStencil() const;

	const Image2D *getColorAttachment(uint32_t binding) const;

	const Image2D *getColorAttachment(const std::string &name) const;

	const VkRenderPass &getRenderPass() const;

	const std::vector<uint32_t> &getSubpassAttachmentCounts() const;

	const std::vector<VkClearValue> &getClearValue() const;

	const VkFramebuffer &getCurrentFramebuffer() const;

	bool hasSwapchainAttachment() const;

	bool hasDepthAttachment() const;

  private:
	void build();

  private:
	std::vector<Attachment> m_attachments;
	std::vector<Subpass>    m_subpasses;

	std::optional<Attachment> m_swapchain_attachment;
	std::optional<Attachment> m_depth_attachment;

	scope<RenderPass>           m_render_pass   = nullptr;
	scope<Framebuffer>          m_framebuffer   = nullptr;
	scope<ImageDepth>           m_depth_stencil = nullptr;
	std::vector<scope<Image2D>> m_color_attachments;

	// Binding - Color attachment image
	std::unordered_map<uint32_t, Image2D *> m_color_attachments_mapping;

	std::vector<VkClearValue> m_clear_values;
	std::vector<uint32_t>     m_subpass_attachment_counts;

	VkRect2D m_render_area;
};
}        // namespace Ilum