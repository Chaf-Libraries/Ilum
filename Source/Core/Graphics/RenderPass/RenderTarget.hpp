#pragma once

#include "Core/Engine/PCH.hpp"

#include "Math/Vector2.h"
#include "Math/Vector4.h"

namespace Ilum
{
class RenderPass;
class ImageDepth;
class Image2D;

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
	Attachment(uint32_t binding, const std::string &name, Type type, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, const Math::Rgba &clear_color = {0, 0, 0, 0}, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

	Attachment() = default;

	uint32_t getBinding() const;

	const std::string &getName() const;

	Type getType() const;

	VkSampleCountFlagBits getSamples() const;

	VkFormat getFormat() const;

	const Math::Rgba &getColor() const;

  private:
	uint32_t           m_binding     = 0;
	std::string        m_name        = "";
	Type               m_type        = {};
	VkSampleCountFlagBits m_samples     = VK_SAMPLE_COUNT_1_BIT;
	VkFormat           m_format      = {};
	Math::Rgba         m_clear_color = {};
};

class RenderArea
{
  public:
	RenderArea(const Math::Vector2 &extent = {}, const Math::Vector2 &offset = {});

	~RenderArea() = default;

	bool operator==(const RenderArea &rhs) const;

	bool operator!=(const RenderArea &rhs) const;

	const Math::Vector2 &getExtent() const;

	void setExtent(const Math::Vector2 &extent);

	const Math::Vector2 &getOffset() const;

	void setOffset(const Math::Vector2 &offset);

  private:
	Math::Vector2 m_extent = {};
	Math::Vector2 m_offset = {};
};

class Subpass
{
  public:
	Subpass(uint32_t binding, const std::vector<uint32_t> &attachment_bindings);

	~Subpass() = default;

	uint32_t getBinding() const;

	const std::vector<uint32_t> &getAttachmentBindings() const;

  private:
	uint32_t              m_binding;
	std::vector<uint32_t> m_attachment_bindings;
};

class RenderTarget
{
  public:
	RenderTarget(const std::vector<Attachment> &attachments = {}, const std::vector<Subpass> &subpasses = {}, const RenderArea &render_area = {});

	const RenderArea &getRenderArea() const;

	const std::vector<Attachment> &getAttachments() const;

	std::optional<Attachment> getAttachment(uint32_t binding) const;

	const std::vector<Subpass> &getSubpasses() const;

	const ImageDepth *getDepthStencil() const;

	const Image2D *getColorAttachment(uint32_t idx) const;

  private:
	std::vector<Attachment> m_attachments;
	std::vector<Subpass>    m_subpasses;

	scope<RenderPass>           m_render_pass   = nullptr;
	scope<ImageDepth>           m_depth_stencil = nullptr;
	std::vector<scope<Image2D>> m_color_attachments;

	RenderArea m_render_area;
};
}        // namespace Ilum