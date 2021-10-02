#pragma once

#include "Core/Engine/PCH.hpp"

#include "Math/Vector2.h"
#include "Math/Vector4.h"

namespace Ilum
{
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
	Attachment(uint32_t binding, const std::string &name, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, bool multisampled = false);

	Attachment() = default;

	uint32_t getBinding() const;

	const std::string &getName() const;

	bool isMultisampled() const;

	VkFormat getFormat() const;

	const Math::Rgba &getColor() const;

  private:
	uint32_t    m_binding      = 0;
	std::string m_name         = "";
	bool        m_multisampled = false;
	VkFormat    m_format       = {};
	Math::Rgba  m_clear_color  = {};
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
	uint32_t m_binding;
	std::vector<uint32_t> m_attachment_bindings;
};

class RenderTarget
{
  public:
	RenderTarget(const std::vector<Attachment> &attachments = {}, const std::vector<Subpass> &subpasses = {}, const RenderArea &m_render_area = {});



  private:
	std::vector<Attachment> m_attachments;
	std::vector<Subpass>    m_subpass;

	RenderArea m_render_area;
};
}        // namespace Ilum