#include "PassNode.hpp"
#include "RenderGraph.hpp"
#include "ResourceNode.hpp"

#include <imnodes.h>

namespace Ilum::Render
{
IPassNode::IPassNode(const std::string &name, RenderGraph &render_graph) :
    RenderNode(name, render_graph)
{
}

bool IPassNode::Bind(int32_t pin, const Graphics::ImageReference &image, AccessMode access)
{
	if (m_image_pin.find(pin) != m_image_pin.end())
	{
		if (m_image_pin[pin]->access != access)
		{
			return false;
		}
		m_image_pin[pin]->images.clear();
		m_image_pin[pin]->images.push_back(image);
		return true;
	}
	if (m_attachment_pin.find(pin) != m_attachment_pin.end())
	{
		if (access != AccessMode::Write)
		{
			return false;
		}
		m_attachment_pin[pin]->images.clear();
		m_attachment_pin[pin]->images.push_back(image);
		return true;
	}
	return false;
}

bool IPassNode::Bind(int32_t pin, const std::vector<Graphics::ImageReference> &images, AccessMode access)
{
	if (m_image_pin.find(pin) != m_image_pin.end())
	{
		if (m_image_pin[pin]->access != access)
		{
			return false;
		}
		m_image_pin[pin]->images = images;
		return true;
	}
	if (m_attachment_pin.find(pin) != m_attachment_pin.end())
	{
		if (access != AccessMode::Write)
		{
			return false;
		}
		m_image_pin[pin]->images = images;

		return true;
	}
	return false;
}

bool IPassNode::Bind(int32_t pin, const Graphics::BufferReference &buffer, AccessMode access)
{
	if (m_buffer_pin.find(pin) != m_buffer_pin.end())
	{
		if (m_buffer_pin[pin]->access != access)
		{
			return false;
		}
		m_buffer_pin[pin]->buffers.clear();
		m_buffer_pin[pin]->buffers.push_back(buffer);
		return true;
	}
	return false;
}

bool IPassNode::Bind(int32_t pin, const std::vector<Graphics::BufferReference> &buffers, AccessMode access)
{
	if (m_buffer_pin.find(pin) != m_buffer_pin.end())
	{
		if (m_buffer_pin[pin]->access != access)
		{
			return false;
		}
		m_buffer_pin[pin]->buffers = buffers;
		return true;
	}
	return false;
}

bool IPassNode::Bind(int32_t pin, const Graphics::SamplerReference &sampler)
{
	if (m_sampler_pin.find(pin) != m_sampler_pin.end())
	{
		m_sampler_pin[pin]->samplers.clear();
		m_sampler_pin[pin]->samplers.push_back(sampler);
		return true;
	}
	return false;
}

bool IPassNode::Bind(int32_t pin, const std::vector<Graphics::SamplerReference> &samplers)
{
	if (m_sampler_pin.find(pin) != m_sampler_pin.end())
	{
		m_sampler_pin[pin]->samplers = samplers;
		return true;
	}
	return false;
}

void IPassNode::OnInitialize()
{
}

void IPassNode::OnUpdate()
{
	if (m_dirty)
	{
		m_dirty = false;
	}
}

void IPassNode::OnImGui()
{
}

void IPassNode::OnImNode()
{
	const float node_width = 100.f;

	ImNodes::BeginNode(m_uuid);

	ImNodes::BeginNodeTitleBar();
	ImGui::TextUnformatted(m_name.c_str());
	ImNodes::EndNodeTitleBar();

	// Image binding
	for (auto &[pin, image_bind_info] : m_image_pin)
	{
		if (image_bind_info->access == AccessMode::Read)
		{
			ImNodes::BeginInputAttribute(pin);
			const float label_width = ImGui::CalcTextSize(image_bind_info->name.c_str()).x;
			ImGui::TextUnformatted(image_bind_info->name.c_str());
			ImNodes::EndInputAttribute();
		}
		else if (image_bind_info->access == AccessMode::Write)
		{
			ImNodes::BeginOutputAttribute(pin);
			const float label_width = ImGui::CalcTextSize(image_bind_info->name.c_str()).x;
			ImGui::Indent(node_width);
			ImGui::TextUnformatted(image_bind_info->name.c_str());
			ImNodes::EndOutputAttribute();
		}
	}

	// Buffer binding
	for (auto &[pin, buffer_bind_info] : m_buffer_pin)
	{
		if (buffer_bind_info->access == AccessMode::Read)
		{
			ImNodes::BeginInputAttribute(pin);
			const float label_width = ImGui::CalcTextSize(buffer_bind_info->name.c_str()).x;
			ImGui::TextUnformatted(buffer_bind_info->name.c_str());
			ImNodes::EndInputAttribute();
		}
		else if (buffer_bind_info->access == AccessMode::Write)
		{
			ImNodes::BeginInputAttribute(pin);
			const float label_width = ImGui::CalcTextSize(buffer_bind_info->name.c_str()).x;
			ImGui::Indent(node_width);
			ImGui::TextUnformatted(buffer_bind_info->name.c_str());
			ImNodes::EndInputAttribute();
		}
	}

	// Sampler binding
	for (auto &[pin, sampler_bind_info] : m_sampler_pin)
	{
		ImNodes::BeginInputAttribute(pin);
		const float label_width = ImGui::CalcTextSize(sampler_bind_info->name.c_str()).x;
		ImGui::TextUnformatted(sampler_bind_info->name.c_str());
		ImNodes::EndInputAttribute();
	}

	// Output attachment
	for (auto &[pin, attachment_bind_info] : m_attachment_pin)
	{
		ImNodes::BeginOutputAttribute(pin);
		const float label_width = ImGui::CalcTextSize(attachment_bind_info->name.c_str()).x;
		ImGui::Indent(node_width - label_width);
		ImGui::Text(attachment_bind_info->name.c_str());
		ImNodes::EndOutputAttribute();
	}

	ImNodes::EndNode();
}

bool IPassNode::IsValid() const
{
	return m_valid;
}

void IPassNode::AddDependency(const std::string &name, VkImageUsageFlagBits usage, AccessMode access)
{
	auto image_bind_info    = std::make_unique<ImageBindInfo>();
	image_bind_info->name   = name;
	image_bind_info->usage  = usage;
	image_bind_info->access = access;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_image_pin.emplace(pin, image_bind_info.get());
	m_image_bind_infos.emplace_back(std::move(image_bind_info));
}

void IPassNode::AddDependency(const std::string &name, VkBufferUsageFlagBits usage, AccessMode access)
{
	auto buffer_bind_info    = std::make_unique<BufferBindInfo>();
	buffer_bind_info->name   = name;
	buffer_bind_info->usage  = usage;
	buffer_bind_info->access = access;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_buffer_pin.emplace(pin, buffer_bind_info.get());
	m_buffer_bind_infos.emplace_back(std::move(buffer_bind_info));
}

void IPassNode::BindImage(uint32_t set, uint32_t bind, const std::string &name, VkImageUsageFlagBits usage, AccessMode access)
{
	auto image_bind_info    = std::make_unique<ImageBindInfo>();
	image_bind_info->name   = name;
	image_bind_info->usage  = usage;
	image_bind_info->access = access;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_image_bind_descriptors[set][bind] = image_bind_info.get();
	m_image_pin.emplace(pin, image_bind_info.get());
	m_image_bind_infos.emplace_back(std::move(image_bind_info));
}

void IPassNode::BindBuffer(uint32_t set, uint32_t bind, const std::string &name, VkBufferUsageFlagBits usage, AccessMode access)
{
	auto buffer_bind_info    = std::make_unique<BufferBindInfo>();
	buffer_bind_info->name   = name;
	buffer_bind_info->usage  = usage;
	buffer_bind_info->access = access;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_buffer_bind_descriptors[set][bind] = buffer_bind_info.get();
	m_buffer_pin.emplace(pin, buffer_bind_info.get());
	m_buffer_bind_infos.emplace_back(std::move(buffer_bind_info));
}

void IPassNode::BindSampler(uint32_t set, uint32_t bind, const std::string &name)
{
	auto sampler_bind_info  = std::make_unique<SamplerBindInfo>();
	sampler_bind_info->name = name;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_sampler_bind_descriptors[set][bind] = sampler_bind_info.get();
	m_sampler_pin.emplace(pin, sampler_bind_info.get());
	m_sampler_bind_infos.emplace_back(std::move(sampler_bind_info));
}

void IPassNode::AddColorAttachment(
    const std::string &                 name,
    VkFormat                            format,
    Graphics::ColorBlendAttachmentState color_blend,
    VkClearColorValue                   color_clear,
    uint32_t                            width,
    uint32_t                            height,
    bool                                mipmaps,
    uint32_t                            layers,
    VkSampleCountFlagBits               samples,
    Graphics::LoadStoreInfo             load_store,
    Graphics::LoadStoreInfo             stencil_load_store)
{
	auto attachment                = std::make_unique<AttachmentBindInfo>();
	attachment->name               = name;
	attachment->format             = format;
	attachment->color_blend        = color_blend;
	attachment->color_clear        = color_clear;
	attachment->width              = width;
	attachment->height             = height;
	attachment->mipmaps            = mipmaps;
	attachment->layers             = layers;
	attachment->samples            = samples;
	attachment->load_store         = load_store;
	attachment->stencil_load_store = stencil_load_store;
	attachment->depth_stencil      = false;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_attachment_infos.emplace_back(std::move(attachment));
	m_attachment_pin.emplace(pin, attachment.get());
}

void IPassNode::AddDepthStencil(
    const std::string &      name,
    VkFormat                 format,
    VkClearDepthStencilValue depth_stencil_clear,
    uint32_t                 width,
    uint32_t                 height,
    bool                     mipmaps,
    uint32_t                 layers,
    VkSampleCountFlagBits    samples,
    Graphics::LoadStoreInfo  load_store,
    Graphics::LoadStoreInfo  stencil_load_store)
{
	auto attachment                 = std::make_unique<AttachmentBindInfo>();
	attachment->name                = name;
	attachment->format              = format;
	attachment->depth_stencil_clear = depth_stencil_clear;
	attachment->width               = width;
	attachment->height              = height;
	attachment->mipmaps             = mipmaps;
	attachment->layers              = layers;
	attachment->samples             = samples;
	attachment->load_store          = load_store;
	attachment->stencil_load_store  = stencil_load_store;
	attachment->depth_stencil       = true;

	int32_t pin = NewUUID();
	m_render_graph.RegisterPin(pin, m_uuid);

	m_attachment_infos.emplace_back(std::move(attachment));
	m_attachment_pin.emplace(pin, attachment.get());
}

}        // namespace Ilum::Render