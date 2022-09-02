#include "Descriptor.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"

namespace Ilum::CUDA
{
Descriptor::Descriptor(RHIDevice *device, const ShaderMeta &meta) :
    RHIDescriptor(device, meta)
{
	std::map<size_t, std::string> resource_orders;

	for (auto &constant : meta.constants)
	{
		resource_orders[constant.spirv_id] = constant.name;
		m_resource_sizes[constant.name]    = constant.size;
	}
	for (auto &descriptor : meta.descriptors)
	{
		resource_orders[descriptor.spirv_id] = descriptor.name;
		m_resource_sizes[descriptor.name]    = sizeof(void *);
		if (descriptor.type == DescriptorType::StructuredBuffer)
		{
			m_resource_sizes[descriptor.name] += sizeof(size_t);
		}
		m_resource_type[descriptor.name] = (size_t) descriptor.type;
	}

	size_t offset = 0;
	for (auto &[spirv_id, name] : resource_orders)
	{
		m_resource_offsets[name] = offset;
		offset += m_resource_sizes[name];
	}

	m_param_data.resize(offset);
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension)
{
	size_t   offset         = m_resource_offsets[name];
	uint64_t texture_handle = 0;
	if (m_resource_type[name] == (size_t) DescriptorType::TextureUAV)
	{
		texture_handle = static_cast<Texture *>(texture)->GetSurfaceHandle();
	}
	else if (m_resource_type[name] == (size_t) DescriptorType::TextureSRV)
	{
		texture_handle = static_cast<Texture *>(texture)->GetTextureHandle();
	}

	std::memcpy(m_param_data.data() + offset, &texture_handle, sizeof(texture_handle));
	return *this;
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, RHISampler *sampler)
{
	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer)
{
	size_t offset        = m_resource_offsets[name];
	size_t stride        = static_cast<Buffer *>(buffer)->GetDesc().stride;
	void  *buffer_handle = static_cast<Buffer *>(buffer)->GetHandle();
	std::memcpy(m_param_data.data() + offset, &buffer_handle, sizeof(buffer_handle));
	if (m_resource_type[name] == (size_t)DescriptorType::StructuredBuffer)
	{
		std::memcpy(m_param_data.data() + offset + sizeof(buffer_handle), &stride, sizeof(stride));
	}
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range)
{
	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers)
{
	return *this;
}

RHIDescriptor &Descriptor::BindConstant(const std::string &name, const void *constant)
{
	return *this;
}

std::vector<uint8_t> &Descriptor::GetParamData()
{
	return m_param_data;
}
}        // namespace Ilum::CUDA