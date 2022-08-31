#include "Descriptor.hpp"
#include "Buffer.hpp"
#include "Texture.hpp"

namespace Ilum::CUDA
{
Descriptor::Descriptor(RHIDevice *device, const ShaderMeta &meta) :
    RHIDescriptor(device, meta)
{
	std::map<std::string, size_t> resource_orders;

	for (auto &constant : meta.constants)
	{
		resource_orders[constant.name]  = constant.spirv_id;
		m_resource_sizes[constant.name] = constant.size;
	}
	for (auto &descriptor : meta.descriptors)
	{
		resource_orders[descriptor.name]  = descriptor.spirv_id;
		m_resource_sizes[descriptor.name] = sizeof(void *);
	}

	size_t offset = 0;
	for (auto &[name, spirv_id] : resource_orders)
	{
		m_resource_offsets[name] = offset;
		offset += m_resource_sizes[name];
	}

	m_param_data.resize(offset);
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension)
{
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
	void  *buffer_handle = static_cast<Buffer *>(buffer)->GetHandle();
	std::memcpy(m_param_data.data(), &buffer_handle, sizeof(buffer_handle));
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