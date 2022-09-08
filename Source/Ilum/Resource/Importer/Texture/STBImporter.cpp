#include "STBImporter.hpp"

#include <Core/Path.hpp>
#include <RHI/RHIContext.hpp>

#include <stb_image.h>

namespace Ilum
{
std::unique_ptr<RHITexture> STBImporter::Import(RHIContext *rhi_context, const std::string &filename, bool mipmap)
{
	TextureDesc desc;

	desc.name = Path::GetInstance().GetFileName(filename, false);

	int32_t width = 0, height = 0, channel = 0;

	const int32_t req_channel = 4;

	void  *data = nullptr;
	size_t size = 0;

	if (stbi_is_hdr(filename.c_str()))
	{
		data        = stbi_loadf(filename.c_str(), &width, &height, &channel, req_channel);
		size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
		desc.format = RHIFormat::R32G32B32A32_FLOAT;
	}
	else if (stbi_is_16_bit(filename.c_str()))
	{
		data        = stbi_load_16(filename.c_str(), &width, &height, &channel, req_channel);
		size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
		desc.format = RHIFormat::R16G16B16A16_FLOAT;
	}
	else
	{
		data        = stbi_load(filename.c_str(), &width, &height, &channel, req_channel);
		size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
		desc.format = RHIFormat::R8G8B8A8_UNORM;
	}

	desc.width  = static_cast<uint32_t>(width);
	desc.height = static_cast<uint32_t>(height);
	desc.mips   = mipmap ? static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1) : 1;
	desc.usage  = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

	BufferDesc buffer_desc = {};
	buffer_desc.size       = size;
	buffer_desc.usage      = RHIBufferUsage::Transfer;
	buffer_desc.memory     = RHIMemoryUsage::CPU_TO_GPU;

	auto staging_buffer = rhi_context->CreateBuffer(buffer_desc);
	std::memcpy(staging_buffer->Map(), data, buffer_desc.size);
	staging_buffer->Flush(0, buffer_desc.size);
	staging_buffer->Unmap();

	auto texture = rhi_context->CreateTexture2D(desc.width, desc.height, desc.format, desc.usage, mipmap);

	auto *cmd_buffer = rhi_context->CreateCommand(RHIQueueFamily::Graphics);
	cmd_buffer->Begin();
	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::Undefined,
	                                        RHIResourceState::TransferDest,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, desc.mips, 0, 1}}},
	                                    {});

	// TODO: Copy buffer to texture
	// TODO: Generate mipmaps

	cmd_buffer->ResourceStateTransition({TextureStateTransition{
	                                        texture.get(),
	                                        RHIResourceState::TransferDest,
	                                        RHIResourceState::Undefined,
	                                        TextureRange{RHITextureDimension::Texture2D, 0, desc.mips, 0, 1}}},
	                                    {});

	cmd_buffer->End();

	auto queue = rhi_context->CreateQueue(RHIQueueFamily::Graphics, 1);
	auto fence = rhi_context->CreateFence();
	queue->Submit({cmd_buffer});
	queue->Execute(fence.get());
	fence->Wait();

	return texture;
}
}        // namespace Ilum