#include <RHI/RHITexture.hpp>
#include <Resource/Importer.hpp>
#include <Resource/Resource/Texture2D.hpp>
#include <Resource/Resource/TextureCube.hpp>
#include <Resource/ResourceManager.hpp>

#include <stb_image.h>

using namespace Ilum;

class STBImporter : public Importer<ResourceType::Texture2D>
{
  protected:
	virtual void Import_(ResourceManager *manager, const std::string &path, RHIContext *rhi_context) override
	{
		std::string texture_name = Path::GetInstance().ValidFileName(path);

		if (manager->Has<ResourceType::Texture2D>(texture_name))
		{
			return;
		}

		TextureDesc desc = {};
		desc.name        = texture_name;
		desc.width       = 1;
		desc.height      = 1;
		desc.depth       = 1;
		desc.mips        = 1;
		desc.layers      = 1;
		desc.samples     = 1;

		int32_t width = 0, height = 0, channel = 0;

		const int32_t req_channel = 4;

		void  *raw_data = nullptr;
		size_t size     = 0;

		if (stbi_is_hdr(path.c_str()))
		{
			raw_data    = stbi_loadf(path.c_str(), &width, &height, &channel, req_channel);
			size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
			desc.format = RHIFormat::R32G32B32A32_FLOAT;
		}
		else if (stbi_is_16_bit(path.c_str()))
		{
			raw_data    = stbi_load_16(path.c_str(), &width, &height, &channel, req_channel);
			size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
			desc.format = RHIFormat::R16G16B16A16_FLOAT;
		}
		else
		{
			raw_data    = stbi_load(path.c_str(), &width, &height, &channel, req_channel);
			size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
			desc.format = RHIFormat::R8G8B8A8_UNORM;
		}

		desc.width  = static_cast<uint32_t>(width);
		desc.height = static_cast<uint32_t>(height);
		desc.usage  = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

		std::vector<uint8_t> data;

		data.resize(size);
		std::memcpy(data.data(), raw_data, size);

		stbi_image_free(raw_data);

		if (Path::GetInstance().GetFileExtension(path) == ".hdr")
		{
			desc.mips = 1;
			manager->Add<ResourceType::TextureCube>(rhi_context, std::move(data), desc);
		}
		else
		{
			desc.mips = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
			manager->Add<ResourceType::Texture2D>(rhi_context, std::move(data), desc);
		}
	}
};

extern "C"
{
	EXPORT_API STBImporter *Create()
	{
		return new STBImporter;
	}
}