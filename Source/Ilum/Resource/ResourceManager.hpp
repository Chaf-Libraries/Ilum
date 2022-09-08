#pragma once

#include <RHI/RHITexture.hpp>

#include <string>
#include <unordered_map>
#include <vector>

namespace Ilum
{
class RHIContext;

class ResourceManager
{
  public:
	ResourceManager(RHIContext *rhi_context);

	~ResourceManager();

	void ImportTexture(const std::string &filename);

  private:
	RHIContext *p_rhi_context = nullptr;

	// Texture
	struct TextureMeta
	{
		TextureDesc desc;

		std::unique_ptr<RHITexture> texture    = nullptr;
		std::unique_ptr<RHITexture> thumbnails = nullptr;

		std::string asset_path;
	};

	std::vector<RHITexture *>                m_texture_array;
	std::vector<TextureMeta>                m_textures;
	std::unordered_map<std::string, size_t>  m_texture_index;

	// Thumbnails
	std::unordered_map<std::string, RHITexture *> m_thumbnails;
};
}        // namespace Ilum