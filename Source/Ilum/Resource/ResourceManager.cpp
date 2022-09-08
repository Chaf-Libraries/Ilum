#include "ResourceManager.hpp"

#include <filesystem>

namespace Ilum
{
ResourceManager::ResourceManager(RHIContext *rhi_context)
{
}

ResourceManager::~ResourceManager()
{
}

void ResourceManager::ImportTexture(const std::string &filename)
{
	std::string asset_name = std::to_string(std::filesystem::hash_value(filename));
	if (m_texture_index.find(asset_name) != m_texture_index.end())
	{
		return;
	}

}
}        // namespace Ilum