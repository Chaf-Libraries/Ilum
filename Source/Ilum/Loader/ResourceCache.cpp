#include "ResourceCache.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

namespace Ilum
{
ImageReference ResourceCache::loadImage(const std::string &filepath)
{
	if (m_image_map.find(filepath) != m_image_map.end())
	{
		return m_image_cache.at(m_image_map.at(filepath));
	}

	m_image_map[filepath] = m_image_cache.size();
	m_image_cache.emplace_back(Image());
	ImageLoader::loadImageFromFile(m_image_cache.back(), filepath);

	return m_image_cache.back();
}

const std::vector<Image> &ResourceCache::getImages() const
{
	return m_image_cache;
}
}        // namespace Ilum