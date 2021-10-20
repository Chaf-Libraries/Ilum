#pragma once

#include "Graphics/Image/Image.hpp"

namespace Ilum
{
class ResourceCache
{
  public:
	ResourceCache() = default;

	~ResourceCache() = default;

	ImageReference loadImage(const std::string &filepath);

	const std::vector<Image> &getImages() const;

  private:
	std::vector<Image>                              m_image_cache;
	std::unordered_map<std::string, uint32_t> m_image_map;
};
}        // namespace Ilum