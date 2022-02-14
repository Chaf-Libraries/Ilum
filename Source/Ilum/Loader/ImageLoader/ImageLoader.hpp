#pragma once

#include "Bitmap.hpp"

namespace Ilum
{
namespace Graphics
{
class Image;
}

class ImageLoader
{
  public:
	static Bitmap loadImage(const std::string &filepath);

	static Cubemap loadCubemap(const std::string &filepath);

	static Cubemap loadCubemap(const std::array<std::string, 6> &filepaths);

	static void loadImage(Graphics::Image &image, const Bitmap &bitmap, bool mipmaps = true);

	static void loadCubemap(Graphics::Image &image, const Cubemap &cubemap);

	static void loadImageFromFile(Graphics::Image &image, const std::string &filepath, bool mipmaps = true);

	static void loadCubemapFromFile(Graphics::Image &image, const std::string &filepath);
};
}        // namespace Ilum