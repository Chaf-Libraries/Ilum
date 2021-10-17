#pragma once

#include "Bitmap.hpp"

namespace Ilum
{
class Image;

class ImageLoader
{
  public:
	static Bitmap loadImage(const std::string &filepath);

	static Cubemap loadCubemap(const std::string &filepath);

	static Cubemap loadCubemap(const std::array<std::string, 6> &filepaths);

	static void loadImage(Image &image, const Bitmap &bitmap, bool mipmaps = true);

	static void loadCubemap(Image &image, const Cubemap &cubemap);

	static void loadImageFromFile(Image &image, const std::string &filepath, bool mipmaps = true);

	static void loadCubemapFromFile(Image &image, const std::string &filepath);
};
}        // namespace Ilum