#pragma once

#include "Bitmap.hpp"

namespace Ilum
{
namespace Graphics
{
class Image;
class Device;
class CommandBuffer;
}        // namespace Graphics

namespace Resource
{
class ImageLoader
{
  public:
	static Bitmap                           LoadTexture2D(const std::string &filepath);
	static std::unique_ptr<Graphics::Image> LoadTexture2D(const Graphics::Device &device, Graphics::CommandBuffer &cmd_buffer, const Bitmap &bitmap, bool mipmaps = true);
	static std::unique_ptr<Graphics::Image> LoadTexture2DFromFile(const Graphics::Device &device, Graphics::CommandBuffer &cmd_buffer, const std::string &filepath, bool mipmaps = true);
};
}        // namespace Resource
}        // namespace Ilum