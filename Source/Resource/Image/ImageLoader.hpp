#pragma once

#include "Bitmap.hpp"

#include <Graphics/Resource/Image.hpp>

#include <Core/JobSystem/SpinLock.hpp>

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

  private:
	inline static Core::SpinLock m_lock;
};
}        // namespace Resource
}        // namespace Ilum