#pragma once

#include "Core/Engine/PCH.hpp"
#include "Core/Resource/IResource.hpp"
#include "Image.hpp"

namespace Ilum
{
class Image2D : public Image, public IResource<Image2D>
{
  public:




  public:
	static ref<Image2D> create(
	    const std::string &  path,
	    VkFilter             filter       = VK_FILTER_LINEAR,
	    VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_REPEAT,
	    bool                 mipmap       = true,
	    bool                 anisotropic  = false);

	private:
	const std::string &m_path;

	bool anisotropic = false;
	bool mipmap      = true;
};
}        // namespace Ilum