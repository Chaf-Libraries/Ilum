#pragma once

#include "Image.hpp"

namespace Ilum
{
class ImageDepth : public Image
{
  public:
	ImageDepth(const uint32_t width, const uint32_t height, VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
};
}        // namespace Ilum