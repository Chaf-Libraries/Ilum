#pragma once

#include <RHI/RHITexture.hpp>

#include <string>

namespace Ilum
{
class RHIContext;

class STBImporter
{
  public:
	static std::unique_ptr<RHITexture> Import(RHIContext* rhi_context, const std::string &filename, bool mipmap);
};
}        // namespace Ilum