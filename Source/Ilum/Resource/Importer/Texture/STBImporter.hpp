#pragma once

#include "TextureImporter.hpp"

#include <Core/Singleton.hpp>

namespace Ilum
{
class RHIContext;

class STBImporter : public TextureImporter, public Singleton<STBImporter>
{
  public:
	virtual void Import(RHIContext *rhi_context, const std::string &filename, std::vector<uint8_t> &data, TextureDesc &desc) override;
};
}        // namespace Ilum