#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct ShadowMapData
{
	// Texture2D Array
	std::unique_ptr<RHITexture> shadow_map = nullptr;

	// Texture2D Array
	std::unique_ptr<RHITexture> cascade_shadow_map = nullptr;

	// TextureCube Array
	std::unique_ptr<RHITexture> omni_shadow_map = nullptr;
};
}        // namespace Ilum