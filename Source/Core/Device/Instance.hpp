#pragma once

#include "Core/Engine/PCH.hpp"

#include <volk.h>

namespace Ilum
{
class Instance
{
  public:
	Instance();

	~Instance();

	bool isDebugEnable() const;

  private:
	VkInstance m_handle = VK_NULL_HANDLE;

#ifdef _DEBUG
	std::vector<const char *> m_extensions        = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_report", "VK_EXT_debug_utils"};
	std::vector<const char *> m_validation_layers = {"VK_LAYER_KHRONOS_validation"};
	std::vector<VkValidationFeatureEnableEXT> m_validation_extensions = {
		VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT, 
		VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT, 
		VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT, 
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT};

	bool m_debug_enable = true;
#else
	std::vector<const char *> m_extensions        = {"VK_KHR_surface", "VK_KHR_win32_surface"};
	std::vector<const char *> m_validation_layers = {};
	std::vector<VkValidationFeatureEnableEXT> m_validation_extensions = {};

	bool m_debug_enable = false;
#endif        // _DEBUG
};
}        // namespace Ilum