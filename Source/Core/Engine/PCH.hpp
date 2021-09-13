#pragma once

#include "Platform.hpp"

#ifdef ILUM_PLATFORM_WINDOWS
#	ifndef NOMINMAX
#		define NOMINMAX
#	endif        // !NOMINMAX
#endif            // ILUM_PLATFORM_WINDOWS

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>
#include <future>

#include <array>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>

#include "Logging/Logger.hpp"

#include "Base.hpp"

#include "Core/Utils/Hash.hpp"

#include "Core/Engine/Vulkan/Vulkan.hpp"