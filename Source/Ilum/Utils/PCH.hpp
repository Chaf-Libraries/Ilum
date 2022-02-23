#pragma once

#include "Platform.hpp"

#ifdef ILUM_PLATFORM_WINDOWS
#	ifndef NOMINMAX
#		define NOMINMAX
#	endif        // !NOMINMAX
#endif            // ILUM_PLATFORM_WINDOWS

#include <algorithm>
#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <typeindex>
#include <utility>

#include <array>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Logging/Logger.hpp"

#include "Base.hpp"

#include "Hash.hpp"

#include <volk.h>
#include <vk_mem_alloc.h>