#pragma once

#ifdef _WIN32
#	ifdef _WIN64
#		define ILUM_PLATFORM_WINDOWS
#	else
#		warn "x86 builds are not supported!"
#	endif        // _WIN64
#elif __linux__
#	define ILUM_PLATFORM_LINUX
#else
#	error "Unknow platform!"
#endif        // _WIN32
