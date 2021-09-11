#pragma once

#ifdef _WIN32
#	ifdef _WIN64
#		define ILUM_PLATFORM_WINDOWS
#	else
#		error "x86 builds are not supported!"
#	endif        // _WIN64
#else
#	error "Unknow platform!"
#endif        // _WIN32
