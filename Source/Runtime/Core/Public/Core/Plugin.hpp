#pragma once

#include "Singleton.hpp"

#include <string>

#ifdef _WIN64
#	include <Windows.h>
#	include <libloaderapi.h>
#else
#	error("Windows only for now")
#endif

namespace Ilum
{
class PluginManager : public Singleton<PluginManager>
{
  public:
	PluginManager();

	~PluginManager();

	template <typename _Ty = void, typename... Args>
	_Ty Call(const std::string &lib_path, const std::string &func_name, Args... args)
	{
		typedef _Ty(CALLBACK * FUNC)(Args...);

		auto lib_handle = GetLibrary(lib_path);

		auto func = (FUNC) GetProcAddress(lib_handle, func_name.c_str());

		if constexpr (std::is_same_v<_Ty, void>)
		{
			return;
		}
		else if constexpr (sizeof...(Args) == 0)
		{
			return func();
		}
		else
		{
			return func(args...);
		}
	}

  private:
	HMODULE GetLibrary(const std::string &lib_path);

  private:
	struct Impl;
	Impl *m_impl = nullptr;
};
}        // namespace Ilum