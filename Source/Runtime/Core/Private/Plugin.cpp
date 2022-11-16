#include "Plugin.hpp"

#include <unordered_map>

namespace Ilum
{
struct PluginManager::Impl
{
	std::unordered_map<std::string, HMODULE> modules;
};

PluginManager::PluginManager()
{
	m_impl = new Impl;
}

PluginManager::~PluginManager()
{
	delete m_impl;
}

HMODULE PluginManager::GetLibrary(const std::string &lib_path)
{
	if (m_impl->modules.find(lib_path) == m_impl->modules.end())
	{
		m_impl->modules.emplace(lib_path, LoadLibraryA(("lib/" + lib_path).c_str()));
	}
	return m_impl->modules.at(lib_path);
}
}        // namespace Ilum