#include "MeshProcess.hpp"

namespace Ilum
{
template <typename _Ty>
std::unique_ptr<_Ty> &MeshProcess<_Ty>::GetInstance(const std::string &plugin)
{
	static std::unique_ptr<_Ty> ptr = std::unique_ptr<_Ty>(PluginManager::GetInstance().Call<_Ty *>(plugin, "Create"));
	return ptr;
}

template  class MeshProcess<Subdivision>;
}        // namespace Ilum