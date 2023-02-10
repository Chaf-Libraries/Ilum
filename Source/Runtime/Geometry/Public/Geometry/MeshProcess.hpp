#pragma once

#include "Mesh/Mesh.hpp"
#include "Precompile.hpp"

namespace Ilum
{
template <typename _Ty>
class  MeshProcess
{
  public:
	static std::unique_ptr<_Ty> &GetInstance(const std::string &plugin);
};

class  Subdivision : public MeshProcess<Subdivision>
{
  public:
	virtual TriMesh Execute(const TriMesh &mesh) = 0;
};
}        // namespace Ilum