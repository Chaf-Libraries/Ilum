#pragma once

#include <memory>
#include <string>
#include <vector>

namespace Ilum
{
class BSDF;

class Material
{
  public:
	Material(const std::string &name);

	virtual ~Material() = default;

	void AddBSDF(std::unique_ptr<BSDF> &&bsdf);



  private:
	std::vector<std::unique_ptr<BSDF>> m_bsdfs;
};
}        // namespace Ilum