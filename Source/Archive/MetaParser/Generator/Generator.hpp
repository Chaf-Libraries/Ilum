#pragma once

#include "Precompile.hpp"
#include "Type/SchemaModule.hpp"

namespace Ilum
{
class Generator
{
  public:
	Generator() = default;

	virtual bool Generate(const std::string &path, const std::string &output_path, const SchemaModule &schema) = 0;

	virtual void OutputFile(const std::string& path) = 0;

  protected:
	std::vector<std::string> m_paths;
};
}        // namespace Ilum