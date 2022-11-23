#pragma once

#include "Generator.hpp"

namespace Ilum
{
class ReflectionGenerator : public Generator
{
  public:
	ReflectionGenerator() = default;

	virtual bool Generate(const std::string &input_path, const std::string &output_path, const SchemaModule &schema) override;

	virtual void OutputFile(const std::string &path) override;
};
}        // namespace Ilum