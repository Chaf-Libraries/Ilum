#pragma once

#include "Meta/MetaInfo.hpp"
#include "Meta/MetaUtils.hpp"
#include "Parser/Cursor.hpp"
#include "Precompile.hpp"

namespace Ilum
{
class TypeInfo
{
  public:
	TypeInfo(const Cursor &cursor, const Namespace &current_namespace);

	virtual ~TypeInfo() = default;

	virtual bool ShouldReflection() const;

	virtual bool ShouldSerialization() const;

	virtual bool ShouldCompile() const;

	const MetaInfo &GetMetaData() const;

	std::string GetSourceFile() const;

	Namespace GetCurrentNamespace() const;

	Cursor &GetCursor();

	virtual kainjow::mustache::data GenerateReflection() const = 0;

	virtual kainjow::mustache::data GenerateSerialization() const
	{
		return kainjow::mustache::data{};
	}

  protected:
	MetaInfo m_meta_info;

	bool m_enabled;

	std::string m_alias_cn;

	Namespace m_namespace;

  private:
	Cursor m_root_cursor;
};
}        // namespace Ilum