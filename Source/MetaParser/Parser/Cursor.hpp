#pragma once

#include "Precompile.hpp"

namespace Ilum
{
class Cursor;

enum class AccessSpecifier
{
	Public,
	Private,
	Protected,
};

class CursorType
{
  public:
	CursorType(const CXType &handle);

	~CursorType() = default;

	std::string GetDispalyName() const;

	int32_t GetArgumentCount() const;

	int32_t GetNumTemplateArguments() const;

	CursorType GetArgument(uint32_t index) const;

	CursorType GetCanonicalType() const;

	Cursor GetDeclaration() const;

	CXTypeKind GetKind() const;

	bool IsConstant() const;

  private:
	CXType m_handle;
};

class Cursor
{
  public:
	Cursor(const CXCursor &handle);

	~Cursor() = default;

	CXCursorKind GetKind() const;

	std::string GetSpelling() const;

	std::string GetDisplayName() const;

	std::string GetSourceFile() const;

	bool IsDefinition() const;

	bool IsPureVirtualMethod() const;

	CursorType GetType() const;

	std::vector<Cursor> GetChildren() const;

	void VisitChildren(CXCursorVisitor visitor, void *data = nullptr);

	AccessSpecifier GetAccessSpecifier() const;

	int32_t GetNumTemplateArguments() const;

  private:
	CXCursor m_handle;
};
}        // namespace Ilum