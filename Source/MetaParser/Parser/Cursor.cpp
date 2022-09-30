#include "Cursor.hpp"

namespace std
{
std::string to_string(const CXString &str)
{
	std::string output;

	auto cstr = clang_getCString(str);
	output    = cstr;
	clang_disposeString(str);
	return output;
}
}        // namespace std

namespace Ilum
{
CursorType::CursorType(const CXType &handle) :
    m_handle(handle)
{
}

std::string CursorType::GetDispalyName() const
{
	return std::to_string(clang_getTypeSpelling(m_handle));
}

int32_t CursorType::GetArgumentCount() const
{
	return clang_getNumArgTypes(m_handle);
}

CursorType CursorType::GetArgument(uint32_t index) const
{
	return clang_getArgType(m_handle, index);
}

CursorType CursorType::GetCanonicalType() const
{
	return clang_getCanonicalType(m_handle);
}

Cursor CursorType::GetDeclaration() const
{
	return clang_getTypeDeclaration(m_handle);
}

CXTypeKind CursorType::GetKind() const
{
	return m_handle.kind;
}

bool CursorType::IsConstant() const
{
	return clang_isConstQualifiedType(m_handle) ? true : false;
}

Cursor::Cursor(const CXCursor &handle) :
    m_handle(handle)
{
}

CXCursorKind Cursor::GetKind() const
{
	return m_handle.kind;
}

std::string Cursor::GetSpelling() const
{
	return std::to_string(clang_getCursorSpelling(m_handle));
}

std::string Cursor::GetDisplayName() const
{
	return std::to_string(clang_getCursorDisplayName(m_handle));
}

std::string Cursor::GetSourceFile() const
{
	auto range = clang_Cursor_getSpellingNameRange(m_handle, 0, 0);

	auto start = clang_getRangeStart(range);

	CXFile   file;
	uint32_t line, column, offset;

	clang_getFileLocation(start, &file, &line, &column, &offset);

	return std::to_string(clang_getFileName(file));
}

bool Cursor::IsDefinition() const
{
	return clang_isCursorDefinition(m_handle);
}

CursorType Cursor::GetType() const
{
	return clang_getCursorType(m_handle);
}

std::vector<Cursor> Cursor::GetChildren() const
{
	std::vector<Cursor> children;

	auto visitor = [](CXCursor cursor, CXCursor parent, CXClientData data) {
		auto container = static_cast<std::vector<Cursor> *>(data);
		container->emplace_back(cursor);
		if (cursor.kind == CXCursor_LastPreprocessing)
		{
			return CXChildVisit_Break;
		}
		return CXChildVisit_Continue;
	};

	clang_visitChildren(m_handle, visitor, &children);

	return children;
}

void Cursor::VisitChildren(CXCursorVisitor visitor, void *data)
{
	clang_visitChildren(m_handle, visitor, data);
}
}        // namespace Ilum