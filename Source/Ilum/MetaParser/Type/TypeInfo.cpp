#include "TypeInfo.hpp"

namespace Ilum
{
TypeInfo::TypeInfo(const Cursor &cursor, const Namespace &current_namespace):
    m_meta_info(cursor), m_root_cursor(cursor), m_namespace(current_namespace), m_enabled(true)
{
}

const MetaInfo &TypeInfo::GetMetaData() const
{
	return m_meta_info;
}

std::string TypeInfo::GetSourceFile() const
{
	return m_root_cursor.GetSourceFile();
}

Namespace TypeInfo::GetCurrentNamespace() const
{
	return m_namespace;
}

Cursor &TypeInfo::GetCursor()
{
	return m_root_cursor;
}
}        // namespace Ilum