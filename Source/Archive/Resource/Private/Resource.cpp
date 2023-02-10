#include "Resource.hpp"

#include <Core/Path.hpp>

namespace Ilum
{
Resource::Resource(size_t uuid) :
    m_uuid(uuid)
{
}

Resource::Resource(size_t uuid, const std::string &meta, RHIContext *rhi_context) :
    m_uuid(uuid), m_meta(meta)
{
}

size_t Resource::GetUUID() const
{
	return m_uuid;
}

const std::string &Resource::GetMeta() const
{
	return m_meta;
}

bool Resource::IsValid() const
{
	return m_valid;
}
}        // namespace Ilum