#pragma once

#include <unordered_map>
#include <typeindex>

namespace Ilum
{
class RenderGraphBlackboard
{
    public:
        RenderGraphBlackboard() = default;

        ~RenderGraphBlackboard()
        {
        }

        template<typename _Ty>
        RenderGraphBlackboard& Add();

        template<typename _Ty>
	    bool Has();

        template<typename _Ty>
        _Ty& Get();

        template<typename _Ty>
        RenderGraphBlackboard &Erase();

    private:
        std::unordered_map<std::type_index, void*> m_data;
};
} // namespace Ilum
