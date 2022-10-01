#pragma once

#include <Core/Hash.hpp>
#include <Core/Macro.hpp>

#include <condition_variable>
#include <deque>
#include <optional>
#include <type_traits>

#define RHI_EXPORT __declspec(dllexport)
#define RHI_IMPORT __declspec(dllimport)