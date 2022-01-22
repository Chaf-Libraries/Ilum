#pragma once

#include <ostream>

namespace Ilum::Core
{
typedef enum class KeyCode : uint32_t
{
	// From glfw3.h
	Space      = 32,
	Apostrophe = 39, /* ' */
	Comma      = 44, /* , */
	Minus      = 45, /* - */
	Period     = 46, /* . */
	Slash      = 47, /* / */

	D0 = 48, /* 0 */
	D1 = 49, /* 1 */
	D2 = 50, /* 2 */
	D3 = 51, /* 3 */
	D4 = 52, /* 4 */
	D5 = 53, /* 5 */
	D6 = 54, /* 6 */
	D7 = 55, /* 7 */
	D8 = 56, /* 8 */
	D9 = 57, /* 9 */

	Semicolon = 59, /* ; */
	Equal     = 61, /* = */

	A = 65,
	B = 66,
	C = 67,
	D = 68,
	E = 69,
	F = 70,
	G = 71,
	H = 72,
	I = 73,
	J = 74,
	K = 75,
	L = 76,
	M = 77,
	N = 78,
	O = 79,
	P = 80,
	Q = 81,
	R = 82,
	S = 83,
	T = 84,
	U = 85,
	V = 86,
	W = 87,
	X = 88,
	Y = 89,
	Z = 90,

	LeftBracket  = 91, /* [ */
	Backslash    = 92, /* \ */
	RightBracket = 93, /* ] */
	GraveAccent  = 96, /* ` */

	World1 = 161, /* non-US #1 */
	World2 = 162, /* non-US #2 */

	/* Function keys */
	Escape      = 256,
	Enter       = 257,
	Tab         = 258,
	Backspace   = 259,
	Insert      = 260,
	Delete      = 261,
	Right       = 262,
	Left        = 263,
	Down        = 264,
	Up          = 265,
	PageUp      = 266,
	PageDown    = 267,
	Home        = 268,
	End         = 269,
	CapsLock    = 280,
	ScrollLock  = 281,
	NumLock     = 282,
	PrintScreen = 283,
	Pause       = 284,
	F1          = 290,
	F2          = 291,
	F3          = 292,
	F4          = 293,
	F5          = 294,
	F6          = 295,
	F7          = 296,
	F8          = 297,
	F9          = 298,
	F10         = 299,
	F11         = 300,
	F12         = 301,
	F13         = 302,
	F14         = 303,
	F15         = 304,
	F16         = 305,
	F17         = 306,
	F18         = 307,
	F19         = 308,
	F20         = 309,
	F21         = 310,
	F22         = 311,
	F23         = 312,
	F24         = 313,
	F25         = 314,

	/* Keypad */
	KP0        = 320,
	KP1        = 321,
	KP2        = 322,
	KP3        = 323,
	KP4        = 324,
	KP5        = 325,
	KP6        = 326,
	KP7        = 327,
	KP8        = 328,
	KP9        = 329,
	KPDecimal  = 330,
	KPDivide   = 331,
	KPMultiply = 332,
	KPSubtract = 333,
	KPAdd      = 334,
	KPEnter    = 335,
	KPEqual    = 336,

	LeftShift    = 340,
	LeftControl  = 341,
	LeftAlt      = 342,
	LeftSuper    = 343,
	RightShift   = 344,
	RightControl = 345,
	RightAlt     = 346,
	RightSuper   = 347,
	Menu         = 348
} Key;

inline std::ostream &operator<<(std::ostream &os, KeyCode key_code)
{
	os << static_cast<uint32_t>(key_code);
	return os;
}

#define KEY_SPACE ::Core::Key::Space
#define KEY_APOSTROPHE ::Core::Key::Apostrophe /* ' */
#define KEY_COMMA ::Core::Key::Comma           /* , */
#define KEY_MINUS ::Core::Key::Minus           /* - */
#define KEY_PERIOD ::Core::Key::Period         /* . */
#define KEY_SLASH ::Core::Key::Slash           /* / */
#define KEY_0 ::Core::Key::D0
#define KEY_1 ::Core::Key::D1
#define KEY_2 ::Core::Key::D2
#define KEY_3 ::Core::Key::D3
#define KEY_4 ::Core::Key::D4
#define KEY_5 ::Core::Key::D5
#define KEY_6 ::Core::Key::D6
#define KEY_7 ::Core::Key::D7
#define KEY_8 ::Core::Key::D8
#define KEY_9 ::Core::Key::D9
#define KEY_SEMICOLON ::Core::Key::Semicolon /* ; */
#define KEY_EQUAL ::Core::Key::Equal         /* = */
#define KEY_A ::Core::Key::A
#define KEY_B ::Core::Key::B
#define KEY_C ::Core::Key::C
#define KEY_D ::Core::Key::D
#define KEY_E ::Core::Key::E
#define KEY_F ::Core::Key::F
#define KEY_G ::Core::Key::G
#define KEY_H ::Core::Key::H
#define KEY_I ::Core::Key::I
#define KEY_J ::Core::Key::J
#define KEY_K ::Core::Key::K
#define KEY_L ::Core::Key::L
#define KEY_M ::Core::Key::M
#define KEY_N ::Core::Key::N
#define KEY_O ::Core::Key::O
#define KEY_P ::Core::Key::P
#define KEY_Q ::Core::Key::Q
#define KEY_R ::Core::Key::R
#define KEY_S ::Core::Key::S
#define KEY_T ::Core::Key::T
#define KEY_U ::Core::Key::U
#define KEY_V ::Core::Key::V
#define KEY_W ::Core::Key::W
#define KEY_X ::Core::Key::X
#define KEY_Y ::Core::Key::Y
#define KEY_Z ::Core::Key::Z
#define KEY_LEFT_BRACKET ::Core::Key::LeftBracket   /* [ */
#define KEY_BACKSLASH ::Core::Key::Backslash        /* \ */
#define KEY_RIGHT_BRACKET ::Core::Key::RightBracket /* ] */
#define KEY_GRAVE_ACCENT ::Core::Key::GraveAccent   /* ` */
#define KEY_WORLD_1 ::Core::Key::World1             /* non-US #1 */
#define KEY_WORLD_2 ::Core::Key::World2             /* non-US #2 */

/* Function keys */
#define KEY_ESCAPE ::Core::Key::Escape
#define KEY_ENTER ::Core::Key::Enter
#define KEY_TAB ::Core::Key::Tab
#define KEY_BACKSPACE ::Core::Key::Backspace
#define KEY_INSERT ::Core::Key::Insert
#define KEY_DELETE ::Core::Key::Delete
#define KEY_RIGHT ::Core::Key::Right
#define KEY_LEFT ::Core::Key::Left
#define KEY_DOWN ::Core::Key::Down
#define KEY_UP ::Core::Key::Up
#define KEY_PAGE_UP ::Core::Key::PageUp
#define KEY_PAGE_DOWN ::Core::Key::PageDown
#define KEY_HOME ::Core::Key::Home
#define KEY_END ::Core::Key::End
#define KEY_CAPS_LOCK ::Core::Key::CapsLock
#define KEY_SCROLL_LOCK ::Core::Key::ScrollLock
#define KEY_NUM_LOCK ::Core::Key::NumLock
#define KEY_PRINT_SCREEN ::Core::Key::PrintScreen
#define KEY_PAUSE ::Core::Key::Pause
#define KEY_F1 ::Core::Key::F1
#define KEY_F2 ::Core::Key::F2
#define KEY_F3 ::Core::Key::F3
#define KEY_F4 ::Core::Key::F4
#define KEY_F5 ::Core::Key::F5
#define KEY_F6 ::Core::Key::F6
#define KEY_F7 ::Core::Key::F7
#define KEY_F8 ::Core::Key::F8
#define KEY_F9 ::Core::Key::F9
#define KEY_F10 ::Core::Key::F10
#define KEY_F11 ::Core::Key::F11
#define KEY_F12 ::Core::Key::F12
#define KEY_F13 ::Core::Key::F13
#define KEY_F14 ::Core::Key::F14
#define KEY_F15 ::Core::Key::F15
#define KEY_F16 ::Core::Key::F16
#define KEY_F17 ::Core::Key::F17
#define KEY_F18 ::Core::Key::F18
#define KEY_F19 ::Core::Key::F19
#define KEY_F20 ::Core::Key::F20
#define KEY_F21 ::Core::Key::F21
#define KEY_F22 ::Core::Key::F22
#define KEY_F23 ::Core::Key::F23
#define KEY_F24 ::Core::Key::F24
#define KEY_F25 ::Core::Key::F25

/* Keypad */
#define KEY_KP_0 ::Core::Key::KP0
#define KEY_KP_1 ::Core::Key::KP1
#define KEY_KP_2 ::Core::Key::KP2
#define KEY_KP_3 ::Core::Key::KP3
#define KEY_KP_4 ::Core::Key::KP4
#define KEY_KP_5 ::Core::Key::KP5
#define KEY_KP_6 ::Core::Key::KP6
#define KEY_KP_7 ::Core::Key::KP7
#define KEY_KP_8 ::Core::Key::KP8
#define KEY_KP_9 ::Core::Key::KP9
#define KEY_KP_DECIMAL ::Core::Key::KPDecimal
#define KEY_KP_DIVIDE ::Core::Key::KPDivide
#define KEY_KP_MULTIPLY ::Core::Key::KPMultiply
#define KEY_KP_SUBTRACT ::Core::Key::KPSubtract
#define KEY_KP_ADD ::Core::Key::KPAdd
#define KEY_KP_ENTER ::Core::Key::KPEnter
#define KEY_KP_EQUAL ::Core::Key::KPEqual

#define KEY_LEFT_SHIFT ::Core::Key::LeftShift
#define KEY_LEFT_CONTROL ::Core::Key::LeftControl
#define KEY_LEFT_ALT ::Core::Key::LeftAlt
#define KEY_LEFT_SUPER ::Core::Key::LeftSuper
#define KEY_RIGHT_SHIFT ::Core::Key::RightShift
#define KEY_RIGHT_CONTROL ::Core::Key::RightControl
#define KEY_RIGHT_ALT ::Core::Key::RightAlt
#define KEY_RIGHT_SUPER ::Core::Key::RightSuper
#define KEY_MENU ::Core::Key::Menu

typedef enum class MouseCode : uint32_t
{
	// From glfw3.h
	Button0 = 0,
	Button1 = 1,
	Button2 = 2,
	Button3 = 3,
	Button4 = 4,
	Button5 = 5,
	Button6 = 6,
	Button7 = 7,

	ButtonLast   = Button7,
	ButtonLeft   = Button0,
	ButtonRight  = Button1,
	ButtonMiddle = Button2
} Mouse;

inline std::ostream &operator<<(std::ostream &os, MouseCode mouse_code)
{
	os << static_cast<uint32_t>(mouse_code);
	return os;
}

#define MOUSE_BUTTON_0 ::Chaf::Mouse::Button0
#define MOUSE_BUTTON_1 ::Chaf::Mouse::Button1
#define MOUSE_BUTTON_2 ::Chaf::Mouse::Button2
#define MOUSE_BUTTON_3 ::Chaf::Mouse::Button3
#define MOUSE_BUTTON_4 ::Chaf::Mouse::Button4
#define MOUSE_BUTTON_5 ::Chaf::Mouse::Button5
#define MOUSE_BUTTON_6 ::Chaf::Mouse::Button6
#define MOUSE_BUTTON_7 ::Chaf::Mouse::Button7
#define MOUSE_BUTTON_LAST ::Chaf::Mouse::ButtonLast
#define MOUSE_BUTTON_LEFT ::Chaf::Mouse::ButtonLeft
#define MOUSE_BUTTON_RIGHT ::Chaf::Mouse::ButtonRight
#define MOUSE_BUTTON_MIDDLE ::Chaf::Mouse::ButtonMiddle
}        // namespace Ilum::Core