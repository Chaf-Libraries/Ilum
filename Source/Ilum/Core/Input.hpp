#pragma once

#include "Precompile.hpp"
#include "Singleton.hpp"
#include "Window.hpp"

#include <glm/glm.hpp>

namespace Ilum
{
/*KeyCode*/
typedef enum class KeyCode : uint16_t
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

inline std::ostream &operator<<(std::ostream &os, KeyCode keyCode)
{
	os << static_cast<int32_t>(keyCode);
	return os;
}

// From glfw3.h
#define KEY_SPACE ::Key::Space
#define KEY_APOSTROPHE ::Key::Apostrophe /* ' */
#define KEY_COMMA ::Key::Comma           /* , */
#define KEY_MINUS ::Key::Minus           /* - */
#define KEY_PERIOD ::Key::Period         /* . */
#define KEY_SLASH ::Key::Slash           /* / */
#define KEY_0 ::Key::D0
#define KEY_1 ::Key::D1
#define KEY_2 ::Key::D2
#define KEY_3 ::Key::D3
#define KEY_4 ::Key::D4
#define KEY_5 ::Key::D5
#define KEY_6 ::Key::D6
#define KEY_7 ::Key::D7
#define KEY_8 ::Key::D8
#define KEY_9 ::Key::D9
#define KEY_SEMICOLON ::Key::Semicolon /* ; */
#define KEY_EQUAL ::Key::Equal         /* = */
#define KEY_A ::Key::A
#define KEY_B ::Key::B
#define KEY_C ::Key::C
#define KEY_D ::Key::D
#define KEY_E ::Key::E
#define KEY_F ::Key::F
#define KEY_G ::Key::G
#define KEY_H ::Key::H
#define KEY_I ::Key::I
#define KEY_J ::Key::J
#define KEY_K ::Key::K
#define KEY_L ::Key::L
#define KEY_M ::Key::M
#define KEY_N ::Key::N
#define KEY_O ::Key::O
#define KEY_P ::Key::P
#define KEY_Q ::Key::Q
#define KEY_R ::Key::R
#define KEY_S ::Key::S
#define KEY_T ::Key::T
#define KEY_U ::Key::U
#define KEY_V ::Key::V
#define KEY_W ::Key::W
#define KEY_X ::Key::X
#define KEY_Y ::Key::Y
#define KEY_Z ::Key::Z
#define KEY_LEFT_BRACKET ::Key::LeftBracket   /* [ */
#define KEY_BACKSLASH ::Key::Backslash        /* \ */
#define KEY_RIGHT_BRACKET ::Key::RightBracket /* ] */
#define KEY_GRAVE_ACCENT ::Key::GraveAccent   /* ` */
#define KEY_WORLD_1 ::Key::World1             /* non-US #1 */
#define KEY_WORLD_2 ::Key::World2             /* non-US #2 */

/* Function keys */
#define KEY_ESCAPE ::Key::Escape
#define KEY_ENTER ::Key::Enter
#define KEY_TAB ::Key::Tab
#define KEY_BACKSPACE ::Key::Backspace
#define KEY_INSERT ::Key::Insert
#define KEY_DELETE ::Key::Delete
#define KEY_RIGHT ::Key::Right
#define KEY_LEFT ::Key::Left
#define KEY_DOWN ::Key::Down
#define KEY_UP ::Key::Up
#define KEY_PAGE_UP ::Key::PageUp
#define KEY_PAGE_DOWN ::Key::PageDown
#define KEY_HOME ::Key::Home
#define KEY_END ::Key::End
#define KEY_CAPS_LOCK ::Key::CapsLock
#define KEY_SCROLL_LOCK ::Key::ScrollLock
#define KEY_NUM_LOCK ::Key::NumLock
#define KEY_PRINT_SCREEN ::Key::PrintScreen
#define KEY_PAUSE ::Key::Pause
#define KEY_F1 ::Key::F1
#define KEY_F2 ::Key::F2
#define KEY_F3 ::Key::F3
#define KEY_F4 ::Key::F4
#define KEY_F5 ::Key::F5
#define KEY_F6 ::Key::F6
#define KEY_F7 ::Key::F7
#define KEY_F8 ::Key::F8
#define KEY_F9 ::Key::F9
#define KEY_F10 ::Key::F10
#define KEY_F11 ::Key::F11
#define KEY_F12 ::Key::F12
#define KEY_F13 ::Key::F13
#define KEY_F14 ::Key::F14
#define KEY_F15 ::Key::F15
#define KEY_F16 ::Key::F16
#define KEY_F17 ::Key::F17
#define KEY_F18 ::Key::F18
#define KEY_F19 ::Key::F19
#define KEY_F20 ::Key::F20
#define KEY_F21 ::Key::F21
#define KEY_F22 ::Key::F22
#define KEY_F23 ::Key::F23
#define KEY_F24 ::Key::F24
#define KEY_F25 ::Key::F25

/* Keypad */
#define KEY_KP_0 ::Key::KP0
#define KEY_KP_1 ::Key::KP1
#define KEY_KP_2 ::Key::KP2
#define KEY_KP_3 ::Key::KP3
#define KEY_KP_4 ::Key::KP4
#define KEY_KP_5 ::Key::KP5
#define KEY_KP_6 ::Key::KP6
#define KEY_KP_7 ::Key::KP7
#define KEY_KP_8 ::Key::KP8
#define KEY_KP_9 ::Key::KP9
#define KEY_KP_DECIMAL ::Key::KPDecimal
#define KEY_KP_DIVIDE ::Key::KPDivide
#define KEY_KP_MULTIPLY ::Key::KPMultiply
#define KEY_KP_SUBTRACT ::Key::KPSubtract
#define KEY_KP_ADD ::Key::KPAdd
#define KEY_KP_ENTER ::Key::KPEnter
#define KEY_KP_EQUAL ::Key::KPEqual

#define KEY_LEFT_SHIFT ::Key::LeftShift
#define KEY_LEFT_CONTROL ::Key::LeftControl
#define KEY_LEFT_ALT ::Key::LeftAlt
#define KEY_LEFT_SUPER ::Key::LeftSuper
#define KEY_RIGHT_SHIFT ::Key::RightShift
#define KEY_RIGHT_CONTROL ::Key::RightControl
#define KEY_RIGHT_ALT ::Key::RightAlt
#define KEY_RIGHT_SUPER ::Key::RightSuper
#define KEY_MENU ::Key::Menu

////////////////////////////////////////////////////////////////////////////

/*Mouse Code*/
typedef enum class MouseCode : uint16_t
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

inline std::ostream &operator<<(std::ostream &os, MouseCode mouseCode)
{
	os << static_cast<int32_t>(mouseCode);
	return os;
}

#define MOUSE_BUTTON_0 ::Mouse::Button0
#define MOUSE_BUTTON_1 ::Mouse::Button1
#define MOUSE_BUTTON_2 ::Mouse::Button2
#define MOUSE_BUTTON_3 ::Mouse::Button3
#define MOUSE_BUTTON_4 ::Mouse::Button4
#define MOUSE_BUTTON_5 ::Mouse::Button5
#define MOUSE_BUTTON_6 ::Mouse::Button6
#define MOUSE_BUTTON_7 ::Mouse::Button7
#define MOUSE_BUTTON_LAST ::Mouse::ButtonLast
#define MOUSE_BUTTON_LEFT ::Mouse::ButtonLeft
#define MOUSE_BUTTON_RIGHT ::Mouse::ButtonRight
#define MOUSE_BUTTON_MIDDLE ::Mouse::ButtonMiddle

class Input : public Singleton<Input>
{
  public:
	Input()  = default;
	~Input() = default;

	void Bind(Window *window);

	bool IsKeyPressed(KeyCode keycode);
	bool IsMouseButtonPressed(MouseCode button);

	glm::vec2 GetMousePosition();

	void SetCursorPosition(const glm::vec2 &pos);

  private:
	Window *m_window;
};
}        // namespace Ilum