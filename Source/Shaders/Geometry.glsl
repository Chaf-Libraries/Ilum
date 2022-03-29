#ifndef _GEOMETRY_GLSL
#define _GEOMETRY_GLSL

const float PI      = 3.14159265358979323846;
const float InvPI   = 0.31830988618379067154;
const float Inv2PI  = 0.15915494309189533577;
const float Inv4PI  = 0.07957747154594766788;
const float PIOver2 = 1.57079632679489661923;
const float PIOver4 = 0.78539816339744830961;
const float Sqrt2   = 1.41421356237309504880;

const float ShadowEpsilon = 0.0001;

float AbsCosTheta(vec3 w)
{
	return abs(w.z);
}

float CosTheta(vec3 w)
{
	return w.z;
}

float Cos2Theta(vec3 w)
{
	return w.z * w.z;
}

float Sin2Theta(vec3 w)
{
	return max(0.0, 1.0 - Cos2Theta(w));
}

float Tan2Theta(vec3 w)
{
	return Sin2Theta(w) / Cos2Theta(w);
}

float SinTheta(vec3 w)
{
	return sqrt(Sin2Theta(w));
}

float SinPhi(vec3 w)
{
	float sinTheta = SinTheta(w);
	return sinTheta == 0.0 ? 0.0 : clamp(w.y / sinTheta, -1.0, 1.0);
}

float Sin2Phi(vec3 w)
{
	return SinPhi(w) * SinPhi(w);
}

float CosPhi(vec3 w)
{
	float sinTheta = SinTheta(w);
	return sinTheta == 0.0 ? 1.0 : clamp(w.x / sinTheta, -1.0, 1.0);
}

float Cos2Phi(vec3 w)
{
	return CosPhi(w) * CosPhi(w);
}

float TanTheta(vec3 w)
{
	return SinTheta(w) / CosTheta(w);
}

vec3 SphericalDirection(float sinTheta, float cosTheta, float phi)
{
	return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

vec3 Faceforward(vec3 v, vec3 v2)
{
	return dot(v, v2) < 0.0 ? -v : v;
}

float ErfInv(float x)
{
	float w, p;
	x = clamp(x, -.99999, .99999);
	w = -log((1 - x) * (1 + x));
	if (w < 5)
	{
		w = w - 2.5;
		p = 2.81022636e-08;
		p = 3.43273939e-07 + p * w;
		p = -3.5233877e-06 + p * w;
		p = -4.39150654e-06 + p * w;
		p = 0.00021858087 + p * w;
		p = -0.00125372503 + p * w;
		p = -0.00417768164 + p * w;
		p = 0.246640727 + p * w;
		p = 1.50140941 + p * w;
	}
	else
	{
		w = sqrt(w) - 3;
		p = -0.000200214257;
		p = 0.000100950558 + p * w;
		p = 0.00134934322 + p * w;
		p = -0.00367342844 + p * w;
		p = 0.00573950773 + p * w;
		p = -0.0076224613 + p * w;
		p = 0.00943887047 + p * w;
		p = 1.00167406 + p * w;
		p = 2.83297682 + p * w;
	}
	return p * x;
}

float Erf(float x)
{
	// constants
	float a1 = 0.254829592f;
	float a2 = -0.284496736f;
	float a3 = 1.421413741f;
	float a4 = -1.453152027f;
	float a5 = 1.061405429f;
	float p  = 0.3275911f;

	// Save the sign of x
	int sign = 1;
	if (x < 0)
	{
		sign = -1;
	}
	x = abs(x);

	// A&S formula 7.1.26
	float t = 1 / (1 + p * x);
	float y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

	return sign * y;
}

bool SameHemisphere(vec3 w, vec3 wp)
{
	return w.z * wp.z > 0;
}

vec3 WorldToLocal(vec3 w, vec3 normal, vec3 tangent, vec3 bitangent)
{
	return vec3(dot(w, tangent), dot(w, bitangent), dot(w, normal));
}

vec3 LocalToWorld(vec3 w, vec3 normal, vec3 tangent, vec3 bitangent)
{
	return tangent * w.x + bitangent * w.y + normal * w.z;
}

float Radians(float deg)
{
	return PI / 180.0 * deg;
}

float Degrees(float rad)
{
	return rad * 180.0 / PI;
}

#endif