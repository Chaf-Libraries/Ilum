#ifndef COMPLEX_HLSLI
#define COMPLEX_HLSLI

struct Complex
{
    float re;
    float im;
};

Complex CreateComplex(float re, float im)
{
    Complex result;
    result.re = re;
    result.im = im;
    return result;
}

Complex ComplexFromReal(float x)
{
    Complex result;
    result.re = x;
    result.im = 0;
    return result;
}

Complex Add(Complex lhs, Complex rhs)
{
    Complex result;
    result.re = lhs.re + rhs.re;
    result.im = lhs.im + rhs.im;
    return result;
}

Complex Sub(Complex lhs, Complex rhs)
{
    Complex result;
    result.re = lhs.re - rhs.re;
    result.im = lhs.im - rhs.im;
    return result;
}

Complex Mul(Complex lhs, Complex rhs)
{
    Complex result;
    float a = lhs.re;
    float b = lhs.im;
    float c = rhs.re;
    float d = rhs.im;
    
    result.re = a * c - b * d;
    result.im = a * d + b * c;
    
    return result;
}

Complex Div(Complex lhs, Complex rhs)
{
    Complex result;
    float a = lhs.re;
    float b = lhs.im;
    float c = rhs.re;
    float d = rhs.im;
    
    float s = rhs.re * rhs.re + rhs.im * rhs.im;
    result.re = (a * c + b * d) / s;
    result.im = (b * c - a * d) / s;
    
    return result;
}

float Norm(Complex x)
{
    return x.re * x.re + x.im * x.im;
}

float Abs(Complex x)
{
    return sqrt(Norm(x));
}

Complex Sqrt(Complex x)
{
    Complex result;
    
    float n = Abs(x);
    float t1 = sqrt(0.5 * (n + abs(x.re)));
    float t2 = 0.5 * x.im / t1;

    if (n == 0)
    {
        return ComplexFromReal(0);
    }

    if (x.re >= 0)
    {
        return CreateComplex(t1, t2);
    }
    else
    {
        return CreateComplex(abs(t2), sign(t1) * x.im);
    }
    
    return ComplexFromReal(0);
}

#endif