RWTexture2D<float4> OutputImage;

[numthreads(8, 8, 1)]
void MainCS(int2 DispatchID : SV_DispatchThreadID)
{
    OutputImage[DispatchID] = float4(float(DispatchID.x) / 100, float(DispatchID.y) / 100, 0, 1);
    //OutputImage[DispatchID] = 1.f; 
}