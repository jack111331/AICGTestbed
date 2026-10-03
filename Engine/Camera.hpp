//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once


class Camera
{
public:
    Camera();
    ~Camera();

    void Get3DViewProjMatricesLH(DirectX::XMFLOAT4X4 *view, DirectX::XMFLOAT4X4 *proj, float fovInDegrees, float screenWidth, float screenHeight);
    void Get3DViewProjMatrices(DirectX::XMFLOAT4X4 *view, DirectX::XMFLOAT4X4 *proj, float fovInDegrees, float screenWidth, float screenHeight);
    void Reset();
    void Set(DirectX::XMVECTOR eye, DirectX::XMVECTOR at, DirectX::XMVECTOR up);
    void MoveForward(float dist);
    void MoveUpward(float dist);
    void MoveRightward(float dist);
    static Camera *get();
    void RotateAroundYAxis(float angleRad);
    void RotateYaw(float angleRad);
    void RotatePitch(float angleRad);
    void GetOrthoProjMatrices(DirectX::XMFLOAT4X4 *view, DirectX::XMFLOAT4X4 *proj, float width, float height);
    DirectX::XMVECTOR mEye; // Where the camera is in world space. Z increases into of the screen when using LH coord system (which we are and DX uses)
    DirectX::XMVECTOR mAt; // What the camera is looking at (world origin)
    DirectX::XMVECTOR mUp; // Which way is up
private:
    static Camera* mCamera;
};