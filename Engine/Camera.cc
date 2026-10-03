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

#include "pch.h"
#include "Camera.hpp"

using namespace DirectX;

Camera* Camera::mCamera = nullptr;

Camera::Camera()
{
    Reset();
    mCamera = this;
}

Camera::~Camera()
{
    mCamera = nullptr;
}

Camera* Camera::get()
{
    return mCamera;
}

void Camera::Get3DViewProjMatricesLH(XMFLOAT4X4 *view, XMFLOAT4X4 *proj, float fovInDegrees, float screenWidth, float screenHeight)
{

    float aspectRatio = (float)screenWidth / (float)screenHeight;
    float fovAngleY = fovInDegrees * XM_PI / 180.0f;

    if (aspectRatio < 1.0f)
    {
        fovAngleY /= aspectRatio;
    }

    XMStoreFloat4x4(view, XMMatrixLookAtLH(mEye, mAt, mUp));
    XMStoreFloat4x4(proj, XMMatrixPerspectiveFovLH(fovAngleY, aspectRatio, 0.01f, 125.0f));
}

void Camera::Get3DViewProjMatrices(XMFLOAT4X4 *view, XMFLOAT4X4 *proj, float fovInDegrees, float screenWidth, float screenHeight)
{
    
    float aspectRatio = (float)screenWidth / (float)screenHeight;
    float fovAngleY = fovInDegrees * XM_PI / 180.0f;

    if (aspectRatio < 1.0f)
    {
        fovAngleY /= aspectRatio;
    }

    XMStoreFloat4x4(view, XMMatrixTranspose(XMMatrixLookAtRH(mEye, mAt, mUp)));
    XMStoreFloat4x4(proj, XMMatrixTranspose(XMMatrixPerspectiveFovRH(fovAngleY, aspectRatio, 0.01f, 125.0f)));
}

void Camera::GetOrthoProjMatrices(XMFLOAT4X4 *view, XMFLOAT4X4 *proj, float width, float height)
{
    XMStoreFloat4x4(view, XMMatrixTranspose(XMMatrixLookAtRH(mEye, mAt, mUp)));
    XMStoreFloat4x4(proj, XMMatrixTranspose(XMMatrixOrthographicRH(width, height, 0.01f, 125.0f)));
}

void Camera::RotateAroundYAxis(float angleRad)
{
    XMMATRIX rotation = XMMatrixRotationY(angleRad);

    mEye = mAt + XMVector3TransformCoord(mEye - mAt, rotation);
    mUp = XMVector3TransformCoord(mUp, rotation);
}

void Camera::RotateYaw(float angleRad)
{
    XMMATRIX rotation = XMMatrixRotationAxis(mUp, angleRad);

    mEye = mAt + XMVector3TransformCoord(mEye - mAt, rotation);
}

void Camera::RotatePitch(float angleRad)
{
    XMVECTOR right = XMVector3Normalize(XMVector3Cross(mAt - mEye, mUp));
    XMMATRIX rotation = XMMatrixRotationAxis(right, angleRad);

    mEye = mAt + XMVector3TransformCoord(mEye - mAt, rotation);
    mUp = XMVector3TransformCoord(mUp, rotation);
}

void Camera::Reset()
{
    mEye = XMVectorSet(0.0f, 0.7f, 1.5f, 0.0f);
    mAt = XMVectorSet(0.0f, -0.1f, 0.0f, 0.0f);
    mUp = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
}

void Camera::Set(XMVECTOR eye, XMVECTOR at, XMVECTOR up)
{
    mEye = eye;
    mAt = at;
    mUp = up;
}

void Camera::MoveForward(float dist) {
    XMVECTOR normalizedForward = XMVector3Normalize(mAt - mEye);
    mEye += dist * normalizedForward;
    mAt += dist * normalizedForward;
}

void Camera::MoveUpward(float dist) {
    XMVECTOR normalizedUp = XMVector3Normalize(mUp);
    mEye += dist * normalizedUp;
    mAt += dist * normalizedUp;
}

void Camera::MoveRightward(float dist) {
    XMVECTOR normalizedRight = XMVector3Normalize(XMVector3Cross(mAt - mEye, mUp));
    mEye += dist * normalizedRight;
    mAt += dist * normalizedRight;
}