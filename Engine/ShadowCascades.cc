#include "pch.h"

#include "ShadowCascades.hpp"

#include <algorithm>
#include <cmath>

using DirectX::SimpleMath::Matrix;
using DirectX::SimpleMath::Vector3;
using DirectX::SimpleMath::Vector4;

namespace NeuralModelIntegrateTestbed {

DirectX::SimpleMath::Vector3 SunDirectionFromEuler(float pitchDegrees,
                                                   float yawDegrees) {
    const float pitch = DirectX::XMConvertToRadians(pitchDegrees);
    const float yaw = DirectX::XMConvertToRadians(yawDegrees);
    const float horizontal = std::cos(pitch);
    // Unit by construction: the squares multiply out to cos^2 + sin^2 over
    // both angles, so nothing here needs normalising afterwards.
    return DirectX::SimpleMath::Vector3(horizontal * std::sin(yaw), std::sin(pitch),
                                        -horizontal * std::cos(yaw));
}

void SunEulerFromDirection(const DirectX::SimpleMath::Vector3& direction,
                           float& outPitchDegrees, float& outYawDegrees) {
    DirectX::SimpleMath::Vector3 unit = direction;
    const float length = unit.Length();
    // A zero direction has no angles to report. Straight down is the useful
    // answer rather than NaN, and it is what an absent sun defaults to.
    unit = length > 1e-6f ? unit / length
                          : DirectX::SimpleMath::Vector3(0.0f, -1.0f, 0.0f);
    outPitchDegrees = DirectX::XMConvertToDegrees(
        std::asin(std::max(-1.0f, std::min(1.0f, unit.y))));
    outYawDegrees = DirectX::XMConvertToDegrees(std::atan2(unit.x, -unit.z));
}

namespace {

// A grid wide enough for the cascades: 1 for one, 2x2 for two to four.
std::uint32_t AtlasGridSide(int cascadeCount) {
    if (cascadeCount <= 1) {
        return 1;
    }
    return 2;
}

// The camera's near and far plane distances, read back out of the projection
// rather than passed in separately, so there is no second copy to disagree.
//
// For the right-handed perspective SimpleMath builds, _33 = f/(n-f) and
// _43 = n*f/(n-f), which solve to n = _43/_33 and f = _43/(_33+1).
bool NearFarFromProjection(const Matrix& projection, float& outNear, float& outFar) {
    const float m33 = projection._33;
    const float m43 = projection._43;
    if (std::fabs(m33) < 1e-9f || std::fabs(m33 + 1.0f) < 1e-9f) {
        return false;
    }
    outNear = m43 / m33;
    outFar = m43 / (m33 + 1.0f);
    return outNear > 0.0f && outFar > outNear;
}

}  // namespace

bool SceneSunViewProjection(const Vector3& lightDirection, const Vector3& sceneMin,
                            const Vector3& sceneMax, std::uint32_t resolution,
                            Matrix& outViewProjection, float& outTexelWorldSize) {
    Vector3 direction = lightDirection;
    const float length = direction.Length();
    if (length < 1e-6f || resolution == 0) {
        return false;
    }
    direction /= length;

    const Vector3 centre = (sceneMin + sceneMax) * 0.5f;
    const float radius = (sceneMax - sceneMin).Length() * 0.5f;
    if (!(radius > 0.0f)) {
        return false;
    }

    // Same guard as the cascade fit: a sun straight down the chosen up axis
    // makes CreateLookAt degenerate.
    Vector3 up = Vector3::UnitY;
    if (std::fabs(direction.Dot(up)) > 0.99f) {
        up = Vector3::UnitZ;
    }

    // The sphere contains the whole scene, so pulling the eye back by its
    // radius clears every caster and a far plane of twice the radius reaches
    // past the last of them. No scene-extent search is needed the way the
    // cascade fit needs one, because here the fit IS the scene.
    const Vector3 eye = centre - direction * radius;
    const Matrix view = Matrix::CreateLookAt(eye, centre, up);
    const Matrix projection = Matrix::CreateOrthographicOffCenter(
        -radius, radius, -radius, radius, 0.0f, 2.0f * radius);

    outViewProjection = view * projection;
    outTexelWorldSize = (2.0f * radius) / static_cast<float>(resolution);
    return true;
}

void ShadowCascadeAtlasRect(int index, int cascadeCount, std::uint32_t resolution,
                            std::uint32_t& outX, std::uint32_t& outY,
                            std::uint32_t& outSize, std::uint32_t& outAtlasResolution) {
    const std::uint32_t side = AtlasGridSide(cascadeCount);
    outSize = resolution;
    outAtlasResolution = resolution * side;
    const std::uint32_t column = static_cast<std::uint32_t>(index) % side;
    const std::uint32_t row = static_cast<std::uint32_t>(index) / side;
    outX = column * resolution;
    outY = row * resolution;
}

ShadowCascades ComputeShadowCascades(const Matrix& view, const Matrix& projection,
                                     const Vector3& lightDirection,
                                     const Vector3& sceneMin, const Vector3& sceneMax,
                                     const ShadowCascadeSetup& setup,
                                     const ShadowCascades* previous) {
    ShadowCascades result;

    Vector3 direction = lightDirection;
    const float directionLength = direction.Length();
    if (directionLength < 1e-6f) {
        return result;
    }
    direction /= directionLength;

    const int count = std::min(std::max(setup.cascadeCount, 1), kMaxShadowCascades);
    if (setup.resolution == 0 || setup.shadowDistance <= 0.0f) {
        return result;
    }

    float cameraNear = 0.0f;
    float cameraFar = 0.0f;
    if (!NearFarFromProjection(projection, cameraNear, cameraFar)) {
        return result;
    }
    const float shadowFar = std::min(cameraFar, cameraNear + setup.shadowDistance);
    if (shadowFar <= cameraNear) {
        return result;
    }

    // The full frustum's eight world-space corners. Everything below is a lerp
    // between a near corner and the matching far one, which is linear in VIEW
    // depth -- exactly what a split expressed as a distance needs.
    const Matrix inverseViewProjection = (view * projection).Invert();
    Vector3 nearCorner[4];
    Vector3 farCorner[4];
    {
        const float xs[4] = {-1.0f, 1.0f, 1.0f, -1.0f};
        const float ys[4] = {-1.0f, -1.0f, 1.0f, 1.0f};
        for (int i = 0; i < 4; ++i) {
            // D3D clip space: z = 0 is the near plane, z = 1 the far one.
            const Vector4 nearClip(xs[i], ys[i], 0.0f, 1.0f);
            const Vector4 farClip(xs[i], ys[i], 1.0f, 1.0f);
            const Vector4 nearWorld = Vector4::Transform(nearClip, inverseViewProjection);
            const Vector4 farWorld = Vector4::Transform(farClip, inverseViewProjection);
            if (std::fabs(nearWorld.w) < 1e-12f || std::fabs(farWorld.w) < 1e-12f) {
                return result;
            }
            nearCorner[i] = Vector3(nearWorld.x, nearWorld.y, nearWorld.z) / nearWorld.w;
            farCorner[i] = Vector3(farWorld.x, farWorld.y, farWorld.z) / farWorld.w;
        }
    }

    // An up vector that is not the light direction. A sun straight overhead is
    // the common case that breaks the obvious choice.
    Vector3 up = Vector3::UnitY;
    if (std::fabs(direction.Dot(up)) > 0.99f) {
        up = Vector3::UnitZ;
    }

    // How far the scene extends along the light direction, so the fitted depth
    // range reaches every caster rather than only the slice.
    float sceneAlongLightMin = FLT_MAX;
    float sceneAlongLightMax = -FLT_MAX;
    for (int corner = 0; corner < 8; ++corner) {
        const Vector3 point((corner & 1) ? sceneMax.x : sceneMin.x,
                            (corner & 2) ? sceneMax.y : sceneMin.y,
                            (corner & 4) ? sceneMax.z : sceneMin.z);
        const float along = point.Dot(direction);
        sceneAlongLightMin = std::min(sceneAlongLightMin, along);
        sceneAlongLightMax = std::max(sceneAlongLightMax, along);
    }
    if (sceneAlongLightMin > sceneAlongLightMax) {
        sceneAlongLightMin = 0.0f;
        sceneAlongLightMax = 0.0f;
    }

    result.count = count;
    result.lightDirection = direction;
    result.sceneMin = sceneMin;
    result.sceneMax = sceneMax;

    // A previous fit is only reusable if it was made for the same light and the
    // same scene bounds -- its held spheres and its fitted depth range were
    // chosen against both -- and for the same cascade layout.
    const bool previousComparable =
        previous != nullptr && previous->count == count &&
        previous->atlasResolution != 0 &&
        (previous->lightDirection - direction).LengthSquared() < 1e-12f &&
        (previous->sceneMin - sceneMin).LengthSquared() < 1e-12f &&
        (previous->sceneMax - sceneMax).LengthSquared() < 1e-12f;

    for (int cascade = 0; cascade < count; ++cascade) {
        // Practical split scheme: blend a uniform division of the range with a
        // logarithmic one.
        const auto splitAt = [&](int index) -> float {
            if (index <= 0) {
                return cameraNear;
            }
            if (index >= count) {
                return shadowFar;
            }
            const float fraction = static_cast<float>(index) / static_cast<float>(count);
            const float uniform = cameraNear + (shadowFar - cameraNear) * fraction;
            const float logarithmic =
                cameraNear * std::pow(shadowFar / cameraNear, fraction);
            return uniform + setup.splitLambda * (logarithmic - uniform);
        };
        const float sliceNear = splitAt(cascade);
        const float sliceFar = splitAt(cascade + 1);

        // Corners of this slice, by depth fraction along the full frustum.
        const float nearT = (sliceNear - cameraNear) / (cameraFar - cameraNear);
        const float farT = (sliceFar - cameraNear) / (cameraFar - cameraNear);
        Vector3 slice[8];
        for (int i = 0; i < 4; ++i) {
            slice[i] = nearCorner[i] + (farCorner[i] - nearCorner[i]) * nearT;
            slice[i + 4] = nearCorner[i] + (farCorner[i] - nearCorner[i]) * farT;
        }

        // Bounding sphere, so the fit does not change with camera rotation.
        Vector3 sliceCentre = Vector3::Zero;
        for (const Vector3& point : slice) {
            sliceCentre += point;
        }
        sliceCentre /= 8.0f;
        float sliceRadius = 0.0f;
        for (const Vector3& point : slice) {
            sliceRadius = std::max(sliceRadius, (point - sliceCentre).Length());
        }
        // Nudged up so a point exactly on the sphere is comfortably inside the
        // projection rather than on its boundary.
        sliceRadius = std::max(sliceRadius * 1.001f, 1e-4f);

        // Hold the previous fit while this slice is still wholly inside the
        // sphere it covered. That is the containment test for two spheres:
        // the new one fits if its centre is within (held - new) of the held
        // centre. Only the projection is carried over -- the split distance is
        // recomputed above either way, since it depends on the camera's near
        // and far planes rather than on where the camera is.
        if (previousComparable) {
            const ShadowCascade& held = previous->cascades[cascade];
            if (held.fittedRadius >= sliceRadius &&
                (sliceCentre - held.fittedCentre).Length() <=
                    held.fittedRadius - sliceRadius) {
                ShadowCascade& reused = result.cascades[cascade];
                reused = held;
                reused.splitDistance = sliceFar;
                reused.refitted = false;
                result.atlasResolution = previous->atlasResolution;
                continue;
            }
        }

        // Padded, so the next few frames of camera motion land inside this fit
        // rather than forcing another one.
        Vector3 centre = sliceCentre;
        const float radius = sliceRadius * (1.0f + std::max(setup.fitMargin, 0.0f));

        const float texelWorldSize = (2.0f * radius) / static_cast<float>(setup.resolution);

        // Snap the centre to whole texels, in a light-aligned basis. Done on
        // the centre's components along the light's right and up axes, which is
        // the same thing as snapping in light space but needs no provisional
        // matrix to do it.
        Vector3 forward = direction;
        Vector3 right = up.Cross(forward);
        if (right.Length() < 1e-6f) {
            right = Vector3::UnitX;
        }
        right.Normalize();
        const Vector3 trueUp = forward.Cross(right);

        const auto snap = [texelWorldSize](float value) {
            return std::floor(value / texelWorldSize) * texelWorldSize;
        };
        const float alongRight = snap(centre.Dot(right));
        const float alongUp = snap(centre.Dot(trueUp));
        const float alongForward = centre.Dot(forward);
        centre = right * alongRight + trueUp * alongUp + forward * alongForward;

        // The light's eye is pulled back far enough to clear every caster, and
        // the far plane pushed past the last of them.
        const float centreAlongLight = centre.Dot(direction);
        const float backOff = (centreAlongLight - sceneAlongLightMin) + radius;
        const Vector3 eye = centre - direction * backOff;
        const Matrix lightView = Matrix::CreateLookAt(eye, centre, up);

        const float nearPlane = 0.0f;
        const float farPlane =
            backOff + (sceneAlongLightMax - centreAlongLight) + radius;

        const Matrix lightProjection = Matrix::CreateOrthographicOffCenter(
            -radius, radius, -radius, radius, nearPlane, farPlane);

        ShadowCascade& out = result.cascades[cascade];
        out.viewProjection = lightView * lightProjection;
        out.splitDistance = sliceFar;
        out.texelWorldSize = texelWorldSize;
        // The sphere this fit covers, for the containment test above. Centred
        // on the SNAPPED centre, which is what the projection was built from.
        out.fittedCentre = centre;
        // Two texels short of what the projection actually spans. Snapping
        // moved the centre by up to a texel along each of two axes, so the
        // sphere guaranteed covered is slightly smaller than the one fitted;
        // being conservative here costs a refit now and then and nothing else,
        // while being optimistic would let a slice sit fractionally outside
        // the map it is sampled from.
        out.fittedRadius = std::max(radius - 2.0f * texelWorldSize, 0.0f);
        out.refitted = true;

        std::uint32_t x = 0, y = 0, size = 0, atlas = 0;
        ShadowCascadeAtlasRect(cascade, count, setup.resolution, x, y, size, atlas);
        result.atlasResolution = atlas;
        const float inverseAtlas = 1.0f / static_cast<float>(atlas);
        out.atlasScaleOffset =
            Vector4(static_cast<float>(size) * inverseAtlas,
                    static_cast<float>(size) * inverseAtlas,
                    static_cast<float>(x) * inverseAtlas,
                    static_cast<float>(y) * inverseAtlas);
    }

    return result;
}

}  // namespace NeuralModelIntegrateTestbed
