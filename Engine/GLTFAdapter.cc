#include "pch.h"
#include "GLTFAdapter.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <memory>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <directx/d3dx12.h>
#include <stb_image.h>
#include "Descriptors/DescriptorContext.hpp"
#include <no_texture.h>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {
    fastgltf::Expected<fastgltf::Asset> GLTFAdapter::Initialize(const std::string &gltfFilepath) {
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
        }

        std::ifstream ifs(gltfFilepath);
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string gltfData = ss.str();
        auto data = fastgltf::GltfDataBuffer::FromBytes(
            reinterpret_cast<const std::byte*>(gltfData.data()), gltfData.size());
        if (data.error() != fastgltf::Error::None) {
            return data.error();
        }
        // KHR_lights_punctual is opt-in: fastgltf leaves Asset::lights empty
        // and drops every Node::lightIndex unless it is requested here.
        fastgltf::Extensions extensions = fastgltf::Extensions::MYLAB_generative |
                                          fastgltf::Extensions::KHR_lights_punctual;
        fastgltf::Parser parser(extensions);
        std::filesystem::path absPath = std::filesystem::absolute(gltfFilepath);
        auto loadedGLTF = parser.loadGltfJson(data.get(), absPath.parent_path(), fastgltf::Options::LoadGLBBuffers | 
                             fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);
        if (loadedGLTF.error() == fastgltf::Error::None) {
            m_gltf = std::move(loadedGLTF.get());
        } else {
            return loadedGLTF.error();
        }
        return fastgltf::Error::None;

    }

    void GLTFAdapter::PrepareBuffer(std::shared_ptr<DX::DeviceResources> deviceResources) {
        auto device = deviceResources->GetD3DDevice();

        // The ported descriptor classes reach for the device and the frame
        // number through this context, so it has to exist before any of them
        // are constructed. Idempotent: safe to call again after a device reset.
        Descriptors::Context().Initialize(device);

        // CPU-visible staging descriptors, grown a page at a time. 256 per page
        // is the upstream default; glTF images are allocated one descriptor at
        // a time so a page covers 256 textures before another is created.
        m_srvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256);

        // GPU-visible heap that per-draw descriptor tables are copied into.
        m_srvDynamicHeap = std::make_unique<Descriptors::DynamicDescriptorHeap>(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1024);

        ResourceUploadBatch resourceUpload(device);

        const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

        resourceUpload.Begin();
        for (const auto &buffer: m_gltf.buffers) {

            const auto desc = CD3DX12_RESOURCE_DESC::Buffer(buffer.byteLength);

            Microsoft::WRL::ComPtr<ID3D12Resource> createdBuffer;
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(createdBuffer.GetAddressOf())));
            createdBuffer->SetName(L"GLTF buffer 1");
            buffers.push_back(createdBuffer);

            SetDebugObjectName(createdBuffer.Get(), L"ModelMeshPart");

            // Create staging buffer via GraphicsMemory and upload buffer data via memcpy, save allocation time using paging mechanism
            // https://github.com/microsoft/DirectXTK12/blob/f003171e0c6a30cc61444864ef756bd3cf2fc310/Src/GraphicsMemory.cpp#L126
            // TODO do we need to hold the SharedGraphicsResource even when the upload is finished?
            SharedGraphicsResource vertexBuffer = GraphicsMemory::Get(device).Allocate(buffer.byteLength, 16, GraphicsMemory::TAG_VERTEX);
            // access buffer.data (std::variant) using std::visitor like https://github.com/spnda/fastgltf/blob/main/examples/gl_viewer/gl_viewer.cpp#L516
            std::visit(fastgltf::visitor {
                [](const auto& arg) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Vector& vector) {
                    memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                },
                [&](const fastgltf::sources::CustomBuffer& customBuffer) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::ByteView& byteView) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Fallback& fallback) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const std::monostate& monostate) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::URI& filePath) {
                    // TODO We may specify file offset
                    assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.

                    const std::string path(filePath.uri.path().begin(), filePath.uri.path().end()); // Thanks C++.
                    std::ifstream meshBin(path.c_str(), std::ios::binary);
                    meshBin.read(reinterpret_cast<char*>(vertexBuffer.Memory()), filePath.fileByteOffset);
                    meshBin.close();
                },
                [&](const fastgltf::sources::Array& vector) {
                    memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                },
                [&](const fastgltf::sources::BufferView& view) {
                    // TODO haven't implement loading from view
                    auto& bufferView = m_gltf.bufferViews[view.bufferViewIndex];
                    const auto& buffer = m_gltf.buffers[bufferView.bufferIndex];
                    // Yes, we've already loaded every buffer into some GL buffer. However, with GL it's simpler
                    // to just copy the buffer data again for the texture. Besides, this is just an example.
                    std::visit(fastgltf::visitor {
                        // We only care about VectorWithMime here, because we specify LoadExternalBuffers, meaning
                        // all buffers are already loaded into a vector.
                        [](auto& arg) {},
                        [&](const fastgltf::sources::Array& vector) {
                            memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                        },

                    }, buffer.data);
                },
            }, buffer.data);

            resourceUpload.Upload(createdBuffer.Get(), vertexBuffer);

            resourceUpload.Transition(createdBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        }

        // Create constant buffer
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(PBREffectConstants));

        /*
        SharedGraphicsResource cBufferResource = GraphicsMemory::Get(device).AllocateConstant(sizeof(PBREffectConstants));

        DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            c_initialCopyTargetState,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(cbuffer.ReleaseAndGetAddressOf())));
        cbuffer->SetName(L"GLTF cbuffer 1");
        PBREffectConstants pbrEffectConstant = {};
        const DirectX::XMFLOAT3 translation(0.0f, 0.0f, 1.0f);
        const DirectX::XMMATRIX modelMatrix = DirectX::XMMatrixTranslationFromVector(DirectX::XMLoadFloat3(&translation));
        const DirectX::XMMATRIX viewMatrix = DirectX::XMMatrixIdentity();
        pbrEffectConstant.worldViewProj = XMMatrixMultiply(XMMatrixPerspectiveFovLH(1.121f, 1280.0/720.f, 0.01f, 1.0f), XMMatrixMultiply(viewMatrix, modelMatrix));

        memcpy(cBufferResource.Memory(), &pbrEffectConstant, sizeof(PBREffectConstants));

        resourceUpload.Upload(cbuffer.Get(), cBufferResource);

        resourceUpload.Transition(cbuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        */

        auto uploadEndFuture = resourceUpload.End(deviceResources->GetCommandQueue());
        uploadEndFuture.wait();

        for (size_t i = 0; i < kBackBufferSize; ++i) {
            for (size_t j = 0; j < kWorkerThreadSize; ++j) {
                DX::ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(commandAllocs[i][j].ReleaseAndGetAddressOf())));
                wchar_t name[25] = {};
                swprintf_s(name, L"Render target %u %u", i, j);
                commandAllocs[i][j]->SetName(name);
            }
        }
        // After CreateCommandList, the command list is in record state, so we actively close it to turn off its record state
        DX::ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocs[0][0].Get(), nullptr, IID_PPV_ARGS(commandList.ReleaseAndGetAddressOf()))); // nodeMask set to 0 for single GPU setup
        DX::ThrowIfFailed(commandList->Close());

        commandList->SetName(L"GLTF command list");
    }

    void GLTFAdapter::PrepareImage(std::shared_ptr<DX::DeviceResources> deviceResources) {
        // Decide each image's colour space from the material slots that
        // reference it, before any texture resource is created.
        m_imageColorSpaces = ClassifyImageColorSpaces(m_gltf);
        {
            std::size_t srgbCount = 0;
            for (const ImageColorSpace space : m_imageColorSpaces.perImage) {
                if (space == ImageColorSpace::Srgb) {
                    ++srgbCount;
                }
            }
            std::printf("glTF images: %zu total, %zu sRGB, %zu linear\n",
                        m_imageColorSpaces.perImage.size(), srgbCount,
                        m_imageColorSpaces.perImage.size() - srgbCount);
            for (const std::size_t imageIdx : m_imageColorSpaces.conflicts) {
                std::printf("  warning: image %zu is used as both colour and linear "
                            "data; loaded as sRGB\n", imageIdx);
            }
        }

        // Light definitions are shared; how many actually get instanced depends
        // on the nodes, which BuildSceneGraph resolves later. Printed here
        // because an asset exported without KHR_lights_punctual is otherwise
        // indistinguishable from one deliberately lit by nothing.
        std::printf("glTF lights: %zu definition(s) in the asset\n", m_gltf.lights.size());

        auto device = deviceResources->GetD3DDevice();
        ResourceUploadBatch resourceUpload(device);

        const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

        resourceUpload.Begin();
        std::size_t imageIndex = 0;
        for (const auto &image: m_gltf.images) {
            // Create staging buffer via GraphicsMemory and upload buffer data via memcpy, save allocation time using paging mechanism
            // https://github.com/microsoft/DirectXTK12/blob/f003171e0c6a30cc61444864ef756bd3cf2fc310/Src/GraphicsMemory.cpp#L126
            // TODO do we need to hold the SharedGraphicsResource even when the upload is finished?
            int width, height, nrChannels;
            std::vector<uint8_t> decodedData;
            std::visit(fastgltf::visitor {
                [](const auto& arg) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Vector& vector) {
                    // memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                },
                [&](const fastgltf::sources::CustomBuffer& customBuffer) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Fallback& fallback) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const std::monostate& monostate) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::URI& filePath) {
                    // TODO We may specify file offset
                    assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.

                    // const std::string path(filePath.uri.path().begin(), filePath.uri.path().end()); // Thanks C++.
                    // std::ifstream meshBin(path.c_str(), std::ios::binary);
                    // meshBin.read(reinterpret_cast<char*>(vertexBuffer.Memory()), filePath.fileByteOffset);
                    // meshBin.close();
                },
                [&](const fastgltf::sources::Array& vector) {
                    // TODO typically the flag LoadExternalImages will lead to this place
                    // TODO we should determine the input image format
                    unsigned char* data = stbi_load_from_memory(
                        reinterpret_cast<const stbi_uc*>(vector.bytes.data()), 
                        static_cast<int>(vector.bytes.size()), 
                        &width, &height, &nrChannels, 4
                    );
                    printf("Load image: %d, %d\n", width, height);
                    decodedData.assign(data, data + (width * height * 4));
                    if (data) {
                        // 'width' and 'height' now contain your image resolution!
                        stbi_image_free(data);
                    }
                },
                [&](const fastgltf::sources::ByteView& byteView) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },

                [&](const fastgltf::sources::BufferView& view) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
            }, image.data);

            // CD3DX12_RESOURCE_DESC::Tex2D defaults mipLevels to 0, which means
            // "allocate the full chain" -- a 1024x1024 image gets 11 mips. Only
            // subresource 0 is uploaded below, so the other 10 were left
            // uninitialised and sampled as black, which is why the model got
            // darker with distance as the sampler selected higher mips.
            //
            // Ask for the full chain only when the mips can actually be filled;
            // otherwise take a single level, so there is never an uninitialised
            // mip to sample either way.
            const ImageColorSpace colorSpace = m_imageColorSpaces.perImage[imageIndex];
            const DXGI_FORMAT imageFormat = (colorSpace == ImageColorSpace::Srgb)
                                                ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                                                : DXGI_FORMAT_R8G8B8A8_UNORM;
            const bool canGenerateMips =
                resourceUpload.IsSupportedForGenerateMips(imageFormat);
            const UINT16 mipLevels = canGenerateMips ? 0 : 1;
            const auto desc = CD3DX12_RESOURCE_DESC::Tex2D(imageFormat, width, height, 1, mipLevels);

            // TODO get image data size from image.data
            Microsoft::WRL::ComPtr<ID3D12Resource> createdTexture;
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(createdTexture.GetAddressOf())));
            createdTexture->SetName(L"GLTF image 1");
            // One page-allocated CPU descriptor per image, kept alongside it.
            // These are never bound directly -- DynamicDescriptorHeap copies
            // them into a shader-visible heap at draw time -- so the allocator
            // heap does not need to be shader visible.
            Descriptors::DescriptorAllocation srv = m_srvAllocator->Allocate(1);
            if (srv.IsNull()) {
                throw std::runtime_error("out of SRV descriptors for glTF images");
            }
            CreateShaderResourceView(device, createdTexture.Get(), srv.GetDescriptorHandle());
            m_imageDescriptors.push_back(std::move(srv));
            images.push_back(createdTexture);

            SetDebugObjectName(createdTexture.Get(), L"Image");
            D3D12_SUBRESOURCE_DATA initData = {};
            initData.pData = decodedData.data();
            // TODO need further understanding
            initData.RowPitch = static_cast<LONG>(width * 4); // the physical size of a single texture row in bytes, since hardware prefer 256-byte for optimal copy performance, the texture in GPU will be padded to align 256 bytes
            initData.SlicePitch = static_cast<LONG>(height * initData.RowPitch);
            resourceUpload.Upload(createdTexture.Get(), 0, &initData, 1);

            resourceUpload.Transition(createdTexture.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

            // Fills mips 1..N from mip 0 on the GPU. Must come after the
            // transition above: GenerateMips requires the resource to already be
            // in PIXEL_SHADER_RESOURCE, and it restores that state when done.
            //
            // sRGB cannot have a UAV created on it directly, so DirectXTK12
            // routes this through a non-sRGB alias internally; nothing extra is
            // needed here beyond the format check.
            if (canGenerateMips) {
                resourceUpload.GenerateMips(createdTexture.Get());
            }

            ++imageIndex;
        }

        // The fallback that fills unused material texture slots. Created here so
        // it rides along with the same upload batch as the glTF images.
        {
            const auto fallbackDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1);
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &fallbackDesc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(m_fallbackTexture.GetAddressOf())));
            m_fallbackTexture->SetName(L"Material fallback (1x1 white)");

            m_fallbackTextureDescriptor = m_srvAllocator->Allocate(1);
            if (m_fallbackTextureDescriptor.IsNull()) {
                throw std::runtime_error("could not allocate the fallback texture descriptor");
            }
            CreateShaderResourceView(device, m_fallbackTexture.Get(),
                                     m_fallbackTextureDescriptor.GetDescriptorHandle());

            // Opaque white, so a slot the material leaves empty multiplies
            // through as a no-op for whatever shading reads it.
            static const uint8_t kWhite[4] = {0xFF, 0xFF, 0xFF, 0xFF};
            D3D12_SUBRESOURCE_DATA initData = {};
            initData.pData = kWhite;
            initData.RowPitch = 4;
            initData.SlicePitch = 4;
            resourceUpload.Upload(m_fallbackTexture.Get(), 0, &initData, 1);
            resourceUpload.Transition(m_fallbackTexture.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        auto uploadEndFuture = resourceUpload.End(deviceResources->GetCommandQueue());
        uploadEndFuture.wait();

    }


    void GLTFAdapter::BuildSceneGraph() {
        // glTF's `scene` property picks the one to display; fall back to the
        // first if the asset does not name one.
        m_currentSceneIdx = m_gltf.defaultScene.value_or(0);
        m_sceneGraph.Build(m_gltf, m_currentSceneIdx, buffers, m_imageDescriptors.size());

        // Skins degrade rather than fail -- a missing attribute or an unreadable
        // bind matrix still renders, just wrongly -- so report what was found
        // and anything suspect about it.
        const auto &skins = m_sceneGraph.Skins();
        if (!skins.empty()) {
            std::printf("glTF skins: %zu\n", skins.size());
            for (const auto &skin : skins) {
                std::printf("  \"%s\": %zu joints, inverse bind matrices %s\n",
                            skin.name.c_str(), skin.JointCount(),
                            skin.inverseBindMatricesLoaded ? "loaded"
                                                           : "substituted (identity)");
            }
        }
        const auto &animations = m_sceneGraph.Animations();
        if (!animations.empty()) {
            std::printf("glTF animations: %zu\n", animations.size());
            for (const auto &animation : animations) {
                std::printf("  \"%s\": %.3fs, %zu channels, %zu samplers",
                            animation.name.c_str(), animation.duration,
                            animation.channels.size(), animation.samplers.size());
                if (animation.skippedChannels != 0) {
                    std::printf(" (%zu channel(s) skipped: unsupported target)",
                                animation.skippedChannels);
                }
                std::printf("\n");
            }
        }
        for (const auto &diagnostic : m_sceneGraph.SkinDiagnostics()) {
            if (diagnostic.nodeIndex.has_value()) {
                std::printf("  warning: skin %zu, node %zu: %s\n", diagnostic.skinIndex,
                            diagnostic.nodeIndex.value(), ToString(diagnostic.issue));
            } else {
                std::printf("  warning: skin %zu: %s\n", diagnostic.skinIndex,
                            ToString(diagnostic.issue));
            }
        }
    }

    void GLTFAdapter::PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources) {
        auto device = deviceResources->GetD3DDevice();
        // One slot per VertexStream, in that order; must match VSInput in
        // shaders/no_texture.fx. Each attribute lives in its own buffer slot
        // rather than being interleaved, so AlignedByteOffset stays 0.
        D3D12_INPUT_ELEMENT_DESC inputElementDesc[kVertexStreamCount] = {
            { "Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, static_cast<UINT>(VertexStream::Position), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT,    static_cast<UINT>(VertexStream::TexCoord0), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "Normal",   0, DXGI_FORMAT_R32G32B32_FLOAT, static_cast<UINT>(VertexStream::Normal),    0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            // 4 components: glTF TANGENT carries the bitangent handedness in w.
            { "Tangent",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, static_cast<UINT>(VertexStream::Tangent), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            // Formats come from the asset, because the vertex buffers are its
            // own bytes bound directly: glTF allows JOINTS_0 as unsigned byte or
            // short and WEIGHTS_0 as float or normalised integer, and this
            // single PSO has to declare whichever one this asset used.
            { "Joints",   0, m_sceneGraph.JointIndexFormat(),  static_cast<UINT>(VertexStream::Joints0),  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "Weights",  0, m_sceneGraph.JointWeightFormat(), static_cast<UINT>(VertexStream::Weights0), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }};

        D3D12_INPUT_LAYOUT_DESC inputLayoutDesc = {};
        inputLayoutDesc.NumElements = static_cast<UINT>(kVertexStreamCount);
        inputLayoutDesc.pInputElementDescs = inputElementDesc;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};

        // Create root signature.
        {
            D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags
                = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

            // Same as CommonStates::StaticLinearClamp
            const CD3DX12_STATIC_SAMPLER_DESC sampler(0, // register
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                0.f,
                16,
                D3D12_COMPARISON_FUNC_LESS_EQUAL,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                0.f,
                D3D12_FLOAT32_MAX,
                D3D12_SHADER_VISIBILITY_PIXEL);

            // Root signature 1.1 rather than 1.0, because
            // Descriptors::RootSignature takes a D3D12_ROOT_SIGNATURE_DESC1 --
            // DynamicDescriptorHeap needs the parsed layout to know which root
            // parameters are descriptor tables and how wide each one is.
            //
            // DESCRIPTORS_VOLATILE keeps 1.0 semantics: the descriptors a table
            // points at may change between Set and Execute, which is exactly
            // what DynamicDescriptorHeap does when it copies into a fresh heap.
            // Must stay in lockstep with NoTextureRootSignature in
            // shaders/no_texture.fx -- the PSO is created with this signature
            // while the shader carries its own copy.
            //
            //   b0  per-node transform constants
            //   t0..t4  material textures (base colour, metallic-roughness,
            //           normal, occlusion, emissive)
            //   s0  sampler
            //   b1  per-primitive material constants
            //   b2  per-node joint matrices (skinning)
            CD3DX12_ROOT_PARAMETER1 rootParameters[5] = {};

            rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            const CD3DX12_DESCRIPTOR_RANGE1 materialTextureRange(
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                static_cast<UINT>(kMaterialTextureSlotCount), 0, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            const CD3DX12_DESCRIPTOR_RANGE1 texture1SamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            rootParameters[1].InitAsDescriptorTable(1, &materialTextureRange, D3D12_SHADER_VISIBILITY_PIXEL);
            rootParameters[2].InitAsDescriptorTable(1, &texture1SamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);
            // Visibility ALL, matching the HLSL "CBV(b1)" which defaults to ALL;
            // a narrower visibility here would not match the shader's copy.
            rootParameters[3].InitAsConstantBufferView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            // Vertex-only: skinning happens before rasterisation, so the pixel
            // shader never needs the joint matrices. Narrowing it here is safe
            // because the shader's own copy says VERTEX too.
            rootParameters[4].InitAsConstantBufferView(2, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_VERTEX);

            D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
            rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
            rsigDesc.pParameters = rootParameters;
            rsigDesc.NumStaticSamplers = 0;
            rsigDesc.pStaticSamplers = nullptr;
            rsigDesc.Flags = rootSignatureFlags;

            // Serializes and creates the ID3D12RootSignature, and records the
            // descriptor-table bit masks and per-table descriptor counts.
            m_rootSignature.SetRootSignatureDesc(rsigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);
            psoDesc.pRootSignature = m_rootSignature.GetRootSignature().Get();

            // NOTE: the dynamic heap is NOT taught the layout here.
            // DynamicDescriptorHeap::Reset() clears the descriptor-table bit mask
            // and nulls every cached BaseDescriptor, so ParseRootSignature has to
            // be re-called every frame after the reset -- see Render(), which
            // does it alongside SetGraphicsRootSignature exactly as upstream's
            // CommandList does.
        }


        const D3D12_SHADER_BYTECODE vertexShader = { g_vertex_shader_bytecode, sizeof(g_vertex_shader_bytecode) };
        psoDesc.VS = vertexShader;
        const D3D12_SHADER_BYTECODE pixelShader = { g_pixel_shader_bytecode, sizeof(g_pixel_shader_bytecode) };
        psoDesc.PS = pixelShader;
        psoDesc.InputLayout = inputLayoutDesc;
        psoDesc.BlendState = CommonStates::Opaque;
        psoDesc.DepthStencilState = CommonStates::DepthDefault;
        psoDesc.RasterizerState = CommonStates::CullClockwise;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;

        psoDesc.SampleMask = UINT_MAX;
        psoDesc.NumRenderTargets = 1;
        DXGI_FORMAT         rtvFormats[1];
        rtvFormats[0] = deviceResources->GetBackBufferFormat();
        DXGI_FORMAT         dsvFormat = deviceResources->GetDepthBufferFormat();
        memcpy(psoDesc.RTVFormats, rtvFormats, sizeof(DXGI_FORMAT) * 1);
        psoDesc.DSVFormat = dsvFormat;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.NodeMask = 0;


        DX::ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_pso)));
    }

    void GLTFAdapter::Render(std::shared_ptr<DX::DeviceResources> deviceResources, const DirectX::SimpleMath::Matrix &worldMatrix, const DirectX::SimpleMath::Matrix &viewMatrix, const DirectX::SimpleMath::Matrix &projectionMatrix) {
        auto device = deviceResources->GetD3DDevice();

        DX::ThrowIfFailed(commandAllocs[deviceResources->GetCurrentFrameIndex()][0]->Reset()); // In reality, deviceResource->GetCurrentFrameIndex() will return current back buffer index
        DX::ThrowIfFailed(commandList->Reset(commandAllocs[deviceResources->GetCurrentFrameIndex()][0].Get(), nullptr));
        PIXBeginEvent(commandList.Get(), PIX_COLOR_DEFAULT, L"Begin GLTF render");
        // https://www.reddit.com/r/GraphicsProgramming/comments/1bnb08z/dx12_confusion_about_root_signatures/
        // https://stackoverflow.com/questions/38535725/what-is-the-point-of-d3d12s-setgraphicsrootsignature
        // The "root signature" in DirectX 12 provides the common layout information for sharing data between the CPU data structures and the GPU shader language execution.
        // NOTE: Although PSO state explicitly declare root signature and primitive topology, we still need to assign it by ourselves because each pso doesn't share this information
        // The setting of pRootSignature through Set*RootSignature() is like holding CPU and GPU function signature along with their constant buffer, texture binding so that when swithcing PSO or partial resources, the other binding doesn't change
        // While the creation of PSO with pRootSignature is like providing hardware and driver info to optimize shader compilation
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview
        float blendFactor[4] = {0.0, 0.0, 0.0, 0.0};
        commandList->OMSetBlendFactor(blendFactor); // Set null means (1, 1, 1, 1)
        
        commandList->OMSetRenderTargets(1, &deviceResources->GetRenderTargetView(), false, &deviceResources->GetDepthStencilView());
        // commandList->OMSetStencilRef(0);

        // Application.hpp's Prepare already clear depth and stencil, so we don't need to do it again
        // D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(deviceResources->GetRenderTarget(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        // commandList->ResourceBarrier(1, &barrier);

        // commandList->ClearDepthStencilView(deviceResources->GetDepthStencilView(), D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, 0, nullptr);
        // float blackColor[4] = {0.0, 0.0, 0.0, 0.0};
        // commandList->ClearRenderTargetView(deviceResources->GetRenderTargetView(), blackColor, 0, nullptr);

        auto viewport = deviceResources->GetScreenViewport();
        auto scissorRect = deviceResources->GetScissorRect();
        commandList->RSSetViewports(1, &viewport);
        commandList->RSSetScissorRects(1, &scissorRect);

        // Descriptor heap bindings do not survive a command list reset, and
        // Application's DirectXTK12 draws rebind their own heaps earlier in the
        // frame, so re-establish ours here. The sampler heap is registered once;
        // the dynamic SRV heap registers itself as it commits, and the binder
        // reissues SetDescriptorHeaps with both whenever either changes.
        m_heapBinder.Reset(commandList.Get());
        m_heapBinder.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, m_commonStates->Heap());

        // The dynamic heap has to be reset in step with the command list. It
        // otherwise keeps last frame's GPU-visible heap in m_CurrentDescriptorHeap
        // and, finding it still has free handles, never re-issues
        // SetDescriptorHeaps -- but the binding died with the command list reset,
        // so the first SetGraphicsRootDescriptorTable fails validation with
        // SET_DESCRIPTOR_TABLE_INVALID.
        //
        // Safe here for the same reason resetting the command allocator above is:
        // Present() has already waited for the previous use of this back buffer
        // index, so the GPU-visible heaps being recycled are no longer in flight.
        m_srvDynamicHeap->Reset();

        commandList->SetGraphicsRootSignature(m_rootSignature.GetRootSignature().Get());
        // Must follow Reset(), which wipes the parsed layout.
        m_srvDynamicHeap->ParseRootSignature(m_rootSignature);
        commandList->SetPipelineState(m_pso.Get());
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // The scene placement, unchanged: this is now the root transform that
        // every node's own transform is composed onto, rather than the single
        // matrix every mesh was drawn with.
        // Via ScenePlacement so retargeting, which has to aim through this
        // same transform, cannot drift from what is actually drawn.
        const DirectX::SimpleMath::Matrix placement = ScenePlacement();

        // Top-down pass: each node's world transform is its local transform
        // composed with its parent's, and hidden subtrees drop out of the draw
        // order here rather than being tested per draw.
        m_sceneGraph.UpdateTransforms(placement);

        // Lights are a property of the scene, not of a node, so they are packed
        // once and copied into every node's constant buffer. UpdateTransforms
        // above has just placed them, so this has to follow it.
        ShaderLight frameLights[kMaxShaderLights] = {};
        const std::size_t lightCount =
            m_sceneGraph.GatherShaderLights(frameLights, kMaxShaderLights);

        // b2 has to be bound for every draw, skinned or not: the vertex shader
        // declares the cbuffer unconditionally, and reading an unbound root CBV
        // is undefined. Unskinned nodes share this one all-identity allocation
        // rather than each paying for 8 KB of identities.
        SharedGraphicsResource identityJoints =
            GraphicsMemory::Get(device).AllocateConstant<JointMatrixConstants>();
        {
            auto *joints = static_cast<JointMatrixConstants *>(identityJoints.Memory());
            for (std::size_t j = 0; j < kMaxJointMatrices; ++j) {
                joints->joints[j] = XMMatrixIdentity();
            }
        }
        std::vector<DirectX::SimpleMath::Matrix> jointMatrices;
        // One constant buffer per node, because each now carries its own world
        // matrix.
        //
        // NOTE: nothing in this project calls GraphicsMemory::Commit(), which is
        // what retires allocator pages against a fence. Measured over 45 s the
        // working set is stable, so this is not a leak -- pages are being
        // recycled -- but recycling without the fence is what Commit() exists to
        // prevent, so a page could in principle be reused while the GPU is still
        // reading it. Pre-existing and never observed to misbehave; flagged
        // rather than changed.
        for (const std::size_t nodeIndex : m_sceneGraph.DrawOrder()) {
            const SceneNode &node = m_sceneGraph.Nodes()[nodeIndex];

            PIXBeginEvent(commandList.Get(), PIX_COLOR_DEFAULT, L"Begin GLTF node");

            PBREffectConstants pbrEffectConstant = {};
            pbrEffectConstant.eyePosition = m_camera->mEye;

            // World and world-inverse-transpose, in the byte order HLSL's
            // column-major cbuffer read expects. Both were wrong before:
            // `world` went in untransposed, and worldInverseTranspose was never
            // written at all, so VSOutput::WorldPosition was garbage and
            // WorldNormal was zero. Nothing consumed either -- the pixel shader
            // returned base colour -- but the lights below are in world space,
            // so the shading about to be written needs them both.
            PackWorldMatrices(node.worldTransform, pbrEffectConstant.world,
                              pbrEffectConstant.worldInverseTranspose);

            pbrEffectConstant.worldViewProj = XMMatrixTranspose(
                XMMatrixMultiply(XMMatrixMultiply(node.worldTransform, viewMatrix), projectionMatrix));

            std::memcpy(pbrEffectConstant.lights, frameLights, sizeof(frameLights));
            pbrEffectConstant.lightCount = static_cast<int>(lightCount);
            // NOTE: Set*Root* only set BufferLocation instead of SizeInBytes, which may cause GPU to crash if shader access data out of bound
            // Therefore, it's suitable for frequently changing resource.
            // https://gamedev.net/forums/topic/678623-d3d12-using-setgraphicsrootview-functions/#post-5291945
            SharedGraphicsResource cBufferResource =
                GraphicsMemory::Get(device).AllocateConstant(pbrEffectConstant);
            commandList->SetGraphicsRootConstantBufferView(0, cBufferResource.GpuAddress());

            // Joint matrices are per (skin, node): the formula divides out this
            // node's world transform, so two nodes sharing a skin do not share
            // matrices. Computed here rather than in UpdateTransforms so an
            // unskinned scene pays nothing.
            D3D12_GPU_VIRTUAL_ADDRESS jointsAddress = identityJoints.GpuAddress();
            SharedGraphicsResource nodeJoints;
            if (node.IsSkinned() && m_sceneGraph.ComputeJointMatrices(nodeIndex, jointMatrices)) {
                nodeJoints =
                    GraphicsMemory::Get(device).AllocateConstant<JointMatrixConstants>();
                auto *joints = static_cast<JointMatrixConstants *>(nodeJoints.Memory());
                const std::size_t count = std::min(jointMatrices.size(), kMaxJointMatrices);
                for (std::size_t j = 0; j < count; ++j) {
                    // Transposed for the same reason PBR_World is: HLSL reads
                    // cbuffer matrices column-major.
                    joints->joints[j] = XMMatrixTranspose(jointMatrices[j]);
                }
                // A skin larger than the array is reported by SceneGraph; the
                // remainder stays identity so an out-of-range index is harmless.
                for (std::size_t j = count; j < kMaxJointMatrices; ++j) {
                    joints->joints[j] = XMMatrixIdentity();
                }
                jointsAddress = nodeJoints.GpuAddress();
            }
            commandList->SetGraphicsRootConstantBufferView(4, jointsAddress);

            for (const PrimitiveResource &primitive : node.primitives) {
                // Per-primitive material factors at b1.
                SharedGraphicsResource materialResource =
                    GraphicsMemory::Get(device).AllocateConstant(primitive.material);
                commandList->SetGraphicsRootConstantBufferView(3, materialResource.GpuAddress());

                // Every slot of the five-wide SRV table gets a descriptor:
                // the material's texture where it has one, the white fallback
                // otherwise. Leaving a slot unwritten would put an
                // uninitialised descriptor in a bound table.
                //
                // Staged one slot at a time because StageDescriptors copies a
                // *contiguous* source range, and these descriptors come from
                // separate allocations.
                for (std::size_t slot = 0; slot < kMaterialTextureSlotCount; ++slot) {
                    const auto imageIndex = primitive.textureImageIndex[slot];
                    const D3D12_CPU_DESCRIPTOR_HANDLE handle =
                        imageIndex.has_value()
                            ? m_imageDescriptors[imageIndex.value()].GetDescriptorHandle()
                            : m_fallbackTextureDescriptor.GetDescriptorHandle();
                    m_srvDynamicHeap->StageDescriptors(1, static_cast<uint32_t>(slot), 1, handle);
                }

                // The sampler table points straight into the DirectXTK12
                // sampler heap, which the binder keeps bound.
                commandList->SetGraphicsRootDescriptorTable(2, m_commonStates->AnisotropicWrap());

                commandList->IASetVertexBuffers(0, static_cast<UINT>(kVertexStreamCount),
                                                primitive.vertexBufferViews);
                commandList->IASetIndexBuffer(&primitive.indexBufferView);

                m_srvDynamicHeap->CommitStagedDescriptorsForDraw(commandList.Get(), m_heapBinder);
                commandList->DrawIndexedInstanced(primitive.indexCount, 1, 0, 0, 0);
            }

            PIXEndEvent(commandList.Get());
        }
        PIXEndEvent(commandList.Get());
        DX::ThrowIfFailed(commandList->Close());
        deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(commandList.GetAddressOf()));

    }

    void GLTFAdapter::UpdateAnimation(float deltaSeconds) {
        const auto &animations = m_sceneGraph.Animations();
        if (animations.empty()) {
            return;
        }

        // Pick the first clip once, so an animated asset moves without the UI
        // being touched. After that an empty selection means the user asked for
        // the authored pose, and must be left alone.
        if (!m_animationAutoSelected) {
            m_animationAutoSelected = true;
            m_activeAnimation = 0;
            m_animationTime = 0.0f;
        }
        if (!m_activeAnimation.has_value()) {
            return;
        }
        if (m_activeAnimation.value() >= animations.size()) {
            m_activeAnimation = 0;
        }

        const SceneAnimation &animation = animations[m_activeAnimation.value()];

        if (m_animationPlaying && animation.duration > 0.0f) {
            m_animationTime += deltaSeconds * m_animationSpeed;
            if (m_animationLoop) {
                // fmod keeps a negative speed working too, where plain
                // subtraction would walk off the start of the timeline.
                m_animationTime = std::fmod(m_animationTime, animation.duration);
                if (m_animationTime < 0.0f) {
                    m_animationTime += animation.duration;
                }
            } else {
                m_animationTime = std::min(m_animationTime, animation.duration);
            }
        }

        // Poses the nodes. Render's UpdateTransforms then composes them and
        // ComputeJointMatrices turns the joints into the b2 matrices.
        m_sceneGraph.ApplyAnimation(m_activeAnimation.value(), m_animationTime);
    }

    void GLTFAdapter::ShowImgui() {
        ImGui::SliderFloat3("GLTF position", m_position, -10.0, 10.0);
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto &animations = m_sceneGraph.Animations();
            if (animations.empty()) {
                ImGui::TextUnformatted("This asset has no animations.");
            } else {
                // "none" restores the authored pose, which is the only way to
                // see the bind pose once playback has started.
                const char *current = m_activeAnimation.has_value()
                                          ? animations[m_activeAnimation.value()].name.c_str()
                                          : "(none -- authored pose)";
                if (ImGui::BeginCombo("Clip", current)) {
                    if (ImGui::Selectable("(none -- authored pose)",
                                          !m_activeAnimation.has_value())) {
                        m_activeAnimation.reset();
                        m_sceneGraph.ResetToBasePose();
                    }
                    for (std::size_t i = 0; i < animations.size(); ++i) {
                        const bool selected = m_activeAnimation.has_value() &&
                                              m_activeAnimation.value() == i;
                        if (ImGui::Selectable(animations[i].name.c_str(), selected)) {
                            m_activeAnimation = i;
                            m_animationTime = 0.0f;
                        }
                    }
                    ImGui::EndCombo();
                }

                if (m_activeAnimation.has_value()) {
                    const SceneAnimation &animation = animations[m_activeAnimation.value()];
                    ImGui::Checkbox("Play", &m_animationPlaying);
                    ImGui::SameLine();
                    ImGui::Checkbox("Loop", &m_animationLoop);
                    ImGui::SliderFloat("Speed", &m_animationSpeed, -2.0f, 2.0f);
                    // Scrubbing while paused is the useful case; dragging while
                    // playing just gets overwritten on the next frame.
                    if (ImGui::SliderFloat("Time", &m_animationTime, 0.0f,
                                           animation.duration > 0.0f ? animation.duration
                                                                     : 1.0f)) {
                        m_sceneGraph.ApplyAnimation(m_activeAnimation.value(),
                                                    m_animationTime);
                    }
                    ImGui::Text("%.3f / %.3f s, %zu channels", m_animationTime,
                                animation.duration, animation.channels.size());
                    if (animation.skippedChannels != 0) {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                           "%zu channel(s) skipped (morph weights "
                                           "are not supported)",
                                           animation.skippedChannels);
                    }
                }
            }
        }
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Textures")) {
            const auto &spaces = m_imageColorSpaces.perImage;
            std::size_t srgbCount = 0;
            for (const ImageColorSpace space : spaces) {
                if (space == ImageColorSpace::Srgb) {
                    ++srgbCount;
                }
            }
            ImGui::Text("%zu images: %zu sRGB, %zu linear", spaces.size(), srgbCount,
                        spaces.size() - srgbCount);
            if (!m_imageColorSpaces.conflicts.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                   "%zu image(s) used as both colour and linear data",
                                   m_imageColorSpaces.conflicts.size());
            }
            for (std::size_t i = 0; i < spaces.size(); ++i) {
                ImGui::Text("  image %zu: %s", i, ToString(spaces[i]));
            }
        }
        if (ImGui::CollapsingHeader("Scene hierarchy", ImGuiTreeNodeFlags_DefaultOpen)) {
            m_sceneGraph.DrawHierarchyUI();
        }
    }

    void GLTFAdapter::ReleaseStaleDescriptors(uint64_t frameNumber) {
        // Returns descriptors freed on or before frameNumber to their pages.
        // The caller passes a frame number it knows the GPU is done with, so
        // nothing still referenced by an in-flight command list is recycled.
        if (m_srvAllocator) {
            m_srvAllocator->ReleaseStaleDescriptors(frameNumber);
        }
        // The dynamic heap's GPU-visible heaps are recycled by its own Reset()
        // at the top of Render(), which is tied to the command list rather than
        // to this frame-retirement margin.
    }

}