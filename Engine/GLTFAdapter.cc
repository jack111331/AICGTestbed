#include "pch.h"
#include "GLTFAdapter.hpp"
#include <string>
#include <sstream>
#include <memory>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <directx/d3dx12.h>
#include <stb_image.h>
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
        fastgltf::Extensions extensions = fastgltf::Extensions::MYLAB_generative;
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
        auto device = deviceResources->GetD3DDevice();
        ResourceUploadBatch resourceUpload(device);

        const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

        resourceUpload.Begin();
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

            const auto desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, height);

            // TODO get image data size from image.data
            Microsoft::WRL::ComPtr<ID3D12Resource> createdTexture;
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(createdTexture.GetAddressOf())));
            createdTexture->SetName(L"GLTF image 1");
            CreateShaderResourceView(device, createdTexture.Get(), m_descriptorHeap->GetCpuHandle(descriptorHeapStartIdx++));
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
        }

        auto uploadEndFuture = resourceUpload.End(deviceResources->GetCommandQueue());
        uploadEndFuture.wait();

    }


    void GLTFAdapter::PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources) {
        auto device = deviceResources->GetD3DDevice();
        // TODO TexCoord0's AlignedByteOffset not determined
        D3D12_INPUT_ELEMENT_DESC inputElementDesc[2] = {{ "Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
                                                        { "TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT, 1, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }};

        D3D12_INPUT_LAYOUT_DESC inputLayoutDesc = {};
        inputLayoutDesc.NumElements = 2;
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

            CD3DX12_ROOT_PARAMETER rootParameters[3] = {};

            // Root parameter descriptor
            CD3DX12_ROOT_SIGNATURE_DESC rsigDesc = {};

            // Constant buffer
            rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_SHADER_VISIBILITY_ALL);
            // TODO Need to understand
            const CD3DX12_DESCRIPTOR_RANGE texture1Range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
            const CD3DX12_DESCRIPTOR_RANGE texture1SamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0);
            rootParameters[1].InitAsDescriptorTable(1, &texture1Range, D3D12_SHADER_VISIBILITY_PIXEL);
            rootParameters[2].InitAsDescriptorTable(1, &texture1SamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);

            rsigDesc.Init(static_cast<UINT>(std::size(rootParameters)), rootParameters, 0, nullptr, rootSignatureFlags);
            Microsoft::WRL::ComPtr<ID3DBlob> pSignature;
            Microsoft::WRL::ComPtr<ID3DBlob> pError;
            HRESULT                          hr = D3D12SerializeRootSignature(&rsigDesc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                pSignature.GetAddressOf(),
                pError.GetAddressOf());
            device->CreateRootSignature(0, pSignature->GetBufferPointer(), pSignature->GetBufferSize(), IID_PPV_ARGS(&pRootSignature));
            psoDesc.pRootSignature = pRootSignature;
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
        float blendFactor[4] = {1.0, 1.0, 1.0, 1.0};
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

        // TODO I don't sure why I need this, I thought I've already set it? Or m_modelResources in Application.cc's own heap just override it?
        ID3D12DescriptorHeap* heaps[] = { m_descriptorHeap->Heap(), m_commonStates->Heap() };
        commandList->SetDescriptorHeaps(_countof(heaps), heaps);

        commandList->SetGraphicsRootSignature(pRootSignature);
        commandList->SetPipelineState(m_pso.Get());
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        PBREffectConstants pbrEffectConstant = {};
        const XMVECTORF32 scale = { 1.0f, 1.0f, 1.0f };
        const XMVECTORF32 translate = { m_position[0], m_position[1], m_position[2] };
        XMVECTOR rotate = DirectX::SimpleMath::Quaternion::CreateFromYawPitchRoll(XM_PI / 2.f, 0.f, -XM_PI / 2.f);
        XMMATRIX local = worldMatrix * XMMatrixTransformation(g_XMZero, DirectX::SimpleMath::Quaternion::Identity, scale, g_XMZero, rotate, translate);
        pbrEffectConstant.eyePosition = m_camera->mEye;
        pbrEffectConstant.world = local;
        pbrEffectConstant.worldViewProj = XMMatrixTranspose(XMMatrixMultiply(XMMatrixMultiply(local, viewMatrix), projectionMatrix));
        SharedGraphicsResource cBufferResource = GraphicsMemory::Get(device).AllocateConstant(pbrEffectConstant);
        // NOTE: Set*Root* only set BufferLocation instead of SizeInBytes, which may cause GPU to crash if shader access data out of bound
        // Therefore, it's suitable for frequently changing resource 
        // https://gamedev.net/forums/topic/678623-d3d12-using-setgraphicsrootview-functions/#post-5291945
        commandList->SetGraphicsRootConstantBufferView(0, cBufferResource.GpuAddress());
        // TODO should group render object
        for (const auto &node: m_gltf.nodes) {
            // TODO render hierarchically
            // node.children
            PIXBeginEvent(commandList.Get(), PIX_COLOR_DEFAULT, L"Begin GLTF mesh");
            if (!node.meshIndex.has_value()) {
                PIXEndEvent(commandList.Get());
                continue;
            }
            const auto &meshIndex = node.meshIndex.value();
            const auto &mesh = m_gltf.meshes[meshIndex];
            for (const auto &primitive: mesh.primitives) {
                const auto &primitiveType = primitive.type;
                const auto &indicesAccessorIdx = primitive.indicesAccessor.value();
                const auto &indicesAccessor = m_gltf.accessors[indicesAccessorIdx];
                const auto &indicesBufferView = m_gltf.bufferViews[indicesAccessor.bufferViewIndex.value()];
                const auto &indicesBufferLocation = buffers[indicesBufferView.bufferIndex]->GetGPUVirtualAddress() + indicesBufferView.byteOffset;
                const auto &positionAttribute = primitive.attributes[0]; // TODO assume the position accessor is always at the first attribute of primitive, need further refinement
                D3D12_VERTEX_BUFFER_VIEW vbView[2] = {};
                if (primitive.attributes.size() > 2) {
                    // TODO NORMAL
                    auto texcoordIt = primitive.findAttribute("TEXCOORD_0");
                    if (texcoordIt != primitive.attributes.end()) {
                        const auto &texcoordAccessor = m_gltf.accessors[texcoordIt->accessorIndex];
                        const auto &texcoordBufferView = m_gltf.bufferViews[texcoordAccessor.bufferViewIndex.value()];
                        const auto &texcoordBufferLocation = buffers[texcoordBufferView.bufferIndex]->GetGPUVirtualAddress() + texcoordBufferView.byteOffset;                
                        vbView[1].BufferLocation = texcoordBufferLocation + texcoordAccessor.byteOffset; // GPU virtual address
                        vbView[1].SizeInBytes = sizeof(float) * texcoordAccessor.count * 2;             // TODO should reference accessor.componentType and accessor.type actually, but we use this default
                        vbView[1].StrideInBytes = sizeof(float) * 2;                         // Size of one element
                    }
                }
                const auto &positionAccessor = m_gltf.accessors[positionAttribute.accessorIndex];
                const auto &positionBufferView = m_gltf.bufferViews[positionAccessor.bufferViewIndex.value()];
                const auto &positionBufferLocation = buffers[positionBufferView.bufferIndex]->GetGPUVirtualAddress() + positionBufferView.byteOffset;                

                // TODO use image
                if (primitive.materialIndex.has_value()) {
                    const auto &materialIdx = primitive.materialIndex.value();
                    if (m_gltf.materials[materialIdx].pbrData.baseColorTexture.has_value()) {
                        const auto &baseColorTexture = m_gltf.materials[materialIdx].pbrData.baseColorTexture.value();
                        commandList->SetGraphicsRootDescriptorTable(1, m_descriptorHeap->GetGpuHandle(3)); // TODO
                        commandList->SetGraphicsRootDescriptorTable(2, m_commonStates->AnisotropicWrap());
                        // commandList->SetGraphicsRootShaderResourceView(0, images[baseColorTexture.textureIndex]->GetGPUVirtualAddress());
                    }
                }

                vbView[0].BufferLocation = positionBufferLocation + positionAccessor.byteOffset; // GPU virtual address
                vbView[0].SizeInBytes = sizeof(float) * positionAccessor.count * 3;             // TODO should reference accessor.componentType and accessor.type actually, but we use this default
                vbView[0].StrideInBytes = sizeof(float) * 3;                         // Size of one element
                commandList->IASetVertexBuffers(0, 2, vbView);

                D3D12_INDEX_BUFFER_VIEW idxView = {};
                idxView.BufferLocation = indicesBufferLocation + indicesAccessor.byteOffset;
                idxView.Format = indicesAccessor.componentType == fastgltf::ComponentType::UnsignedInt ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
                idxView.SizeInBytes = indicesAccessor.count * (indicesAccessor.componentType == fastgltf::ComponentType::UnsignedInt?sizeof(uint32_t):sizeof(uint16_t)); // TODO not sure indicesAccessor.count
                commandList->IASetIndexBuffer(&idxView);

                // TODO bind texture and shader
                commandList->DrawIndexedInstanced(indicesAccessor.count, 1, 0, 0, 0);
            }
            PIXEndEvent(commandList.Get());
        }
        PIXEndEvent(commandList.Get());
        DX::ThrowIfFailed(commandList->Close());
        deviceResources->GetCommandQueue()->ExecuteCommandLists(1, CommandListCast(commandList.GetAddressOf()));

    }

    void GLTFAdapter::ShowImgui() {
        ImGui::SliderFloat3("GLTF position", m_position, -10.0, 10.0);
    }

}