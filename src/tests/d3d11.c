#include "gpu_tests.h"
#include "d3d11/gpu.h"
#include <dxgi1_2.h>

int main()
{
    pl_log log = pl_test_logger();
    IDXGIFactory1 *factory = NULL;
    IDXGIAdapter1 *adapter1 = NULL;
    HRESULT hr;

    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    if (!dxgi)
        return SKIP;

    PFN_CREATE_DXGI_FACTORY pCreateDXGIFactory1 =
        (void *) GetProcAddress(dxgi, "CreateDXGIFactory1");
    if (!pCreateDXGIFactory1)
        return SKIP;

    pCreateDXGIFactory1(&IID_IDXGIFactory1, (void **) &factory);

    // Test all attached devices
    for (int i = 0;; i++) {
        hr = IDXGIFactory1_EnumAdapters1(factory, i, &adapter1);
        if (hr == DXGI_ERROR_NOT_FOUND)
            break;
        if (FAILED(hr)) {
            printf("Failed to enumerate adapters\n");
            return SKIP;
        }

        DXGI_ADAPTER_DESC1 desc;
        hr = IDXGIAdapter1_GetDesc1(adapter1, &desc);
        if (FAILED(hr)) {
            printf("Failed to enumerate adapters\n");
            return SKIP;
        }
        SAFE_RELEASE(adapter1);

        const struct pl_d3d11 *d3d11;
        struct pl_d3d11_params params = pl_d3d11_default_params;
        params.debug = true;
        params.adapter_luid = desc.AdapterLuid;
        d3d11 = pl_d3d11_create(log, &params);
        REQUIRE(d3d11);

        gpu_shader_tests(d3d11->gpu);

        pl_d3d11_destroy(&d3d11);
    }

    SAFE_RELEASE(factory);
}
