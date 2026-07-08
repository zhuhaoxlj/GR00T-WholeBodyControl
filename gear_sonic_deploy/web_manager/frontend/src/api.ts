import type { SimConfigResponse, SimManifestResponse, WbcManifestResponse } from "./types";

async function requestJson<ResponseType>(url: string): Promise<ResponseType> {
  const response = await fetch(url);
  const responseBody = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(JSON.stringify(responseBody.detail ?? responseBody));
  }
  return responseBody as ResponseType;
}

export async function loadSimConfig(): Promise<SimConfigResponse> {
  return requestJson<SimConfigResponse>("/api/sim/config");
}

export async function loadSimAssetManifest(): Promise<SimManifestResponse> {
  return requestJson<SimManifestResponse>("/api/sim/assets/manifest");
}

export async function loadWbcManifest(): Promise<WbcManifestResponse> {
  return requestJson<WbcManifestResponse>("/api/sim/wbc/manifest");
}

export async function loadTextAsset(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load asset ${url}: HTTP ${response.status}`);
  }
  return response.text();
}

export async function loadBinaryAsset(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load asset ${url}: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
