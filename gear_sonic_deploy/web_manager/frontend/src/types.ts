export interface SimConfigResponse {
  ok: boolean;
  simulator: string;
  model_root: string;
  scene_path: string;
  scene_url: string;
  manifest_url: string;
  wbc_config_path: string;
  default_timestep: number;
  notes: string[];
}

export interface SimAssetFile {
  path: string;
  url: string;
  size_bytes: number;
}

export interface XmlAssetReferences {
  includes: string[];
  meshes: string[];
  textures: string[];
  meshdir: string | null;
}

export interface IncludeAssetReference {
  path: string;
  url: string;
  references: XmlAssetReferences;
}

export interface SimAssetManifest {
  model_root: string;
  scene_path: string;
  scene_url: string;
  wbc_config_path: string;
  scene_references: XmlAssetReferences;
  include_references: IncludeAssetReference[];
  files: SimAssetFile[];
}

export interface SimManifestResponse {
  ok: boolean;
  manifest: SimAssetManifest;
}

export interface RuntimeStatus {
  phase: "idle" | "loading" | "ready" | "running" | "paused" | "error";
  message: string;
  moduleExportNames: string[];
  assetFileCount: number;
  modelLoaded: boolean;
  bodyCount: number;
  qposCount: number;
  qvelCount: number;
  actuatorCount: number;
  sceneBytes: number;
  simulationTime: number;
  stepCount: number;
  stepRate: number;
  lastError: string | null;
}
