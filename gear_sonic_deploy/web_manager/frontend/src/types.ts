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

export interface WbcPolicyAsset {
  path: string;
  url: string;
  size_bytes: number;
}

export interface WbcPolicyConfig {
  config_path: string;
  resource_root: string;
  simulation_dt: number;
  control_decimation: number;
  num_actions: number;
  num_obs: number;
  obs_history_len: number;
  kps: number[];
  kds: number[];
  default_angles: number[];
  action_scale: number;
  ang_vel_scale: number;
  dof_pos_scale: number;
  dof_vel_scale: number;
  cmd_scale: number[];
  cmd_init: number[];
  height_cmd: number;
  rpy_cmd: number[];
  freq_cmd: number;
  joint_names: string[];
}

export interface WbcPolicyManifest {
  config: WbcPolicyConfig;
  policies: {
    balance: WbcPolicyAsset;
    walk: WbcPolicyAsset;
  };
}

export interface WbcManifestResponse {
  ok: boolean;
  manifest: WbcPolicyManifest;
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
  wbcLoaded: boolean;
  policyMode: "off" | "balance" | "walk";
  command: [number, number, number];
  policyInferenceMs: number;
  lastError: string | null;
}
