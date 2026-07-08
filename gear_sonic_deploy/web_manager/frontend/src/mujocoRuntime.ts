import loadMujoco, {
  type MainModule,
  type MjData,
  type MjModel,
  type MjvCamera,
  type MjvGeom,
  type MjvOption,
  type MjvPerturb,
  type MjvScene
} from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import * as THREE from "three";
import type { RuntimeStatus, SimConfigResponse, SimManifestResponse } from "./types";

interface SimAssetContent {
  path: string;
  data: Uint8Array;
}

interface RuntimeSnapshot {
  status: RuntimeStatus;
}

interface ThreeVisual {
  mesh: THREE.Mesh;
  signature: string;
}

const DEFAULT_JOINT_POSE: Array<[string, number]> = [
  ["left_hip_pitch_joint", -0.1],
  ["left_hip_roll_joint", 0.0],
  ["left_hip_yaw_joint", 0.0],
  ["left_knee_joint", 0.3],
  ["left_ankle_pitch_joint", -0.2],
  ["left_ankle_roll_joint", 0.0],
  ["right_hip_pitch_joint", -0.1],
  ["right_hip_roll_joint", 0.0],
  ["right_hip_yaw_joint", 0.0],
  ["right_knee_joint", 0.3],
  ["right_ankle_pitch_joint", -0.2],
  ["right_ankle_roll_joint", 0.0],
  ["waist_yaw_joint", 0.0],
  ["waist_roll_joint", 0.0],
  ["waist_pitch_joint", 0.0],
  ["left_shoulder_pitch_joint", 0.0],
  ["left_shoulder_roll_joint", 0.0],
  ["left_shoulder_yaw_joint", 0.0],
  ["left_elbow_joint", 0.0],
  ["left_wrist_roll_joint", 0.0],
  ["left_wrist_pitch_joint", 0.0],
  ["left_wrist_yaw_joint", 0.0],
  ["right_shoulder_pitch_joint", 0.0],
  ["right_shoulder_roll_joint", 0.0],
  ["right_shoulder_yaw_joint", 0.0],
  ["right_elbow_joint", 0.0],
  ["right_wrist_roll_joint", 0.0],
  ["right_wrist_pitch_joint", 0.0],
  ["right_wrist_yaw_joint", 0.0]
];

const DEFAULT_STATUS: RuntimeStatus = {
  phase: "idle",
  message: "等待加载 MuJoCo WASM 和 G1 场景。",
  moduleExportNames: [],
  assetFileCount: 0,
  modelLoaded: false,
  bodyCount: 0,
  qposCount: 0,
  qvelCount: 0,
  actuatorCount: 0,
  sceneBytes: 0,
  simulationTime: 0,
  stepCount: 0,
  stepRate: 0,
  lastError: null
};

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) {
    return error.stack ?? error.message;
  }
  return String(error);
}

function listExportNames(moduleValue: unknown): string[] {
  if (!moduleValue || typeof moduleValue !== "object") {
    return [];
  }
  return Object.keys(moduleValue as Record<string, unknown>).sort();
}

function ensureDirectoryExists(mujocoModule: MainModule, directoryPath: string): void {
  if (!directoryPath || directoryPath === "/") {
    return;
  }
  const segments = directoryPath.split("/").filter(Boolean);
  let currentPath = "";
  for (const segment of segments) {
    currentPath += `/${segment}`;
    try {
      mujocoModule.FS.mkdir(currentPath);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      if (!errorMessage.includes("File exists")) {
        throw error;
      }
    }
  }
}

function getNumberProperty(source: unknown, propertyName: string): number {
  if (!source || typeof source !== "object") {
    return 0;
  }
  const value = (source as Record<string, unknown>)[propertyName];
  return typeof value === "number" ? value : 0;
}

function getObjectProperty(source: unknown, propertyName: string): Record<string, unknown> | null {
  if (!source || typeof source !== "object") {
    return null;
  }
  const value = (source as Record<string, unknown>)[propertyName];
  return value && typeof value === "object" ? value as Record<string, unknown> : null;
}

function getNumericArray(source: unknown, propertyName: string): ArrayLike<number> | null {
  if (!source || typeof source !== "object") {
    return null;
  }
  const value = (source as Record<string, unknown>)[propertyName];
  if (value && typeof value === "object" && "length" in value) {
    return value as ArrayLike<number>;
  }
  return null;
}

function readArray(source: ArrayLike<number> | null | undefined, offset: number, length: number, fallback = 0): number[] {
  return Array.from({ length }, (_, index) => {
    const value = Number(source?.[offset + index] ?? fallback);
    return Number.isFinite(value) ? value : fallback;
  });
}

function materialKey(rgba: number[]): string {
  return rgba.map((value) => value.toFixed(3)).join(",");
}

function getEnumValue(enumValue: unknown, fallback: number): number {
  if (enumValue && typeof enumValue === "object" && "value" in enumValue) {
    const value = Number((enumValue as { value: number }).value);
    return Number.isFinite(value) ? value : fallback;
  }
  return fallback;
}

function getScalarValue(value: unknown, fallback = 0): number {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : fallback;
  }
  if (value && typeof value === "object" && "length" in value) {
    const firstValue = Number((value as ArrayLike<number>)[0]);
    return Number.isFinite(firstValue) ? firstValue : fallback;
  }
  return fallback;
}

export class WebMujocoRuntime {
  private config: SimConfigResponse | null = null;
  private manifestResponse: SimManifestResponse | null = null;
  private assetContents: SimAssetContent[] = [];
  private sceneXml = "";
  private canvas: HTMLCanvasElement | null = null;
  private mujocoModule: MainModule | null = null;
  private model: MjModel | null = null;
  private data: MjData | null = null;
  private mjvScene: MjvScene | null = null;
  private mjvOption: MjvOption | null = null;
  private mjvPerturb: MjvPerturb | null = null;
  private mjvCamera: MjvCamera | null = null;
  private fallbackCanvasContext: CanvasRenderingContext2D | null = null;
  private threeScene: THREE.Scene | null = null;
  private threeCamera: THREE.PerspectiveCamera | null = null;
  private threeRenderer: THREE.WebGLRenderer | null = null;
  private visualRoot: THREE.Group | null = null;
  private visuals = new Map<number, ThreeVisual>();
  private meshGeometryCache = new Map<number, THREE.BufferGeometry>();
  private materialCache = new Map<string, THREE.MeshStandardMaterial>();
  private animationFrameId: number | null = null;
  private accumulatedSimulationSeconds = 0;
  private previousFrameTimestamp = 0;
  private status: RuntimeStatus = { ...DEFAULT_STATUS };

  private readonly maxFrameDeltaSeconds = 0.1;
  private readonly maxStepsPerFrame = 64;
  private readonly orbitTarget = new THREE.Vector3(0, 0, 0.65);
  private orbitRadius = 4.4;
  private orbitAzimuth = -0.92;
  private orbitElevation = 0.24;

  public get snapshot(): RuntimeSnapshot {
    return { status: { ...this.status } };
  }

  public setCanvas(canvas: HTMLCanvasElement): void {
    this.canvas = canvas;
    try {
      this.initializeThreeRenderer(canvas);
      this.fallbackCanvasContext = null;
    } catch (error) {
      this.fallbackCanvasContext = canvas.getContext("2d");
      this.status = {
        ...this.status,
        phase: "error",
        message: "当前浏览器无法创建 WebGL context，无法渲染 MuJoCo 机器人。",
        lastError: formatUnknownError(error)
      };
    }
    this.renderCanvas();
  }

  public setInputs(
    config: SimConfigResponse,
    manifestResponse: SimManifestResponse,
    sceneXml: string,
    assetContents: SimAssetContent[]
  ): void {
    this.config = config;
    this.manifestResponse = manifestResponse;
    this.sceneXml = sceneXml;
    this.assetContents = assetContents;
    this.meshGeometryCache.clear();
  }

  public async load(): Promise<void> {
    if (!this.threeRenderer) {
      throw new Error("WebGL renderer is not available. MuJoCo robot rendering requires a browser with WebGL enabled.");
    }
    if (!this.config || !this.manifestResponse || !this.sceneXml || this.assetContents.length === 0) {
      throw new Error("Simulation config, manifest, scene XML, and binary assets must be loaded before runtime init.");
    }

    this.stopLoop();
    this.deleteModelData();
    this.clearVisuals();
    this.status = {
      ...DEFAULT_STATUS,
      phase: "loading",
      message: "正在加载 @mujoco/mujoco，并写入 MJCF/mesh 到 WASM 文件系统...",
      assetFileCount: this.manifestResponse.manifest.files.length,
      sceneBytes: new TextEncoder().encode(this.sceneXml).byteLength
    };
    this.renderCanvas();

    try {
      const mujocoModule = await loadMujoco({
        locateFile: (path: string) => (path.endsWith(".wasm") ? mujocoWasmUrl : path)
      });
      this.mujocoModule = mujocoModule;

      this.writeAssetsToVirtualFileSystem(mujocoModule);

      const modelPath = `/model/${this.manifestResponse.manifest.scene_path}`;
      const model = mujocoModule.MjModel.from_xml_path(modelPath);
      this.disableModelGravity(model);
      const data = new mujocoModule.MjData(model);
      this.applyDefaultJointPose(model, data);
      mujocoModule.mj_forward(model, data);

      this.model = model;
      this.data = data;
      this.createMujocoSceneObjects(mujocoModule, model);
      this.updateMujocoScene();
      this.status = {
        ...this.status,
        phase: "ready",
        message: "MuJoCo 模型已加载。当前使用 Three.js 渲染 MuJoCo scene geoms，浏览器预览默认关闭重力。",
        moduleExportNames: listExportNames(mujocoModule),
        modelLoaded: true,
        bodyCount: getNumberProperty(model, "nbody"),
        qposCount: getNumberProperty(model, "nq"),
        qvelCount: getNumberProperty(model, "nv"),
        actuatorCount: getNumberProperty(model, "nu"),
        simulationTime: getNumberProperty(data, "time"),
        lastError: null
      };
      this.renderCanvas();
    } catch (error) {
      this.status = {
        ...this.status,
        phase: "error",
        message: "MuJoCo 模型加载或仿真初始化失败。请检查 MJCF include、mesh 路径和 WASM 资源路径。",
        lastError: formatUnknownError(error)
      };
      this.renderCanvas();
    }
  }

  public start(): void {
    if (this.status.phase !== "ready" && this.status.phase !== "paused") {
      return;
    }

    this.status = {
      ...this.status,
      phase: "running",
      message: "MuJoCo WASM 仿真运行中。浏览器预览未接入 WBC，已关闭重力避免自由坍塌。"
    };
    this.accumulatedSimulationSeconds = 0;
    this.previousFrameTimestamp = performance.now();
    this.animationFrameId = requestAnimationFrame((timestamp) => this.tick(timestamp));
  }

  public pause(): void {
    this.stopLoop();
    if (this.status.phase === "running") {
      this.status = {
        ...this.status,
        phase: "paused",
        message: "已暂停。"
      };
    }
  }

  public reset(): void {
    this.stopLoop();
    if (this.mujocoModule && this.model && this.data) {
      this.mujocoModule.mj_resetData(this.model, this.data);
      this.applyDefaultJointPose(this.model, this.data);
      this.mujocoModule.mj_forward(this.model, this.data);
      this.updateMujocoScene();
    }
    this.status = {
      ...this.status,
      phase: this.status.moduleExportNames.length > 0 ? "ready" : "idle",
      message: "已重置仿真状态。",
      simulationTime: 0,
      stepCount: 0,
      stepRate: 0
    };
    this.accumulatedSimulationSeconds = 0;
    this.renderCanvas();
  }

  public stepOnce(): void {
    if (this.status.phase === "idle" || this.status.phase === "loading" || this.status.phase === "error") {
      return;
    }
    this.advanceOneStep(1000 / 60);
  }

  private tick(timestamp: number): void {
    if (this.status.phase !== "running") {
      return;
    }
    const elapsedMilliseconds = Math.max(0, timestamp - this.previousFrameTimestamp);
    this.previousFrameTimestamp = timestamp;
    this.advanceSimulation(elapsedMilliseconds);
    this.animationFrameId = requestAnimationFrame((nextTimestamp) => this.tick(nextTimestamp));
  }

  private advanceOneStep(elapsedMilliseconds: number): void {
    this.advanceSimulation(elapsedMilliseconds, 1);
  }

  private advanceSimulation(elapsedMilliseconds: number, forcedStepCount?: number): void {
    if (!this.mujocoModule || !this.model || !this.data) {
      return;
    }
    const timestep = this.getModelTimestep();
    const elapsedSeconds = Math.min(elapsedMilliseconds / 1000, this.maxFrameDeltaSeconds);
    const targetStepCount = forcedStepCount ?? this.consumeAvailableStepCount(elapsedSeconds, timestep);

    if (targetStepCount <= 0) {
      this.renderCanvas();
      return;
    }

    for (let stepIndex = 0; stepIndex < targetStepCount; stepIndex += 1) {
      this.mujocoModule.mj_step(this.model, this.data);
    }

    this.updateMujocoScene();

    const nextStepCount = this.status.stepCount + targetStepCount;
    this.status = {
      ...this.status,
      simulationTime: getNumberProperty(this.data, "time") || this.status.simulationTime + timestep,
      stepCount: nextStepCount,
      stepRate: elapsedSeconds > 0 ? targetStepCount / elapsedSeconds : this.status.stepRate
    };
    this.renderCanvas();
  }

  private consumeAvailableStepCount(elapsedSeconds: number, timestep: number): number {
    this.accumulatedSimulationSeconds += elapsedSeconds;
    const availableStepCount = Math.floor(this.accumulatedSimulationSeconds / timestep);
    const stepCount = Math.min(availableStepCount, this.maxStepsPerFrame);
    this.accumulatedSimulationSeconds -= stepCount * timestep;
    return stepCount;
  }

  private getModelTimestep(): number {
    const modelOption = getObjectProperty(this.model, "opt");
    const modelTimestep = getNumberProperty(modelOption, "timestep");
    if (modelTimestep > 0) {
      return modelTimestep;
    }
    return this.config?.default_timestep ?? 0.005;
  }

  private initializeThreeRenderer(canvas: HTMLCanvasElement): void {
    this.threeRenderer?.dispose();

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setClearColor(0x071126, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x071126);
    scene.fog = new THREE.Fog(0x071126, 5, 18);

    const camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.01, 100);
    camera.up.set(0, 0, 1);
    this.applyOrbitCamera(camera);
    this.installCanvasOrbitControls(canvas);

    const visualRoot = new THREE.Group();
    scene.add(visualRoot);

    const ambientLight = new THREE.HemisphereLight(0xc7d2fe, 0x1f2937, 1.5);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 2.6);
    keyLight.position.set(3, -4, 5);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x93c5fd, 1.4);
    fillLight.position.set(-3, 2, 2);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(6, 24, 0x334155, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(6, 6),
      new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.82, metalness: 0.05 })
    );
    floor.receiveShadow = true;
    scene.add(floor);

    this.threeRenderer = renderer;
    this.threeScene = scene;
    this.threeCamera = camera;
    this.visualRoot = visualRoot;
  }

  private installCanvasOrbitControls(canvas: HTMLCanvasElement): void {
    let activePointerId: number | null = null;
    let previousX = 0;
    let previousY = 0;

    canvas.onpointerdown = (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      activePointerId = event.pointerId;
      previousX = event.clientX;
      previousY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    };

    canvas.onpointermove = (event) => {
      if (activePointerId !== event.pointerId) {
        return;
      }
      event.preventDefault();
      const deltaX = event.clientX - previousX;
      const deltaY = event.clientY - previousY;
      previousX = event.clientX;
      previousY = event.clientY;
      this.rotateOrbit(deltaX, deltaY);
    };

    const stopPointerDrag = (event: PointerEvent) => {
      if (activePointerId !== event.pointerId) {
        return;
      }
      activePointerId = null;
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };

    canvas.onpointerup = stopPointerDrag;
    canvas.onpointercancel = stopPointerDrag;
    canvas.onwheel = (event) => {
      event.preventDefault();
      this.orbitRadius = Math.max(1.4, Math.min(12, this.orbitRadius * Math.exp(event.deltaY * 0.001)));
      if (this.threeCamera) {
        this.applyOrbitCamera(this.threeCamera);
        this.renderCanvas();
      }
    };

    let isMouseDragging = false;
    let previousMouseX = 0;
    let previousMouseY = 0;
    canvas.onmousedown = (event) => {
      if (event.button !== 0) {
        return;
      }
      event.preventDefault();
      isMouseDragging = true;
      previousMouseX = event.clientX;
      previousMouseY = event.clientY;
    };
    window.onmousemove = (event) => {
      if (!isMouseDragging) {
        return;
      }
      event.preventDefault();
      const deltaX = event.clientX - previousMouseX;
      const deltaY = event.clientY - previousMouseY;
      previousMouseX = event.clientX;
      previousMouseY = event.clientY;
      this.rotateOrbit(deltaX, deltaY);
    };
    window.onmouseup = () => {
      isMouseDragging = false;
    };
  }

  private rotateOrbit(deltaX: number, deltaY: number): void {
    this.orbitAzimuth -= deltaX * 0.006;
    this.orbitElevation = Math.max(-1.25, Math.min(1.25, this.orbitElevation + deltaY * 0.006));
    if (this.threeCamera) {
      this.applyOrbitCamera(this.threeCamera);
      this.renderCanvas();
    }
  }

  private applyOrbitCamera(camera: THREE.PerspectiveCamera): void {
    const horizontalRadius = this.orbitRadius * Math.cos(this.orbitElevation);
    camera.position.set(
      this.orbitTarget.x + horizontalRadius * Math.cos(this.orbitAzimuth),
      this.orbitTarget.y + horizontalRadius * Math.sin(this.orbitAzimuth),
      this.orbitTarget.z + this.orbitRadius * Math.sin(this.orbitElevation)
    );
    camera.lookAt(this.orbitTarget);
  }

  private createMujocoSceneObjects(mujocoModule: MainModule, model: MjModel): void {
    this.mjvOption = new mujocoModule.MjvOption();
    this.mjvPerturb = new mujocoModule.MjvPerturb();
    this.mjvCamera = new mujocoModule.MjvCamera();
    this.mjvScene = new mujocoModule.MjvScene(model, Math.max(2048, getNumberProperty(model, "ngeom") * 4));
    mujocoModule.mjv_defaultOption(this.mjvOption);
    mujocoModule.mjv_defaultPerturb(this.mjvPerturb);
    mujocoModule.mjv_defaultCamera(this.mjvCamera);
    mujocoModule.mjv_defaultFreeCamera(model, this.mjvCamera);
  }

  private disableModelGravity(model: MjModel): void {
    const gravity = model.opt.gravity as ArrayLike<number> & { [index: number]: number };
    gravity[0] = 0;
    gravity[1] = 0;
    gravity[2] = 0;
  }

  private applyDefaultJointPose(model: MjModel, data: MjData): void {
    const qpos = getNumericArray(data, "qpos") as (ArrayLike<number> & { [index: number]: number }) | null;
    const qvel = getNumericArray(data, "qvel") as (ArrayLike<number> & { [index: number]: number }) | null;
    if (!qpos) {
      return;
    }

    for (const [jointName, jointValue] of DEFAULT_JOINT_POSE) {
      const joint = model.jnt(jointName);
      const qposAddress = getScalarValue(joint.qposadr, -1);
      if (qposAddress >= 0 && qposAddress < qpos.length) {
        qpos[qposAddress] = jointValue;
      }
      const qvelAddress = getScalarValue(joint.dofadr, -1);
      if (qvel && qvelAddress >= 0 && qvelAddress < qvel.length) {
        qvel[qvelAddress] = 0;
      }
    }
  }

  private updateMujocoScene(): void {
    if (!this.mujocoModule || !this.model || !this.data || !this.mjvOption || !this.mjvPerturb || !this.mjvCamera || !this.mjvScene) {
      return;
    }

    this.mujocoModule.mjv_updateScene(
      this.model,
      this.data,
      this.mjvOption,
      this.mjvPerturb,
      this.mjvCamera,
      getEnumValue(this.mujocoModule.mjtCatBit.mjCAT_ALL, 7),
      this.mjvScene
    );
    this.syncThreeVisualsFromMujocoScene();
  }

  private syncThreeVisualsFromMujocoScene(): void {
    if (!this.mujocoModule || !this.model || !this.mjvScene || !this.visualRoot) {
      return;
    }

    const activeVisualIndexes = new Set<number>();
    for (let geomIndex = 0; geomIndex < this.mjvScene.ngeom; geomIndex += 1) {
      const geom = this.mjvScene.geoms.get(geomIndex);
      if (!geom) {
        continue;
      }
      const visual = this.getOrCreateVisual(geomIndex, geom);
      if (!visual) {
        continue;
      }
      this.applyGeomTransform(visual.mesh, geom);
      activeVisualIndexes.add(geomIndex);
    }

    for (const [geomIndex, visual] of this.visuals) {
      if (!activeVisualIndexes.has(geomIndex)) {
        this.visualRoot.remove(visual.mesh);
        this.visuals.delete(geomIndex);
      }
    }
  }

  private getOrCreateVisual(geomIndex: number, geom: MjvGeom): ThreeVisual | null {
    const geomType = Number(geom.type);
    const dataId = this.getModelGeomDataId(geom);
    const rgba = readArray(geom.rgba as ArrayLike<number>, 0, 4, 1);
    const signature = `${geomType}:${dataId}:${materialKey(rgba)}`;
    const existingVisual = this.visuals.get(geomIndex);
    if (existingVisual?.signature === signature) {
      return existingVisual;
    }

    if (existingVisual && this.visualRoot) {
      this.visualRoot.remove(existingVisual.mesh);
    }

    const geometry = this.createGeometryForGeom(geom);
    if (!geometry || !this.visualRoot) {
      return null;
    }

    const mesh = new THREE.Mesh(geometry, this.getMaterial(rgba));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    this.visualRoot.add(mesh);

    const visual = { mesh, signature };
    this.visuals.set(geomIndex, visual);
    return visual;
  }

  private createGeometryForGeom(geom: MjvGeom): THREE.BufferGeometry | null {
    if (!this.mujocoModule || !this.model) {
      return null;
    }

    const geomType = Number(geom.type);
    const mjtGeom = this.mujocoModule.mjtGeom;
    if (geomType === getEnumValue(mjtGeom.mjGEOM_MESH, 7)) {
      return this.getMeshGeometry(this.getModelGeomDataId(geom));
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_BOX, 6)) {
      return new THREE.BoxGeometry(2, 2, 2);
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_SPHERE, 2) || geomType === getEnumValue(mjtGeom.mjGEOM_ELLIPSOID, 4)) {
      return new THREE.SphereGeometry(1, 32, 16);
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_CYLINDER, 5)) {
      const geometry = new THREE.CylinderGeometry(1, 1, 2, 32);
      geometry.rotateX(Math.PI / 2);
      return geometry;
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_CAPSULE, 3)) {
      const geometry = new THREE.CapsuleGeometry(1, 2, 12, 24);
      geometry.rotateX(Math.PI / 2);
      return geometry;
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_PLANE, 0)) {
      return null;
    }
    return null;
  }

  private getModelGeomDataId(geom: MjvGeom): number {
    if (this.mujocoModule && this.model) {
      const geomObjectType = getEnumValue(this.mujocoModule.mjtObj?.mjOBJ_GEOM, 5);
      const geomId = Number(geom.objid);
      const geomDataIds = getNumericArray(this.model, "geom_dataid");
      const modelGeomDataId = Number(geomDataIds?.[geomId] ?? -1);
      if (Number(geom.objtype) === geomObjectType && geomId >= 0 && modelGeomDataId >= 0) {
        return modelGeomDataId;
      }
    }
    return Number(geom.dataid);
  }

  private getMeshGeometry(meshId: number): THREE.BufferGeometry | null {
    if (!this.model || meshId < 0) {
      return null;
    }
    const cachedGeometry = this.meshGeometryCache.get(meshId);
    if (cachedGeometry) {
      return cachedGeometry;
    }

    const compiledGeometry = this.getCompiledMeshGeometry(meshId);
    if (compiledGeometry) {
      this.meshGeometryCache.set(meshId, compiledGeometry);
    }
    return compiledGeometry;
  }

  private getCompiledMeshGeometry(meshId: number): THREE.BufferGeometry | null {
    if (!this.model || meshId < 0) {
      return null;
    }

    const meshVertAdr = getNumericArray(this.model, "mesh_vertadr");
    const meshVertNum = getNumericArray(this.model, "mesh_vertnum");
    const meshFaceAdr = getNumericArray(this.model, "mesh_faceadr");
    const meshFaceNum = getNumericArray(this.model, "mesh_facenum");
    const meshVert = getNumericArray(this.model, "mesh_vert");
    const meshFace = getNumericArray(this.model, "mesh_face");
    if (!meshVertAdr || !meshVertNum || !meshFaceAdr || !meshFaceNum || !meshVert || !meshFace) {
      return null;
    }

    const vertexOffset = Number(meshVertAdr[meshId] ?? 0);
    const vertexCount = Number(meshVertNum[meshId] ?? 0);
    const faceOffset = Number(meshFaceAdr[meshId] ?? 0);
    const faceCount = Number(meshFaceNum[meshId] ?? 0);
    if (vertexCount <= 0 || faceCount <= 0) {
      return null;
    }

    const positions = new Float32Array(vertexCount * 3);
    for (let vertexIndex = 0; vertexIndex < vertexCount * 3; vertexIndex += 1) {
      positions[vertexIndex] = Number(meshVert[vertexOffset * 3 + vertexIndex] ?? 0);
    }

    const indices = new Uint32Array(faceCount * 3);
    for (let faceIndex = 0; faceIndex < faceCount * 3; faceIndex += 1) {
      indices[faceIndex] = Number(meshFace[faceOffset * 3 + faceIndex] ?? 0);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return geometry;
  }

  private getMaterial(rgba: number[]): THREE.MeshStandardMaterial {
    const key = materialKey(rgba);
    const cachedMaterial = this.materialCache.get(key);
    if (cachedMaterial) {
      return cachedMaterial;
    }
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      metalness: 0.08,
      roughness: 0.58,
      transparent: rgba[3] < 0.98,
      opacity: rgba[3]
    });
    this.materialCache.set(key, material);
    return material;
  }

  private applyGeomTransform(mesh: THREE.Mesh, geom: MjvGeom): void {
    const { position, rotation } = this.getGeomWorldTransform(geom);
    const scale = this.getGeomScale(geom);

    mesh.matrix.set(
      rotation[0] * scale.x, rotation[1] * scale.y, rotation[2] * scale.z, position[0],
      rotation[3] * scale.x, rotation[4] * scale.y, rotation[5] * scale.z, position[1],
      rotation[6] * scale.x, rotation[7] * scale.y, rotation[8] * scale.z, position[2],
      0, 0, 0, 1
    );
    mesh.matrixWorldNeedsUpdate = true;
  }

  private getGeomWorldTransform(geom: MjvGeom): { position: number[]; rotation: number[] } {
    if (this.mujocoModule && this.data) {
      const geomObjectType = getEnumValue(this.mujocoModule.mjtObj?.mjOBJ_GEOM, 5);
      const geomId = Number(geom.objid);
      const geomXpos = getNumericArray(this.data, "geom_xpos");
      const geomXmat = getNumericArray(this.data, "geom_xmat");
      if (Number(geom.objtype) === geomObjectType && geomId >= 0 && geomXpos && geomXmat) {
        return {
          position: readArray(geomXpos, geomId * 3, 3, 0),
          rotation: readArray(geomXmat, geomId * 9, 9, 0)
        };
      }
    }

    return {
      position: readArray(geom.pos as ArrayLike<number>, 0, 3, 0),
      rotation: readArray(geom.mat as ArrayLike<number>, 0, 9, 0)
    };
  }

  private getGeomScale(geom: MjvGeom): THREE.Vector3 {
    if (!this.mujocoModule) {
      return new THREE.Vector3(1, 1, 1);
    }

    const geomType = Number(geom.type);
    const size = readArray(geom.size as ArrayLike<number>, 0, 3, 1);
    const mjtGeom = this.mujocoModule.mjtGeom;
    if (geomType === getEnumValue(mjtGeom.mjGEOM_MESH, 7)) {
      return new THREE.Vector3(1, 1, 1);
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_BOX, 6) || geomType === getEnumValue(mjtGeom.mjGEOM_ELLIPSOID, 4)) {
      return new THREE.Vector3(size[0], size[1], size[2]);
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_SPHERE, 2)) {
      return new THREE.Vector3(size[0], size[0], size[0]);
    }
    if (geomType === getEnumValue(mjtGeom.mjGEOM_CYLINDER, 5) || geomType === getEnumValue(mjtGeom.mjGEOM_CAPSULE, 3)) {
      return new THREE.Vector3(size[0], size[0], size[1]);
    }
    return new THREE.Vector3(1, 1, 1);
  }

  private stopLoop(): void {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  private writeAssetsToVirtualFileSystem(mujocoModule: MainModule): void {
    ensureDirectoryExists(mujocoModule, "/model");
    for (const assetContent of this.assetContents) {
      const virtualPath = `/model/${assetContent.path}`;
      const directoryPath = virtualPath.slice(0, virtualPath.lastIndexOf("/"));
      ensureDirectoryExists(mujocoModule, directoryPath);
      try {
        mujocoModule.FS.unlink(virtualPath);
      } catch {
        // Missing files are expected on first load.
      }
      mujocoModule.FS.writeFile(virtualPath, assetContent.data);
    }
  }

  private renderCanvas(): void {
    if (!this.threeRenderer || !this.threeScene || !this.threeCamera) {
      this.renderFallbackCanvas();
      return;
    }

    this.resizeRendererToCanvas();
    this.threeRenderer.render(this.threeScene, this.threeCamera);
  }

  private resizeRendererToCanvas(): void {
    if (!this.canvas || !this.threeRenderer || !this.threeCamera) {
      return;
    }
    const width = Math.max(1, Math.floor(this.canvas.clientWidth || this.canvas.width));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight || this.canvas.height));
    const drawingBufferSize = new THREE.Vector2();
    this.threeRenderer.getSize(drawingBufferSize);
    if (drawingBufferSize.x !== width || drawingBufferSize.y !== height) {
      this.threeRenderer.setSize(width, height, false);
      this.threeCamera.aspect = width / height;
      this.threeCamera.updateProjectionMatrix();
    }
  }

  private renderFallbackCanvas(): void {
    if (!this.canvas || !this.fallbackCanvasContext) {
      return;
    }
    const canvasContext = this.fallbackCanvasContext;
    const width = this.canvas.width;
    const height = this.canvas.height;
    canvasContext.clearRect(0, 0, width, height);
    canvasContext.fillStyle = "#071126";
    canvasContext.fillRect(0, 0, width, height);
    canvasContext.strokeStyle = "rgba(148, 163, 184, 0.18)";
    for (let x = 0; x <= width; x += 48) {
      canvasContext.beginPath();
      canvasContext.moveTo(x, 0);
      canvasContext.lineTo(x, height);
      canvasContext.stroke();
    }
    for (let y = 0; y <= height; y += 48) {
      canvasContext.beginPath();
      canvasContext.moveTo(0, y);
      canvasContext.lineTo(width, y);
      canvasContext.stroke();
    }
    canvasContext.fillStyle = "#dbeafe";
    canvasContext.font = "22px Inter, system-ui, sans-serif";
    canvasContext.textAlign = "center";
    canvasContext.textBaseline = "middle";
    canvasContext.fillText(this.status.message, width / 2, height / 2);
  }

  private clearVisuals(): void {
    if (this.visualRoot) {
      for (const visual of this.visuals.values()) {
        this.visualRoot.remove(visual.mesh);
      }
    }
    this.visuals.clear();
  }

  private deleteModelData(): void {
    this.mjvScene?.delete();
    this.mjvOption?.delete();
    this.mjvPerturb?.delete();
    this.mjvCamera?.delete();
    this.data?.delete();
    this.model?.delete();
    this.mjvScene = null;
    this.mjvOption = null;
    this.mjvPerturb = null;
    this.mjvCamera = null;
    this.data = null;
    this.model = null;
  }
}
