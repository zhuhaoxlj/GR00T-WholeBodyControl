import type { MjData, MjModel } from "@mujoco/mujoco";
import * as ort from "onnxruntime-web";
import ortWasmSimdThreadedMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasmSimdThreadedWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import type { WbcPolicyConfig, WbcPolicyManifest } from "./types";

type MutableNumericArray = ArrayLike<number> & { [index: number]: number };
export type WbcPolicyMode = "off" | "balance" | "walk";

interface JointControlMapping {
  jointName: string;
  qposAddress: number;
  qvelAddress: number;
  actuatorIndex: number;
}

export interface WbcControllerStatus {
  loaded: boolean;
  policyMode: WbcPolicyMode;
  command: [number, number, number];
  policyInferenceMs: number;
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

function getNumericArray(source: unknown, propertyName: string): MutableNumericArray | null {
  if (!source || typeof source !== "object") {
    return null;
  }
  const value = (source as Record<string, unknown>)[propertyName];
  if (value && typeof value === "object" && "length" in value) {
    return value as MutableNumericArray;
  }
  return null;
}

function vectorNorm3(values: ArrayLike<number>): number {
  const x = values[0] ?? 0;
  const y = values[1] ?? 0;
  const z = values[2] ?? 0;
  return Math.hypot(x, y, z);
}

function quatRotateInverse(quat: ArrayLike<number>, vector: [number, number, number]): [number, number, number] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const q0 = w;
  const q1 = -x;
  const q2 = -y;
  const q3 = -z;
  const vx = vector[0];
  const vy = vector[1];
  const vz = vector[2];
  return [
    vx * (q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2) +
      vy * 2 * (q1 * q2 - q0 * q3) +
      vz * 2 * (q1 * q3 + q0 * q2),
    vx * 2 * (q1 * q2 + q0 * q3) +
      vy * (q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2) +
      vz * 2 * (q2 * q3 - q0 * q1),
    vx * 2 * (q1 * q3 - q0 * q2) +
      vy * 2 * (q2 * q3 + q0 * q1) +
      vz * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
  ];
}

function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}

export class WbcPolicyController {
  private readonly config: WbcPolicyConfig;
  private readonly manifest: WbcPolicyManifest;
  private balanceSession: ort.InferenceSession | null = null;
  private walkSession: ort.InferenceSession | null = null;
  private jointMappings: JointControlMapping[] = [];
  private action: Float32Array;
  private targetDofPos: Float32Array;
  private obsHistory: Float32Array[] = [];
  private readonly singleObsDim = 86;
  private stepCounter = 0;
  private command: [number, number, number];
  private policyMode: WbcPolicyMode = "off";
  private policyInferenceMs = 0;

  public constructor(manifest: WbcPolicyManifest) {
    this.manifest = manifest;
    this.config = manifest.config;
    this.command = [
      this.config.cmd_init[0] ?? 0,
      this.config.cmd_init[1] ?? 0,
      this.config.cmd_init[2] ?? 0
    ];
    this.action = new Float32Array(this.config.num_actions);
    this.targetDofPos = new Float32Array(this.config.default_angles.slice(0, this.config.num_actions));
    this.resetObservationHistory();
  }

  public get status(): WbcControllerStatus {
    return {
      loaded: Boolean(this.balanceSession && this.walkSession),
      policyMode: this.policyMode,
      command: [...this.command],
      policyInferenceMs: this.policyInferenceMs
    };
  }

  public async load(): Promise<void> {
    const wasmEnv = ort.env.wasm as typeof ort.env.wasm & {
      wasmPaths?: Record<string, string>;
      numThreads?: number;
      proxy?: boolean;
    };
    wasmEnv.wasmPaths = {
      "ort-wasm-simd-threaded.wasm": ortWasmSimdThreadedWasmUrl,
      "ort-wasm-simd-threaded.mjs": ortWasmSimdThreadedMjsUrl
    };
    wasmEnv.numThreads = 1;
    wasmEnv.proxy = false;

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };
    const [balanceSession, walkSession] = await Promise.all([
      ort.InferenceSession.create(this.manifest.policies.balance.url, sessionOptions),
      ort.InferenceSession.create(this.manifest.policies.walk.url, sessionOptions)
    ]);
    this.balanceSession = balanceSession;
    this.walkSession = walkSession;
  }

  public initializeModel(model: MjModel, data: MjData): void {
    this.jointMappings = this.config.joint_names.map((jointName, index) => {
      const joint = model.jnt(jointName);
      return {
        jointName,
        qposAddress: getScalarValue(joint.qposadr, -1),
        qvelAddress: getScalarValue(joint.dofadr, -1),
        actuatorIndex: this.resolveActuatorIndex(model, jointName, index)
      };
    });
    this.applyDefaultPose(data);
    this.resetPolicyState();
  }

  public setCommand(command: Partial<{ x: number; y: number; yaw: number }>): void {
    this.command = [
      Number.isFinite(command.x) ? Number(command.x) : this.command[0],
      Number.isFinite(command.y) ? Number(command.y) : this.command[1],
      Number.isFinite(command.yaw) ? Number(command.yaw) : this.command[2]
    ];
  }

  public nudgeCommand(axis: "x" | "y" | "yaw", delta: number): void {
    const next = { x: this.command[0], y: this.command[1], yaw: this.command[2] };
    next[axis] = clamp(next[axis] + delta, -1.2, 1.2);
    this.setCommand(next);
  }

  public stopCommand(): void {
    this.setCommand({ x: 0, y: 0, yaw: 0 });
  }

  public resetPolicyState(): void {
    this.stepCounter = 0;
    this.policyInferenceMs = 0;
    this.policyMode = "off";
    this.action.fill(0);
    for (let index = 0; index < this.config.num_actions; index += 1) {
      this.targetDofPos[index] = this.config.default_angles[index] ?? 0;
    }
    this.resetObservationHistory();
  }

  public applyDefaultPose(data: MjData): void {
    const qpos = getNumericArray(data, "qpos");
    const qvel = getNumericArray(data, "qvel");
    if (!qpos) {
      return;
    }
    this.jointMappings.forEach((mapping, index) => {
      if (mapping.qposAddress >= 0 && mapping.qposAddress < qpos.length) {
        qpos[mapping.qposAddress] = this.config.default_angles[index] ?? 0;
      }
      if (qvel && mapping.qvelAddress >= 0 && mapping.qvelAddress < qvel.length) {
        qvel[mapping.qvelAddress] = 0;
      }
    });
  }

  public applyTorques(model: MjModel, data: MjData): void {
    const qpos = getNumericArray(data, "qpos");
    const qvel = getNumericArray(data, "qvel");
    const ctrl = getNumericArray(data, "ctrl");
    if (!qpos || !qvel || !ctrl) {
      return;
    }

    this.jointMappings.forEach((mapping, index) => {
      if (mapping.actuatorIndex < 0 || mapping.actuatorIndex >= ctrl.length) {
        return;
      }
      const currentQ = mapping.qposAddress >= 0 ? qpos[mapping.qposAddress] ?? 0 : 0;
      const currentDq = mapping.qvelAddress >= 0 ? qvel[mapping.qvelAddress] ?? 0 : 0;
      const targetQ =
        index < this.config.num_actions
          ? this.targetDofPos[index] ?? 0
          : this.config.default_angles[index] ?? 0;
      const kp = this.config.kps[index] ?? 0;
      const kd = this.config.kds[index] ?? 0;
      const torque = (targetQ - currentQ) * kp - currentDq * kd;
      const [minControl, maxControl] = this.getControlRange(model, mapping.actuatorIndex);
      ctrl[mapping.actuatorIndex] = clamp(torque, minControl, maxControl);
    });
  }

  public async afterMujocoStep(data: MjData): Promise<void> {
    this.stepCounter += 1;
    if (this.stepCounter % this.config.control_decimation !== 0) {
      return;
    }
    await this.updatePolicy(data);
  }

  private resetObservationHistory(): void {
    this.obsHistory = Array.from(
      { length: this.config.obs_history_len },
      () => new Float32Array(this.singleObsDim)
    );
  }

  private resolveActuatorIndex(model: MjModel, jointName: string, fallbackIndex: number): number {
    const candidates = [jointName, jointName.replace(/_joint$/, "")];
    for (const candidate of candidates) {
      try {
        const actuator = model.actuator(candidate);
        const actuatorId = getScalarValue(actuator.id, -1);
        if (actuatorId >= 0) {
          return actuatorId;
        }
      } catch {
        // Fall through to the next candidate or the model-order fallback.
      }
    }
    const actuatorCount = Number((model as unknown as Record<string, unknown>).nu ?? 0);
    return fallbackIndex < actuatorCount ? fallbackIndex : -1;
  }

  private getControlRange(model: MjModel, actuatorIndex: number): [number, number] {
    const ctrlRange = getNumericArray(model, "actuator_ctrlrange");
    const minControl = Number(ctrlRange?.[actuatorIndex * 2] ?? -Infinity);
    const maxControl = Number(ctrlRange?.[actuatorIndex * 2 + 1] ?? Infinity);
    return [
      Number.isFinite(minControl) ? minControl : -Infinity,
      Number.isFinite(maxControl) ? maxControl : Infinity
    ];
  }

  private async updatePolicy(data: MjData): Promise<void> {
    if (!this.balanceSession || !this.walkSession) {
      return;
    }
    const observation = this.computeObservation(data);
    this.obsHistory.push(observation);
    while (this.obsHistory.length > this.config.obs_history_len) {
      this.obsHistory.shift();
    }

    const obsBuffer = new Float32Array(this.config.num_obs);
    this.obsHistory.forEach((historyItem, index) => {
      obsBuffer.set(historyItem, index * this.singleObsDim);
    });

    const commandNorm = vectorNorm3(this.command);
    const nextPolicyMode: Exclude<WbcPolicyMode, "off"> = commandNorm <= 0.05 ? "balance" : "walk";
    const session = nextPolicyMode === "balance" ? this.balanceSession : this.walkSession;
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const inputTensor = new ort.Tensor("float32", obsBuffer, [1, this.config.num_obs]);
    const startedAt = performance.now();
    const outputs = await session.run({ [inputName]: inputTensor });
    this.policyInferenceMs = performance.now() - startedAt;
    this.policyMode = nextPolicyMode;

    const outputData = outputs[outputName].data;
    for (let index = 0; index < this.config.num_actions; index += 1) {
      const actionValue = Number(outputData[index] ?? 0);
      this.action[index] = Number.isFinite(actionValue) ? actionValue : 0;
      this.targetDofPos[index] =
        this.action[index] * this.config.action_scale + (this.config.default_angles[index] ?? 0);
    }
  }

  private computeObservation(data: MjData): Float32Array {
    const qpos = getNumericArray(data, "qpos");
    const qvel = getNumericArray(data, "qvel");
    const singleObs = new Float32Array(this.singleObsDim);

    singleObs[0] = this.command[0] * (this.config.cmd_scale[0] ?? 1);
    singleObs[1] = this.command[1] * (this.config.cmd_scale[1] ?? 1);
    singleObs[2] = this.command[2] * (this.config.cmd_scale[2] ?? 1);
    singleObs[3] = this.config.height_cmd;
    singleObs[4] = this.config.rpy_cmd[0] ?? 0;
    singleObs[5] = this.config.rpy_cmd[1] ?? 0;
    singleObs[6] = this.config.rpy_cmd[2] ?? 0;

    singleObs[7] = (qvel?.[3] ?? 0) * this.config.ang_vel_scale;
    singleObs[8] = (qvel?.[4] ?? 0) * this.config.ang_vel_scale;
    singleObs[9] = (qvel?.[5] ?? 0) * this.config.ang_vel_scale;

    const gravity = quatRotateInverse(
      [qpos?.[3] ?? 1, qpos?.[4] ?? 0, qpos?.[5] ?? 0, qpos?.[6] ?? 0],
      [0, 0, -1]
    );
    singleObs[10] = gravity[0];
    singleObs[11] = gravity[1];
    singleObs[12] = gravity[2];

    this.jointMappings.forEach((mapping, index) => {
      const q = mapping.qposAddress >= 0 ? qpos?.[mapping.qposAddress] ?? 0 : 0;
      const dq = mapping.qvelAddress >= 0 ? qvel?.[mapping.qvelAddress] ?? 0 : 0;
      singleObs[13 + index] = (q - (this.config.default_angles[index] ?? 0)) * this.config.dof_pos_scale;
      singleObs[13 + this.jointMappings.length + index] = dq * this.config.dof_vel_scale;
    });

    const actionOffset = 13 + this.jointMappings.length * 2;
    for (let index = 0; index < this.config.num_actions; index += 1) {
      singleObs[actionOffset + index] = this.action[index] ?? 0;
    }
    return singleObs;
  }
}
