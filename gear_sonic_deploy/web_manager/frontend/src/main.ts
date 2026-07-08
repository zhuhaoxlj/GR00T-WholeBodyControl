import { loadBinaryAsset, loadSimAssetManifest, loadSimConfig, loadTextAsset, loadWbcManifest } from "./api";
import { WebMujocoRuntime } from "./mujocoRuntime";
import "./styles.css";

const runtime = new WebMujocoRuntime();

const appElement = document.querySelector<HTMLDivElement>("#app");
if (!appElement) {
  throw new Error("Missing #app root element.");
}

appElement.innerHTML = `
  <main class="shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">MuJoCo WASM / WBC LAB</p>
        <h1>Sonic Web Simulator</h1>
        <p class="subtitle">
          Browser-side G1 scene, mesh assets, MuJoCo stepping, GR00T WBC ONNX inference, and WebGL scene rendering.
        </p>
      </div>
      <div class="hero-actions">
        <a class="link-button ghost" href="/">Motion Manager</a>
        <button id="loadButton" class="load-action">Load Runtime</button>
      </div>
    </section>

    <section class="layout">
      <article class="panel simulator-panel">
        <div class="canvas-header">
          <div>
            <p class="panel-kicker">Scene View</p>
            <h2>G1 Control Surface</h2>
          </div>
          <div class="phase-pill">
            <span>Phase</span>
            <strong id="phaseValue">idle</strong>
          </div>
        </div>
        <canvas class="sim-canvas" id="simCanvas" width="1280" height="720"></canvas>
        <div class="control-deck">
          <div class="control-group">
            <span class="control-label">Playback</span>
            <button id="startButton" class="secondary">Start</button>
            <button id="pauseButton" class="secondary">Pause</button>
            <button id="stepButton" class="secondary">Step</button>
            <button id="resetButton" class="ghost">Reset</button>
          </div>
          <div class="control-group command-bar">
            <span class="control-label">Command</span>
            <button id="walkBackwardButton" class="ghost">-X</button>
            <button id="walkForwardButton" class="secondary">+X</button>
            <button id="strafeLeftButton" class="ghost">+Y</button>
            <button id="strafeRightButton" class="ghost">-Y</button>
            <button id="turnLeftButton" class="ghost">+Yaw</button>
            <button id="turnRightButton" class="ghost">-Yaw</button>
            <button id="stopCommandButton" class="secondary">Stop Cmd</button>
          </div>
        </div>
      </article>

      <aside class="panel side-panel">
        <div class="panel-heading">
          <p class="panel-kicker">Runtime</p>
          <h2>Telemetry</h2>
        </div>
        <dl class="metric-grid">
          <div><dt>Sim Time</dt><dd id="simTimeValue">0.000s</dd></div>
          <div><dt>Steps</dt><dd id="stepCountValue">0</dd></div>
          <div><dt>Step Rate</dt><dd id="stepRateValue">0.0Hz</dd></div>
          <div><dt>Scene Bytes</dt><dd id="sceneBytesValue">0</dd></div>
          <div><dt>Assets</dt><dd id="assetFileCountValue">0</dd></div>
          <div><dt>Bodies</dt><dd id="bodyCountValue">0</dd></div>
          <div><dt>qpos/qvel</dt><dd id="stateSizeValue">0 / 0</dd></div>
          <div><dt>Actuators</dt><dd id="actuatorCountValue">0</dd></div>
          <div><dt>WBC</dt><dd id="wbcValue">off</dd></div>
          <div><dt>Policy</dt><dd id="policyValue">off</dd></div>
          <div><dt>Cmd</dt><dd id="commandValue">0 / 0 / 0</dd></div>
          <div><dt>ONNX</dt><dd id="inferenceValue">0.0ms</dd></div>
          <div><dt>Exports</dt><dd id="exportsValue">0</dd></div>
        </dl>
        <p id="runtimeMessage" class="message">等待加载。</p>
        <pre id="runtimeError" class="error-box"></pre>
      </aside>
    </section>

    <section class="panel diagnostics-panel">
      <div class="panel-heading">
        <p class="panel-kicker">Asset Diagnostics</p>
        <h2>Loaded Manifests</h2>
      </div>
      <div class="diagnostics-grid">
        <div>
          <h3>Scene</h3>
          <pre id="sceneDetails">未加载</pre>
        </div>
        <div>
          <h3>Manifest</h3>
          <pre id="manifestDetails">未加载</pre>
        </div>
        <div>
          <h3>WBC Policy</h3>
          <pre id="wbcDetails">未加载</pre>
        </div>
      </div>
    </section>
  </main>
`;

const phaseValue = document.querySelector<HTMLDListElement>("#phaseValue")!;
const simTimeValue = document.querySelector<HTMLDListElement>("#simTimeValue")!;
const stepCountValue = document.querySelector<HTMLDListElement>("#stepCountValue")!;
const stepRateValue = document.querySelector<HTMLDListElement>("#stepRateValue")!;
const sceneBytesValue = document.querySelector<HTMLDListElement>("#sceneBytesValue")!;
const assetFileCountValue = document.querySelector<HTMLDListElement>("#assetFileCountValue")!;
const bodyCountValue = document.querySelector<HTMLDListElement>("#bodyCountValue")!;
const stateSizeValue = document.querySelector<HTMLDListElement>("#stateSizeValue")!;
const actuatorCountValue = document.querySelector<HTMLDListElement>("#actuatorCountValue")!;
const wbcValue = document.querySelector<HTMLDListElement>("#wbcValue")!;
const policyValue = document.querySelector<HTMLDListElement>("#policyValue")!;
const commandValue = document.querySelector<HTMLDListElement>("#commandValue")!;
const inferenceValue = document.querySelector<HTMLDListElement>("#inferenceValue")!;
const exportsValue = document.querySelector<HTMLDListElement>("#exportsValue")!;
const runtimeMessage = document.querySelector<HTMLParagraphElement>("#runtimeMessage")!;
const runtimeError = document.querySelector<HTMLPreElement>("#runtimeError")!;
const sceneDetails = document.querySelector<HTMLPreElement>("#sceneDetails")!;
const manifestDetails = document.querySelector<HTMLPreElement>("#manifestDetails")!;
const wbcDetails = document.querySelector<HTMLPreElement>("#wbcDetails")!;
const simCanvas = document.querySelector<HTMLCanvasElement>("#simCanvas")!;

runtime.setCanvas(simCanvas);

function renderRuntimeStatus(): void {
  const { status } = runtime.snapshot;
  phaseValue.textContent = status.phase;
  simTimeValue.textContent = `${status.simulationTime.toFixed(3)}s`;
  stepCountValue.textContent = String(status.stepCount);
  stepRateValue.textContent = `${status.stepRate.toFixed(1)}Hz`;
  sceneBytesValue.textContent = String(status.sceneBytes);
  assetFileCountValue.textContent = String(status.assetFileCount);
  bodyCountValue.textContent = String(status.bodyCount);
  stateSizeValue.textContent = `${status.qposCount} / ${status.qvelCount}`;
  actuatorCountValue.textContent = String(status.actuatorCount);
  wbcValue.textContent = status.wbcLoaded ? "loaded" : "off";
  policyValue.textContent = status.policyMode;
  commandValue.textContent = status.command.map((value) => value.toFixed(2)).join(" / ");
  inferenceValue.textContent = `${status.policyInferenceMs.toFixed(1)}ms`;
  exportsValue.textContent = String(status.moduleExportNames.length);
  runtimeMessage.textContent = status.message;
  runtimeError.textContent = status.lastError ?? "";
}

async function loadSimulationInputs(): Promise<void> {
  runtimeMessage.textContent = "正在加载 FastAPI 仿真配置和资产 manifest...";
  const [config, manifestResponse, wbcManifestResponse] = await Promise.all([
    loadSimConfig(),
    loadSimAssetManifest(),
    loadWbcManifest()
  ]);
  const sceneXml = await loadTextAsset(config.scene_url);
  const assetContents = await Promise.all(
    manifestResponse.manifest.files.map(async (assetFile) => ({
      path: assetFile.path,
      data: await loadBinaryAsset(assetFile.url)
    }))
  );
  runtime.setInputs(config, manifestResponse, wbcManifestResponse.manifest, sceneXml, assetContents);
  sceneDetails.textContent = JSON.stringify({
    scene_path: config.scene_path,
    scene_url: config.scene_url,
    default_timestep: config.default_timestep,
    first_240_chars: sceneXml.slice(0, 240)
  }, null, 2);
  manifestDetails.textContent = JSON.stringify({
    file_count: manifestResponse.manifest.files.length,
    loaded_asset_bytes: assetContents.reduce((totalBytes, asset) => totalBytes + asset.data.byteLength, 0),
    scene_references: manifestResponse.manifest.scene_references,
    include_references: manifestResponse.manifest.include_references.map((reference) => ({
      path: reference.path,
      meshes: reference.references.meshes.length,
      meshdir: reference.references.meshdir
    }))
  }, null, 2);
  wbcDetails.textContent = JSON.stringify({
    config_path: wbcManifestResponse.manifest.config.config_path,
    resource_root: wbcManifestResponse.manifest.config.resource_root,
    num_actions: wbcManifestResponse.manifest.config.num_actions,
    num_obs: wbcManifestResponse.manifest.config.num_obs,
    obs_history_len: wbcManifestResponse.manifest.config.obs_history_len,
    control_decimation: wbcManifestResponse.manifest.config.control_decimation,
    policies: wbcManifestResponse.manifest.policies
  }, null, 2);
  await runtime.load();
  renderRuntimeStatus();
}

document.querySelector<HTMLButtonElement>("#loadButton")!.addEventListener("click", () => {
  loadSimulationInputs().catch((error: unknown) => {
    runtimeError.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
  });
});
document.querySelector<HTMLButtonElement>("#startButton")!.addEventListener("click", () => runtime.start());
document.querySelector<HTMLButtonElement>("#pauseButton")!.addEventListener("click", () => runtime.pause());
document.querySelector<HTMLButtonElement>("#stepButton")!.addEventListener("click", () => runtime.stepOnce());
document.querySelector<HTMLButtonElement>("#resetButton")!.addEventListener("click", () => runtime.reset());
document.querySelector<HTMLButtonElement>("#walkForwardButton")!.addEventListener("click", () => runtime.nudgeCommand("x", 0.1));
document.querySelector<HTMLButtonElement>("#walkBackwardButton")!.addEventListener("click", () => runtime.nudgeCommand("x", -0.1));
document.querySelector<HTMLButtonElement>("#strafeLeftButton")!.addEventListener("click", () => runtime.nudgeCommand("y", 0.1));
document.querySelector<HTMLButtonElement>("#strafeRightButton")!.addEventListener("click", () => runtime.nudgeCommand("y", -0.1));
document.querySelector<HTMLButtonElement>("#turnLeftButton")!.addEventListener("click", () => runtime.nudgeCommand("yaw", 0.1));
document.querySelector<HTMLButtonElement>("#turnRightButton")!.addEventListener("click", () => runtime.nudgeCommand("yaw", -0.1));
document.querySelector<HTMLButtonElement>("#stopCommandButton")!.addEventListener("click", () => runtime.stopCommand());

setInterval(renderRuntimeStatus, 100);
renderRuntimeStatus();
