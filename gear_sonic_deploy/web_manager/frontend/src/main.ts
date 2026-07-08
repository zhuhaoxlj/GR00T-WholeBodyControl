import { loadBinaryAsset, loadSimAssetManifest, loadSimConfig, loadTextAsset } from "./api";
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
      <div>
        <p class="eyebrow">MuJoCo WASM Phase 2</p>
        <h1>Sonic Web Simulator</h1>
        <p class="subtitle">
          这个页面现在会在浏览器中加载 G1 MJCF 和 mesh，创建 MuJoCo WASM model/data，
          执行真实 mj_step，并用 WebGL 渲染 MuJoCo scene geoms。
        </p>
      </div>
      <a class="link-button" href="/">返回 Motion Manager</a>
    </section>

    <section class="layout">
      <article class="panel simulator-panel">
        <canvas class="sim-canvas" id="simCanvas" width="1280" height="720"></canvas>
        <div class="controls">
          <button id="loadButton">Load</button>
          <button id="startButton" class="secondary">Start</button>
          <button id="pauseButton" class="secondary">Pause</button>
          <button id="stepButton" class="secondary">Step</button>
          <button id="resetButton" class="ghost">Reset</button>
        </div>
      </article>

      <aside class="panel side-panel">
        <h2>Runtime</h2>
        <dl class="metric-grid">
          <div><dt>Phase</dt><dd id="phaseValue">idle</dd></div>
          <div><dt>Sim Time</dt><dd id="simTimeValue">0.000s</dd></div>
          <div><dt>Steps</dt><dd id="stepCountValue">0</dd></div>
          <div><dt>Step Rate</dt><dd id="stepRateValue">0.0Hz</dd></div>
          <div><dt>Scene Bytes</dt><dd id="sceneBytesValue">0</dd></div>
          <div><dt>Assets</dt><dd id="assetFileCountValue">0</dd></div>
          <div><dt>Bodies</dt><dd id="bodyCountValue">0</dd></div>
          <div><dt>qpos/qvel</dt><dd id="stateSizeValue">0 / 0</dd></div>
          <div><dt>Actuators</dt><dd id="actuatorCountValue">0</dd></div>
          <div><dt>Exports</dt><dd id="exportsValue">0</dd></div>
        </dl>
        <p id="runtimeMessage" class="message">等待加载。</p>
        <pre id="runtimeError" class="error-box"></pre>
      </aside>
    </section>

    <section class="panel diagnostics-panel">
      <h2>Asset Diagnostics</h2>
      <div class="diagnostics-grid">
        <div>
          <h3>Scene</h3>
          <pre id="sceneDetails">未加载</pre>
        </div>
        <div>
          <h3>Manifest</h3>
          <pre id="manifestDetails">未加载</pre>
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
const exportsValue = document.querySelector<HTMLDListElement>("#exportsValue")!;
const runtimeMessage = document.querySelector<HTMLParagraphElement>("#runtimeMessage")!;
const runtimeError = document.querySelector<HTMLPreElement>("#runtimeError")!;
const sceneDetails = document.querySelector<HTMLPreElement>("#sceneDetails")!;
const manifestDetails = document.querySelector<HTMLPreElement>("#manifestDetails")!;
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
  exportsValue.textContent = String(status.moduleExportNames.length);
  runtimeMessage.textContent = status.message;
  runtimeError.textContent = status.lastError ?? "";
}

async function loadSimulationInputs(): Promise<void> {
  runtimeMessage.textContent = "正在加载 FastAPI 仿真配置和资产 manifest...";
  const [config, manifestResponse] = await Promise.all([loadSimConfig(), loadSimAssetManifest()]);
  const sceneXml = await loadTextAsset(config.scene_url);
  const assetContents = await Promise.all(
    manifestResponse.manifest.files.map(async (assetFile) => ({
      path: assetFile.path,
      data: await loadBinaryAsset(assetFile.url)
    }))
  );
  runtime.setInputs(config, manifestResponse, sceneXml, assetContents);
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

setInterval(renderRuntimeStatus, 100);
renderRuntimeStatus();
