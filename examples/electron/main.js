/**
 * Electron main process.
 *
 * Demonstrates both integration points:
 *  - main/Node side: FluidVad via the package's Node path (fs read, asar-transparent)
 *  - renderer side: wasm bytes handed over by the preload bridge (contextIsolation on)
 *
 * FLUIDVAD_SMOKE=1 runs both sides headless and exits 0/1 — used by CI.
 */
import { app, BrowserWindow, ipcMain, systemPreferences } from "electron";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const smoke = process.env.FLUIDVAD_SMOKE === "1";

// 64 whole frames (64 × 512 = 32768 samples = 2.048 s of silence)
const SMOKE_SAMPLES = 32768;

/** Main-process check: package Node path works (Electron main = Node env). */
async function mainProcessVad() {
  const { createVad, FluidVad } = await import("@fluidinference/fluidvad");
  const vad = await createVad();
  const events = vad.push(new Float32Array(SMOKE_SAMPLES));
  if (vad.isSpeaking) throw new Error("silence must not trigger speech");
  return { frames: SMOKE_SAMPLES / FluidVad.frameSize(), events: events.length };
}

async function createWindow() {
  const win = new BrowserWindow({
    width: 480,
    height: 360,
    show: !smoke,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false, // preload needs fs to read the wasm
    },
  });
  await win.loadFile("index.html");
  return win;
}

async function run() {
  // macOS: prompt for the microphone before the renderer asks (no-op elsewhere)
  if (process.platform === "darwin" && !smoke) {
    await systemPreferences.askForMediaAccess("microphone");
  }

  if (!smoke) {
    const result = await mainProcessVad();
    console.log(`[main] node-path VAD ok: ${result.frames} frames processed`);
    await createWindow();
    return;
  }

  // ---- smoke mode: verify both processes, then exit ----
  let failed = false;
  try {
    const result = await mainProcessVad();
    console.log(`[smoke] main ok: ${result.frames} frames`);
  } catch (e) {
    console.error("[smoke] main FAILED:", e);
    failed = true;
  }

  let settle;
  const rendererResult = new Promise((resolve) => {
    settle = resolve;
    ipcMain.once("smoke-result", (_e, payload) => resolve(payload));
  });
  try {
    await createWindow();
  } catch (e) {
    settle({ ok: false, error: `window creation failed: ${e}` });
  }
  // start the watchdog only after the page has loaded (or failed) — a cold
  // first Electron launch on a CI runner shouldn't eat the renderer's budget
  setTimeout(() => settle({ ok: false, error: "renderer timeout (20s)" }), 20000);

  const r = await rendererResult;
  if (r.ok) {
    console.log(`[smoke] renderer ok: ${r.frames} frames, isSpeaking=${r.isSpeaking}`);
  } else {
    console.error("[smoke] renderer FAILED:", r.error);
    failed = true;
  }
  app.exit(failed ? 1 : 0);
}

app.whenReady().then(() =>
  run().catch((e) => {
    // no unhandled-rejection hangs: any failure exits nonzero (CI would
    // otherwise sit at the job timeout)
    console.error(smoke ? "[smoke] FAILED:" : "[main] error:", e);
    app.exit(1);
  })
);

// In smoke mode only app.exit() decides the exit code — a closing window
// (Cmd+Q, session logout) must not convert an incomplete run into a pass.
app.on("window-all-closed", () => {
  if (!smoke) app.quit();
});
