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

/** Main-process check: package Node path works (Electron main = Node env). */
async function mainProcessVad() {
  const { createVad } = await import("@fluidinference/fluidvad");
  const vad = await createVad();
  // 2s of silence exercises load + inference + state machine end-to-end
  const events = vad.push(new Float32Array(32000));
  if (vad.isSpeaking) throw new Error("silence must not trigger speech");
  return { frames: 32000 / 512, events: events.length };
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

app.whenReady().then(async () => {
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

  const rendererResult = new Promise((resolve) => {
    ipcMain.once("smoke-result", (_e, payload) => resolve(payload));
    setTimeout(() => resolve({ ok: false, error: "renderer timeout (20s)" }), 20000);
  });
  await createWindow();
  const r = await rendererResult;
  if (r.ok) {
    console.log(`[smoke] renderer ok: ${r.frames} frames, isSpeaking=${r.isSpeaking}`);
  } else {
    console.error("[smoke] renderer FAILED:", r.error);
    failed = true;
  }
  app.exit(failed ? 1 : 0);
});

app.on("window-all-closed", () => app.quit());
