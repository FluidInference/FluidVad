/**
 * Preload bridge (contextIsolation on): reads the wasm binary from disk
 * (asar-transparent) and exposes the bytes to the renderer, which cannot
 * fetch() file:// URLs. This is the recommended Electron integration.
 */
const { contextBridge, ipcRenderer } = require("electron");
const fs = require("node:fs");

const wasmPath = require.resolve("@fluidinference/fluidvad/dist/fluidvad_bg.wasm");
const bytes = fs.readFileSync(wasmPath);

contextBridge.exposeInMainWorld("fluidvad", {
  // Uint8Array is structured-cloneable across the bridge
  wasmBytes: new Uint8Array(bytes),
  reportSmoke: (payload) => ipcRenderer.send("smoke-result", payload),
  isSmoke: process.env.FLUIDVAD_SMOKE === "1",
});
