import { defineConfig } from "vite";

export default defineConfig({
  base: "/sim/",
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8080"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
