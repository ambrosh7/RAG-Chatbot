import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/chat": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/documents": "http://127.0.0.1:8000",
    },
  },
});
