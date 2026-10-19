import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Frontend dev server. /api requests are proxied to the FastAPI backend on 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: Number(process.env.PORT || 5173),
    proxy: {
      "/api": {
        target: process.env.TRAINKIT_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
