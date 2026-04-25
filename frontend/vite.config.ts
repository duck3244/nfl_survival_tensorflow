import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 백엔드는 8000 포트가 점유되어 8001로 운용. 환경변수로 덮어쓰기 가능.
const backend = process.env.VITE_API_PROXY_TARGET ?? 'http://127.0.0.1:8001'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true, // 다른 포트로 자동 이동하지 않게 (CORS 와 어긋남 방지)
    proxy: {
      '/api': {
        target: backend,
        changeOrigin: true,
      },
    },
  },
})
