import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Change 'war-gpu' to your GitHub repo name
  base: '/war-gpu/',
})
