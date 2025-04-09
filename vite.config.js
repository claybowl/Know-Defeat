import { defineConfig } from 'vite';
import { vitePlugin as remix } from '@remix-run/dev';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [
    remix({
      serverModuleFormat: 'cjs', // Changed from 'esm' to 'cjs' for compatibility with Express server
    }),
    tsconfigPaths(),
  ],
  server: {
    port: 3000,
  },
});