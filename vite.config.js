import { defineConfig } from 'vite';
import { vitePlugin as remix } from '@remix-run/dev';
import tsconfigPaths from 'vite-tsconfig-paths';

export default defineConfig({
  plugins: [
    remix({
      serverModuleFormat: 'esm',
    }),
    tsconfigPaths(),
  ],
  server: {
    port: 3000,
  },
});