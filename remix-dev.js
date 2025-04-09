/**
 * This is a simple, direct way to run the Remix development server
 * when you're having trouble with the npm scripts.
 * 
 * Run with: node remix-dev.js
 */

const { spawnSync } = require('child_process');
const path = require('path');

// Get the path to the remix-dev executable
const remixDevPath = path.resolve(
  __dirname,
  'node_modules',
  '.bin',
  process.platform === 'win32' ? 'remix-dev.cmd' : 'remix-dev'
);

// Run remix-dev with the appropriate arguments
console.log('Starting Remix development server...');
const result = spawnSync(remixDevPath, ['dev'], { 
  stdio: 'inherit',
  shell: true,
  env: {
    ...process.env,
    NODE_ENV: 'development',
  }
});

if (result.error) {
  console.error('Error starting Remix development server:', result.error);
  process.exit(1);
}