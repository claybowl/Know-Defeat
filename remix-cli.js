/**
 * This is a direct way to run Remix's CLI
 * 
 * Run with: node remix-cli.js <command>
 * Examples:
 *   node remix-cli.js dev     # Start development server
 *   node remix-cli.js build   # Build for production
 *   node remix-cli.js routes  # Show routes
 */

const { spawnSync } = require('child_process');
const path = require('path');

// Get the command from arguments or default to 'dev'
const command = process.argv[2] || 'dev';

// Get the path to the remix executable
const remixPath = path.resolve(
  __dirname,
  'node_modules',
  '@remix-run',
  'dev',
  'dist',
  'cli.js'
);

// Run remix with the given command
console.log(`Running Remix command: ${command}`);
const result = spawnSync('node', [remixPath, command], { 
  stdio: 'inherit',
  env: {
    ...process.env,
    NODE_ENV: command === 'build' ? 'production' : 'development',
  }
});

if (result.error) {
  console.error(`Error running Remix ${command}:`, result.error);
  process.exit(1);
}