#!/bin/bash
# Setup development environment for Know-Defeat Remix minimal app

echo "Setting up development environment for Know-Defeat UI..."

# Install remix dependencies
echo "Installing dependencies..."
npm install

# Create directories if they don't exist
mkdir -p public/build

echo "Initializing TypeScript configuration..."
cat > tsconfig.json << EOF
{
  "include": ["remix.env.d.ts", "**/*.ts", "**/*.tsx"],
  "compilerOptions": {
    "lib": ["DOM", "DOM.Iterable", "ES2019"],
    "isolatedModules": true,
    "esModuleInterop": true,
    "jsx": "react-jsx",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "target": "ES2019",
    "strict": true,
    "allowJs": true,
    "forceConsistentCasingInFileNames": true,
    "baseUrl": ".",
    "paths": {
      "~/*": ["./app/*"]
    },
    "skipLibCheck": true,
    "noEmit": true
  }
}
EOF

echo "Creating TypeScript environment definition file..."
cat > remix.env.d.ts << EOF
/// <reference types="@remix-run/dev" />
/// <reference types="@remix-run/node" />
EOF

echo "Creating .gitignore file..."
cat > .gitignore << EOF
node_modules
.cache
build
public/build
EOF

echo "Setup complete! Next steps:"
echo "1. Run 'npm run dev' to start the development server"
echo "2. Visit http://localhost:3000 to see your application"
echo "3. Edit files in app/ directory to make changes"