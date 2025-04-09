# UI Project Setup Instructions

There seems to be an issue with the Rollup dependency in the UI project. This is a common issue with Vite and Rollup on Windows machines. To fix this issue, please follow these steps:

1. Navigate to the UI directory:
```bash
cd ui
```

2. Remove the node_modules folder and package-lock.json:
```bash
rm -rf node_modules package-lock.json
```

3. Install the dependencies again, along with the Material UI icons package:
```bash
npm install
npm install @mui/icons-material
```

4. Start the UI development server:
```bash
npm run dev
```

If you continue to experience issues with Rollup, you may need to add the @rollup/rollup-win32-x64-msvc package explicitly:
```bash
npm install @rollup/rollup-win32-x64-msvc
```

## Alternative: Using Vite directly in Development

If the UI project still doesn't start, you can try running Vite directly:

```bash
npx vite
```

## Starting both API and UI

After fixing the UI issues, you can start both the API and UI servers:

1. In one terminal, start the API:
```bash
cd api
npm run dev
```

2. In another terminal, start the UI:
```bash
cd ui
npm run dev
```

The API will be available at http://localhost:8080 and the UI will be available at http://localhost:5173.