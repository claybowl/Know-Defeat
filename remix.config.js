/** @type {import('@remix-run/dev').AppConfig} */
module.exports = {
  ignoredRouteFiles: ["**/.*"],
  appDirectory: "app",
  assetsBuildDirectory: "public/build",
  publicPath: "/build/",
  serverBuildPath: "build/index.js",
  future: {},
  serverModuleFormat: "cjs",
  // Add the aliases configuration
  serverDependenciesToBundle: [
    "@chakra-ui/icons",
    "@chakra-ui/react",
    "@emotion/react",
    "@emotion/styled",
    "framer-motion"
  ],
  routes(defineRoutes) {
    return defineRoutes((route) => {
      // Define your routes here
    });
  },
  // Add tilde alias for app directory paths
  watchPaths: ["./app"],
  mdx: {
    remarkPlugins: [],
    rehypePlugins: []
  }
};