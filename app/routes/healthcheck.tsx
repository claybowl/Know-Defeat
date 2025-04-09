import type { LoaderFunction } from "@remix-run/node";

/**
 * Health check endpoint for Google Cloud Run
 * This ensures Cloud Run can verify our application is running properly
 */
export const loader: LoaderFunction = async () => {
  // You could add database connection check here
  return new Response("OK", {
    status: 200,
    headers: {
      "Content-Type": "text/plain",
    },
  });
};

// No component rendering needed for this route
export default function Healthcheck() {
  return null;
}