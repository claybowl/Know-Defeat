import { cssBundleHref } from "@remix-run/css-bundle";
import { json } from "@remix-run/node";
import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useLoaderData,
} from "@remix-run/react";

export const loader = async () => {
  return json({
    ENV: {
      NODE_ENV: process.env.NODE_ENV,
    },
  });
};

export default function App() {
  const data = useLoaderData();
  
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
        <title>Know-Defeat Trading System</title>
        <style>
          {`
            body {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 1200px;
              margin: 0 auto;
              padding: 20px;
              background-color: #f8f9fa;
            }
            header {
              background-color: #2c5282;
              color: white;
              padding: 1rem;
              border-radius: 8px;
              margin-bottom: 2rem;
            }
            h1 {
              margin: 0;
              font-size: 1.8rem;
            }
            nav {
              display: flex;
              gap: 1rem;
              margin-top: 1rem;
            }
            nav a {
              color: white;
              text-decoration: none;
              font-weight: 500;
              padding: 0.5rem 0;
              border-bottom: 2px solid transparent;
              transition: border-color 0.3s;
            }
            nav a:hover {
              border-color: white;
            }
            main {
              background: white;
              border-radius: 8px;
              padding: 20px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            footer {
              margin-top: 2rem;
              text-align: center;
              font-size: 0.9rem;
              color: #666;
            }
            .card {
              background: #f8f9fa;
              border-radius: 8px;
              padding: 20px;
              margin-bottom: 20px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
          `}
        </style>
      </head>
      <body>
        <header>
          <h1>Know-Defeat Trading System</h1>
          <nav>
            <a href="/">Dashboard</a>
            <a href="/bots">Bots</a>
            <a href="/trades">Trades</a>
            <a href="/metrics">Metrics</a>
          </nav>
        </header>
        <main>
          <Outlet />
        </main>
        <footer>
          <p>© {new Date().getFullYear()} Know-Defeat Trading System. All rights reserved.</p>
        </footer>
        <script
          dangerouslySetInnerHTML={{
            __html: `window.ENV = ${JSON.stringify(data.ENV)}`,
          }}
        />
        <ScrollRestoration />
        <Scripts />
        <LiveReload />
      </body>
    </html>
  );
}