/// <reference types="@remix-run/dev" />
/// <reference types="vite/client" />

import type { MetaFunction as RemixMetaFunction } from "@remix-run/node"; // `@remix-run/server-runtime` or `@remix-run/cloudflare`

declare global {
  /**
   * A generic interface for the meta export
   * allowing for easy composition and extension.
   */
  interface MetaFunction<Loader extends (...args: any) => any = (...args: any) => any>
    extends RemixMetaFunction<Loader> {}
}

// This types our app's environment variables
// which we can access via process.env
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test';
      DATABASE_URL?: string;
      DB_HOST?: string;
      DB_PORT?: string;
      DB_NAME?: string;
      DB_USER?: string;
      DB_PASSWORD?: string;
    }
  }
}