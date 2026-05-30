/** @type {import('next').NextConfig} */

// The FastAPI backend (api/server.py). Override with BACKEND_URL when deploying.
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  // Strict Mode double-invokes effects in dev, which fires every panel's data
  // fetch twice. For the slow options/scan endpoints that means two concurrent
  // multi-second scans where the first gets aborted (socket hang up / ECONNRESET).
  // Disable it so each panel fetches once.
  reactStrictMode: false,
  // Proxy API + websocket calls to FastAPI so the browser sees a single origin
  // (no CORS) during development. In production, put both behind one gateway.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` },
      { source: "/ws/:path*", destination: `${BACKEND_URL}/ws/:path*` },
    ];
  },
};

export default nextConfig;
