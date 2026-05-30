/** @type {import('next').NextConfig} */

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig = {
  // Avoid double-fetching slow scan endpoints in dev.
  reactStrictMode: false,
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` },
      { source: "/ws/:path*", destination: `${BACKEND_URL}/ws/:path*` },
    ];
  },
};

export default nextConfig;
