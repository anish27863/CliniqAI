import type { NextConfig } from "next";

const target = process.env.API_PROXY_TARGET || "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${target}/:path*` }];
  },
};

export default nextConfig;