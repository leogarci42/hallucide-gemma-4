import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev badge sits bottom-left, over the sidebar footer, and would show
  // up on stage. Off in every mode so dev and demo look the same.
  devIndicators: false,
};

export default nextConfig;
