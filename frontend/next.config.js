// Enable React strict mode and standalone output for Vercel deployment
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  async rewrites() {
    // Only use localhost proxy in development
    if (process.env.NODE_ENV === 'development') {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/api/:path*',
        },
      ];
    }
    // In production, let Vercel handle the routing
    return [];
  },
};

module.exports = nextConfig; 