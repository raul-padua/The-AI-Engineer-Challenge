// Enable React strict mode and standalone output for Vercel deployment
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  // Remove rewrites - let Vercel handle API routing via vercel.json
};

module.exports = nextConfig; 