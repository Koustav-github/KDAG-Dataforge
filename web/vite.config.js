import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  // Root-hosted by default (Vercel, Netlify, any custom domain). GitHub Pages
  // serves from /<repo>/, so set BASE_PATH at build time there:
  //   BASE_PATH=/KDAG-Dataforge/ npm run build
  base: process.env.BASE_PATH || '/',
  plugins: [svelte()],
});
