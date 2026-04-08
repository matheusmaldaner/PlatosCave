// PlatosCave/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,jsx,ts,tsx}",
    "./src/components/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        'serif': ['"EB Garamond"', 'Georgia', 'Cambria', 'serif'],
        'sans': ['"DM Sans"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        'base-white': '#FFFFFF',
        'base-gray': '#F7F7F7',
        'brand-green': {
          light: '#D4E9D4',
          DEFAULT: '#4CAF50',
          dark: '#388E3C',
        },
        'text-primary': '#202123',
        'text-secondary': '#6E6E6E',
        'academic': {
          ink: '#1a1a2e',
          muted: '#6b7280',
          border: '#e2e2e2',
          surface: '#fafaf9',
          cream: '#f8f7f4',
          accent: '#2563eb',
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}