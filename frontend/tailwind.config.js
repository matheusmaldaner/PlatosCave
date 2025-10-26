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
        sans: [
          'Trajan Pro 3',
          'Trajan Pro',
          'Trajan',
          'Cormorant Garamond',
          'Cinzel',
          'serif'
        ],
      },
      colors: {
        'base-white': '#FFFFFF',
        'base-gray': '#F7F7F7', // Sidebar background
        'brand-green': {
          light: '#D4E9D4', // Light green for elements
          DEFAULT: '#4CAF50', // Main green for buttons, text
          dark: '#388E3C',   // Darker green for hover states
        },
        'text-primary': '#202123',
        'text-secondary': '#6E6E6E',
      }
    },
  },
  // This is the important part!
  plugins: [
    require('@tailwindcss/typography'),
  ],
}

