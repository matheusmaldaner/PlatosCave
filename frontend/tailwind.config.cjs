// PlatosCave/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,jsx,ts,tsx}",
    "./src/components/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'base-white': '#FFFFFF',
        'base-gray': '#F7F7F7',
        // Warm cream palette inspired by FluxRFP
        'cream': {
          50: '#FFFDF9',
          100: '#FDF9F3',
          200: '#FAF5ED',
          300: '#F5EDE0',
          400: '#EDE3D3',
        },
        // Premium green accents
        'brand-green': {
          50: '#ECFDF5',
          100: '#D1FAE5',
          200: '#A7F3D0',
          300: '#6EE7B7',
          400: '#34D399',
          DEFAULT: '#10B981',
          600: '#059669',
          700: '#047857',
          dark: '#065F46',
        },
        // Text colors
        'text-primary': '#1A1A1A',
        'text-secondary': '#6B7280',
        'text-muted': '#9CA3AF',
        // Card accents
        'card-border': '#E5E7EB',
        'card-shadow': 'rgba(0, 0, 0, 0.04)',
      },
      fontFamily: {
        'serif': ['Playfair Display', 'Georgia', 'Cambria', 'Times New Roman', 'serif'],
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      fontSize: {
        'hero': ['3.5rem', { lineHeight: '1.1', letterSpacing: '-0.02em' }],
        'hero-lg': ['4.5rem', { lineHeight: '1.05', letterSpacing: '-0.02em' }],
      },
      boxShadow: {
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)',
        'card-lg': '0 10px 25px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04)',
        'card-xl': '0 20px 50px -10px rgba(0, 0, 0, 0.12), 0 8px 16px -6px rgba(0, 0, 0, 0.06)',
        'pill': '0 2px 8px rgba(0, 0, 0, 0.06)',
        'search': '0 4px 20px rgba(0, 0, 0, 0.08)',
      },
      borderRadius: {
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.7s ease-out both',
        'fade-in-up-delayed': 'fadeInUp 0.8s ease-out 0.15s both',
        'fade-in-right': 'fadeInRight 0.8s ease-out 0.3s both',
        'float': 'float 6s ease-in-out infinite',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInRight: {
          '0%': { opacity: '0', transform: 'translateX(30px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
      },
      backgroundImage: {
        'gradient-warm': 'linear-gradient(135deg, #FFFDF9 0%, #FDF9F3 50%, #FAF5ED 100%)',
        'gradient-hero': 'radial-gradient(ellipse at top right, #FDF9F3 0%, #FFFDF9 50%, #FAF5ED 100%)',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
