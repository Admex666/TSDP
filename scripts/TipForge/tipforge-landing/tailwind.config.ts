const config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-base': '#1E1E1E',
        'dark-darker': '#121212',
        'neon-blue': '#00D4FF',
        'silver': '#C0C0C0',
        'gray-custom': '#A9A9A9',
        'success-green': '#00D98E',
        'warning-orange': '#FF6B35',
      },
      fontFamily: {
        'heading': ['Montserrat', 'sans-serif'],
        'body': ['Inter', 'sans-serif'],
        'mono': ['Roboto Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;