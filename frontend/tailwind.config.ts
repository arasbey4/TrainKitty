export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        surface: {
          light: "#ffffff",
          dark: "#111418",
        },
        ink: {
          DEFAULT: "#1a1d23",
          soft: "#5a6472",
          faint: "#8b95a3",
        },
        accent: {
          DEFAULT: "#2563eb",
          hover: "#1d4ed8",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15,20,30,0.06), 0 1px 3px rgba(15,20,30,0.04)",
      },
    },
  },
  plugins: [],
};
