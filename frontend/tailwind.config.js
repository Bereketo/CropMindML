/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js}"],
  theme: {
    extend: {
      keyframes:{
        animate:{
          '0%,10%,100%':{
            width:'0%'
          },
          '70%,80%,90%':{
            width:'100%'
          },
        }
      }
    },
    animation:{
      animate:'animate 2s linear infinite'
    }
  },
  plugins: [],
}