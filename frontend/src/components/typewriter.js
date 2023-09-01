// Inside TypeWriterEffect.js

import React, { useState, useEffect } from "react";

function TypeWriterEffect({ text }) {
  const [typedText, setTypedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setTypedText((prevTypedText) => prevTypedText + text[currentIndex]);
        setCurrentIndex((prevIndex) => prevIndex + 1);
      }, 10);

      return () => clearTimeout(timer);
    } 
  }, [currentIndex, text]);

  return <p>{typedText}</p>;
}

export default TypeWriterEffect;
