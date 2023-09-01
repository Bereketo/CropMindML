import React, { useState, useEffect } from 'react';

const TypeWriterEffect = ({ text }) => {
  const [displayText, setDisplayText] = useState('');
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index < text.length) {
      const timer = setTimeout(() => {
        setDisplayText(prevDisplayText => prevDisplayText + text.charAt(index));
        setIndex(prevIndex => prevIndex + 1);
      }, 25); // Adjust the typing speed as needed

      return () => clearTimeout(timer);
    }
  }, [index, text]);

  return (
    <p className="text-[#396E8D]">{displayText}</p>
  );
};

export default TypeWriterEffect;
