import React from 'react';
import image from '../images/maximize.jpg'

function Maximize() {
  return (
    <div className=" h-80 lg:h-70 md:h-50 sm:h-30 bg-cover bg-center bg-fixed" style={{ backgroundImage: `url(${image})` }}>
      <div className="flex items-center justify-center h-full">
        <h1 className="text-white text-2xl md:text-6xl lg:text-7x1 text-center px-4 font-bold">
          Maximize Your Yield With <br />
          Data-Driven Recommendations
        </h1>
      </div>
    </div>
  );
}

export default Maximize;
