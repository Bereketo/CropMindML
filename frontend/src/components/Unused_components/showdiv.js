import React, { useState } from 'react';

const CenteredDivExample = () => {
  const [showDiv, setShowDiv] = useState(false);

  return (
    <div className="relative h-screen flex items-center justify-center bg-gray-100">
      <button
        onClick={() => setShowDiv(!showDiv)}
        className="bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:outline-none"
      >
        Toggle Div
      </button>
      {showDiv && (
        <div className="absolute inset-0 flex items-center justify-center bg-white">
          <p>This is the centered div content.</p>
        </div>
      )}
    </div>
  );
};

export default CenteredDivExample;
