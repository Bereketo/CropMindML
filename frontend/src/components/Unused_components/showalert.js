import React, { useState } from 'react';

const ShowAlertPage = () => {
  const [message, setMessage] = useState('');
  const [isError, setIsError] = useState(false);

  const handleShowError = () => {
    setMessage('Please enter valid inputs.');
    setIsError(true);
  };


  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className={`max-w-md w-full bg-white p-6 border rounded-lg shadow-lg transition-transform transform ${isError ? 'scale-105' : ''}`}>
        <h2 className="text-2xl font-semibold mb-4">{isError ? 'Error' : 'Success'}</h2>
        <p className="text-gray-700">{message}</p>
      </div>
      <div className="flex flex-col space-y-4 mt-4">
        <button
          onClick={handleShowError}
          className="bg-red-500 hover:bg-red-600 text-white py-2 px-4 rounded-md focus:outline-none"
        >
          Show Error
        </button>
      </div>
    </div>
  );
};

export default ShowAlertPage;
