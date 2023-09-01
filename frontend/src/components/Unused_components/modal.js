import React from "react";
import DescriptionPage from "./DescriptionPage";
const ModalComponent = ({ result, onClose, description, image }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-opacity-50 bg-gray-500">
      <div className="bg-white w-10/12 p-4 rounded-md shadow-lg">
        <button
          onClick={onClose}
          className="absolute top-0 right-0 m-3 text-gray-600 hover:text-red-600 focus:outline-none"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <h2 className="text-xl font-bold mb-4 text-gray-800">Prediction Result</h2>
        <p className="mb-4">{`The predicted crop is: ${result}`}</p>

        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Description:</h3>
          <p>{description}</p> {/* Changed `${description}` to `{description}` */}
        </div>

        <div className="mb-4">
          <h3 className="text-lg font-semibold mb-2">Result Image:</h3>
          <img src={image} alt="Predicted Crop" className="w-28 h-auto" />
        </div>

        {/* Add more content here if needed */}
      </div>
    </div>
  );
};

export default ModalComponent;
