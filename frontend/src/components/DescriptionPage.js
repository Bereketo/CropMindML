import React from 'react';
import cropDescriptions from './CropDescriptions'; // Path to your CropDescriptions file
import TypeWriterEffect from './typewriter'; // Path to your TypeWriterEffect component

const DescriptionPage = ({ predictedCrop}) => {
  const description = cropDescriptions[predictedCrop.toLowerCase()];

  return (
    <div className="description">
      <TypeWriterEffect text={description}
      />
    </div>
  );
};

export default DescriptionPage;
