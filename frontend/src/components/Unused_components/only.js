import React, { useState } from 'react';

const ImageDictionary = () => {
  const [userInput, setUserInput] = useState('');
  const imageContext = require.context('./Pictures/', false, /\.(png|jpe?g|svg)$/);
  const imageFiles = imageContext.keys();
  
  // Construct the imageDictionary dynamically from image filenames
  const imageDictionary = {
    "Apple": "apple.png",
    "Banana": "banana.png",
    "AdzukiBeans": "Adzuki_Beans.png",
    "BlackGram": "black_gram.png",
    "Chickpea": "Chickpea.png",
    "Coconut": "coconut.png",
    "Coffee": "coffee.png",
    "Cotton": "cotton.png",
    "Grapes": "grapes.png",
    "Ground Nut": "ground_nut.png",
    "Jute": "jute.png",
    "KidneyBeans": "kidney_beans.png",
    "Lentil": "lentil.png",
    "Maize": "maize.png",
    "Mango": "mango.png",
    "Millet": "Millet.png",
    "MothBeans": "moth_beans.png",
    "MungBeans": "mung_beans.png",
    "Muskmelon": "muskmelon.png",
    "Orange": "orange.png",
    "Papaya": "papaya.png",
    "Peas": "peas.png",
    "Pigeon Peas": "pigeon_peas.png",
    "Rice": "rice.png",
    "Rubber": "rubber.png",
    "Sugarcane": "sugarcane.png",
    "Tea": "tea.png",
    "Tobacco": "tobacco.png",
    "Watermelon": "watermelon.png",
    "Wheat": "Wheat.png"
  };
  imageFiles.forEach((filePath) => {
    const imageName = filePath.replace('./', '').replace('.png', '');
    imageDictionary[imageName] = filePath;
  });

  const handleInputChange = (event) => {
    setUserInput(event.target.value.toLowerCase());
  };

  const filteredImages = Object.keys(imageDictionary).filter((imageName) =>
    imageName.toLowerCase().includes(userInput)
  );

  return (
    <div className="p-10">
      <h1 className="text-3xl font-semibold mb-4">Image Dictionary</h1>
      <input
        type="text"
        className="border p-2 rounded-md shadow-md mb-4"
        placeholder="Search for images"
        value={userInput}
        onChange={handleInputChange}
      />
      <div className="grid grid-cols-3 gap-4">
        {filteredImages.map((imageName) => (
          <div key={imageName} className="flex flex-col items-center">
            <img
              src={imageDictionary[imageName]}
              alt={imageName}
              className="w-24 h-24 mb-2"
            />
            <p className="text-sm font-semibold">{imageName}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ImageDictionary;
