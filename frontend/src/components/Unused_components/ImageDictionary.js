// // ImageDictionary.js
// import React, { useState } from 'react';
// import image_index from './index_image';

// const ImageDictionary = () => {
//   const [userInput, setUserInput] = useState('');

//   const handleInputChange = (event) => {
//     setUserInput(event.target.value.toLowerCase());
//   };

//   const filteredImages = Object.keys(image_index).filter((imageName) =>
//     imageName.toLowerCase().includes(userInput)
//   );

//   return (
//     <div className="p-10">
//       <h1 className="text-3xl font-semibold mb-4">Image Dictionary</h1>
//       <input
//         type="text"
//         className="border p-2 rounded-md shadow-md mb-4"
//         placeholder="Search for images"
//         value={userInput}
//         onChange={handleInputChange}
//       />
//       <div className="grid grid-cols-3 gap-4">
//         {filteredImages.map((imageName) => (
//           <div key={imageName} className="flex flex-col items-center">
//             <img
//               src={image_index[imageName]}
//               alt={imageName}
//               className="w-24 h-24 mb-2"
//             />
//             <p className="text-sm font-semibold">{imageName}</p>
//           </div> */}
//         ))}
//        </div>
//     </div>
//   );
// };

// export default ImageDictionary;
