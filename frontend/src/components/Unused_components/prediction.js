import React, { useState } from 'react';
import image_index from './index_image';
import axios from 'axios';
import plant from '../images/plant.png'


const PredictionPage = () => {
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [showDiv, setShowDiv] = useState(false);
  const [humidity, setHumidity] = useState('');
  const [rainfall, setRainfall] = useState('');
  const [temperature, setTemperature] = useState('');
  const [ph, setPh] = useState('');
  
  // Assuming response.data contains the string you want to modify


  const handlePredict = async () => {
    const humidityValue = parseFloat(humidity);
    const rainfallValue = parseFloat(rainfall);
    const temperatureValue = parseFloat(temperature);
    const phValue = parseFloat(ph);
    // setImageToShow(predictedImage);

    if (
      (isNaN(humidityValue) || isNaN(rainfallValue) || isNaN(temperatureValue) || isNaN(phValue)) ||
      (humidityValue === 0 || rainfallValue === 0 || temperatureValue === 0 || phValue === 0)
    ) {
      setShowDiv(true);
      setAlertMessage('Please enter valid inputs.');
      setShowAlert(true);
    } else if (phValue === 0) {
      setShowDiv(true);
      setAlertMessage("pH value can't be zero.");
      setShowAlert(true);
    } else {
      try {
        // Create a data object with form values
        const formData = {
          temperature: temperatureValue,
          humidity: humidityValue,
          ph: phValue,
          rainfall: rainfallValue
        };

        //Send form data to Django backend
        const response = await axios.post('http://localhost:8000/crop/home/form-data/', formData);


        const originalString = response.data;

        // Convert to lowercase
        const lowercaseString = originalString.toLowerCase();
        
        // Remove spaces
        const stringWithoutSpaces = lowercaseString.replace(/\s/g, '');
        
        
          const predictedCrop = stringWithoutSpaces ; 
          
          const predictedImage = image_index[predictedCrop];

        // Display the response from the backend
        setShowDiv(true);
        setAlertMessage(`Prediction: ${response.data}`);
        setShowAlert(true);
        setImageToShow(predictedImage);
      } catch (error) {
        console.error(error);
      }
    }
 
  };

  const [imageToShow, setImageToShow] = useState(null);
  return (
    
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <div className="bg-white w-full absolute md:w-11/12 lg:w-11/12 xl:w-11/12 p-5 border-t-4 border-4 border-[#396E8D] relative p-5">
      <img
  src={plant}
  alt="Background"
  className="w-28 h-28 absolute left-0"
/>
      <h1 className="text-[#396E8D] font-Poppins font-bold text-xl sm:text-lg md:text-xl lg:text-2xl xl:text-3xl mb-4 border-b-4 border-[#396E8D]">
  Please Enter These Values:
</h1>
        {/* <hr style={{ width: '1340.008px', height: '9px', background: '#396E8D' }} /> */}
<br/>
        <div className="flex items-center justify-center h-full">
          <div className={`absolute inset-0 flex items-center justify-center ${showDiv ? 'visible' : 'hidden'} w-4/5 h-5/6 m-auto border-4 border-[#396E8D] bg-zinc-300 p-6 rounded-lg shadow-lg transition-transform transform flex-col`}>
          <h1 className='text-left absolute top-0 font-bold mb-4 border-b-4 border-[#396E8D] w-full text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-5xl py-2 px-4 sm:px-6 lg:px-8 text-[#396E8D]'>
  Result
</h1>
         <button
              onClick={() => setShowDiv(false)}
              className="absolute top-0 right-0 p-2 text-gray-600 hover:text-red-600 focus:outline-none"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <p className="text-[#396E8D]">{alertMessage}</p>
            <div className="flex items-center justify-center mt-6">
  {imageToShow && (
    <img
      src={imageToShow}
      alt="Predicted Crop"
      className="max-w-full w-36 h-36 absolute bottom-10"
    />
  )}
</div>

          </div>
          <div className="flex flex-col space-y-8 ">
          <div className="flex flex-col">
              <label htmlFor="temperature" className="text-[#396E8D] font-bold mb-2">
                Temperature
              </label>
              <input
                type="number"
                id="temperature"
                className="border-2 bg-gray-200 p-2 rounded-md focus:border-2-[#000] w-w-80"
                placeholder="0.00"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                required
              />
            </div>
            <div className="flex flex-col">
              <label htmlFor="humidity" className="text-[#396E8D] font-bold mb-2">
                Humidity
              </label>
              <input 
                type="number"
                id="humidity"
                className="border-2 bg-gray-200 p-2 rounded-md focus:border-2-[#000] w-80"
                placeholder="0.00"
                value={humidity}
                onChange={(e) => setHumidity(e.target.value)}
                required
              />
            </div>
            <div className="flex flex-col">
              <label htmlFor="ph" className="text-[#396E8D] font-bold mb-2">
                pH
              </label>
              <input
                type="number"
                id="ph"
                className="border-2 bg-gray-200 p-2 rounded-md focus:border-2-[#000] w-w-80"
                placeholder="0.00"
                value={ph}
                onChange={(e) => setPh(e.target.value)}
                required
              />
            </div>
            <div className="flex flex-col">
              <label htmlFor="rainfall" className="text-[#396E8D] font-bold mb-2">
                Rainfall
              </label>
              <input
                type="number"
                id="rainfall"
                className="border-2 bg-gray-200 p-2 rounded-md focus:border-2-[#000] w-w-80"
                placeholder="0.00"
                value={rainfall}
                onChange={(e) => setRainfall(e.target.value)}
                required
              />
            </div>
          </div>
        </div>
        <button
          onClick={handlePredict}
          className="mt-4 bg-[#396E8D] text-white py-2 px-10 font-bold text-2xl rounded-md focus:outline-none cursor-pointer"
        >
          Predict
        </button>
      </div>
    </div>
  );
};

export default PredictionPage;