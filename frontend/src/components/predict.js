import React, { useState, useEffect } from "react";
import image_index from "./index_image";
import axios from "axios";
import DescriptionPage from "./DescriptionPage"; // Path to your DescriptionPage component
import TypeWriterEffect from "./typewriter"; // Path to your TypeWriterEffect component
import plant from "../images/plant.png";

const PredictionPage = () => {
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState("");
  const [showDiv, setShowDiv] = useState(false);
  const [humidity, setHumidity] = useState("");
  const [rainfall, setRainfall] = useState("");
  const [redPoint, setredPoint] = useState("");
  const [ShowredPoint, setShowredPoint] = useState("");
  const [temperature, setTemperature] = useState("");
  const [ph, setPh] = useState("");
  const [descriptionToShow, setDescriptionToShow] = useState("");
  const [predictionInProgress, setPredictionInProgress] = useState(false);

  useEffect(() => {
    if (!predictionInProgress) {
      setShowDiv(false);
      setShowAlert(false);
      setredPoint(false);
      setDescriptionToShow("");
      setImageToShow(null);
    }
  }, [predictionInProgress]);

  const handlePredict = async () => {
    const humidityValue = parseFloat(humidity);
    const rainfallValue = parseFloat(rainfall);
    const temperatureValue = parseFloat(temperature);
    const phValue = parseFloat(ph);
    if (!temperatureValue || !humidityValue || !phValue || !rainfallValue) {
      alert("Please fill in all required fields");
    } else if (temperatureValue < -50 || temperatureValue > 50) {
      alert("Please enter a valid temperature (-50 to 50)");
    } else if (humidityValue < 0 || humidityValue > 100) {
      alert("Please enter a valid humidity percentage (0 to 100)");
    } else if (rainfallValue < 0) {
      alert(
        "Please enter a valid rainfall amount (greater than or equal to 0)"
      );
    } else if (phValue < 0 || phValue > 14) {
      alert("Please enter a valid pH level (0 to 14)");
    } else {
      try {
        const formData = {
          temperature: temperatureValue,
          humidity: humidityValue,
          ph: phValue,
          rainfall: rainfallValue,
        };

        const response = await axios.post(
          "http://localhost:8000/crop/home/form-data/",
          formData
        );

        const originalString = response.data;
        const lowercaseString = originalString.toLowerCase();
        const stringWithoutSpaces = lowercaseString.replace(/\s/g, "");
        const predictedCrop = stringWithoutSpaces;
        const predictedImage = image_index[predictedCrop];

        setPredictionInProgress(true);

        setShowDiv(true);
        setAlertMessage(`${response.data}`);
        setShowAlert(true);
        setImageToShow(predictedImage);
        setDescriptionToShow(predictedCrop);
        setredPoint(false);
      } catch (error) {
        console.error(error);
      } finally {
        setPredictionInProgress(false);
      }
    }
  };

  const [imageToShow, setImageToShow] = useState(null);

  return (
    <div className="flex items-center justify-center h-full bg-gray-100 mb-8 mt-8 relative " id="table">
      <img src={plant} className=" absolute truncate sm:block" />
      <div className=" bg-white w-full md:w-3/5 lg:w-3/5 xl:w-3/5 py-10 border-t-4 border-4 border-[#396E8D] relative">
        <h1 className="text-[#396E8D] flex items-center justify-center font-Poppins font-bold text-xl sm:text-lg md:text-xl lg:text-2xl xl:text-3xl border-b-4 border-[#396E8D]">
          Please Enter These Values:
        </h1>
        <br />
        <div className="flex items-center justify-center h-full">
          <div
            className={` absolute inset-0 flex  justify-center ${
              showDiv ? "visible" : "hidden"
            } w-full h-full m-auto bg-white p-6 shadow-lg transition-transform transform flex-col`}
          >
            <div>
            <h1 className="text-left font-bold mb-4 flex justify-between items-center lg:mt-8 sm:mt-0 border-b-2 border-[#396E8D] w-full text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-5xl py-2 px-4 sm:px-6 lg:px-8 text-[#396E8D]">
              Result
            </h1>
            
            <button
              onClick={() => {
                setShowDiv(false);
                setDescriptionToShow(""); // Passing an empty string to clear the description
              }}
              className="absolute top-0 right-0  p-2 text-gray-600 hover:text-red-600 font-bold  focus:outline-none"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
            </div>
            <div className="flex flex-col">
              <h1 className="text-[#396E8D] font-bold mt-0 lg:text-4xl sm:3xl mt-40px">
                {" "}
                The Predicted Crop is:{" "}
                <span className="text-black font-bold capitalize border-b-2 border-black">{alertMessage} </span>
                <br />
              </h1>
              </div>
              <div className="flex items-center justify-center lg:mt-14 sm:mt-5 sm:mb-3 w-[200px] m-auto">
            {imageToShow && (
                <img
                  src={imageToShow}
                  alt="Predicted Crop"
                  className="w-36"
                />
              )}
            </div>
              <div>
              <p className=" ml-auto mr-auto py-5 w-full  lg:w-10/12 h-80 mt-0  text-start font-bold">
                {descriptionToShow && (
                  <DescriptionPage predictedCrop={descriptionToShow} />
                )}
              </p>
              </div>
          </div>
          <div className="flex flex-col space-y-8 ">
          <div className="flex flex-col">
  <label
    htmlFor="temperature"
    className="text-[#396E8D] font-bold mb-2"
  >
    Temperature
  </label>
  <input
    type="number"
    id="temperature"
    className={`border-2 ${
      temperature < -50 || temperature > 50
        ? "border-red-500"
        : "border-gray-200"
    } p-2 rounded-md focus:border-2-[#000] w-80`}
    placeholder={`${-50}\u00b0C to ${50}\u00b0C`}
    value={temperature}
    onChange={(e) => setTemperature(e.target.value)}
    required
  />
  {temperature < -50 || temperature > 50 ? (
    <p className="text-red-500 text-sm mt-1">
      Temperature must be between -50 and 50
    </p>
  ) : null}
</div>
            <div className="flex flex-col">
              <label
                htmlFor="humidity"
                className="text-[#396E8D] font-bold mb-2"
              >
                Humidity
              </label>
              <input
                type="number"
                id="humidity"
                className={`border-2 ${
                  humidity < 0 || humidity > 100
                    ? "border-red-500"
                    : "border-gray-200"
                } p-2 rounded-md focus:border-2-[#000] w-80`}
                placeholder="0 to 100%"
                value={humidity}
                onChange={(e) => setHumidity(e.target.value)}
                required
              />
              {humidity < 0 || humidity > 100 ? (
                <p className="text-red-500 text-sm mt-1">
                  Humidity must be between 0 and 100
                </p>
              ) : null}
            </div>
            <div className="flex flex-col">
              <label htmlFor="ph" className="text-[#396E8D] font-bold mb-2">
                pH
              </label>
              <input
                type="number"
                id="ph"
                className={`border-2 ${
                  ph < 0 || ph > 14 ? "border-red-500" : "border-gray-200"
                } p-2 rounded-md focus:border-2-[#000] w-80`}
                placeholder="0 to 14"
                value={ph}
                onChange={(e) => setPh(e.target.value)}
                required
              />
              {ph < 0 || ph > 14 ? (
                <p className="text-red-500 text-sm mt-1">
                  pH must be between 0 and 14
                </p>
              ) : null}
            </div>
            <div className="flex flex-col">
              <label
                htmlFor="rainfall"
                className="text-[#396E8D] font-bold mb-2"
              >
                Rainfall
              </label>
              <input
                type="number"
                id="rainfall"
                className={`border-2 ${
                  rainfall < 0 ? "border-red-500" : "border-gray-200"
                } p-2 rounded-md focus:border-2-[#000] w-80`}
                placeholder="greater than or equal to 0mm"
                value={rainfall}
                onChange={(e) => setRainfall(e.target.value)}
                required
              />
              {rainfall < 0 ? (
                <p className="text-red-500 text-sm mt-1">
                  Rainfall must be greater than or equal to 0
                </p>
              ) : null}
            </div>
          </div>
        </div>
        <button
          onClick={handlePredict}
          className="mt-4 bg-[#396E8D] text-white py-2 px-10 font
          bold text-2xl rounded-md focus:outline-none cursor-pointer"
        >
          Predict
        </button>
        {/* DescriptionPage component */}
        {/* {descriptionToShow && (
          <DescriptionPage predictedCrop={descriptionToShow} />
        )} */}
      </div>
    </div>
  );
};

export default PredictionPage;
