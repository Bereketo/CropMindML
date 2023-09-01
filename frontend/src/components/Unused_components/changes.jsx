import React, { useState, useEffect } from "react";
import image_index from "./index_image";
import axios from "axios";
import DescriptionPage from "./DescriptionPage"; // Path to your DescriptionPage component
import TypeWriterEffect from "./TypeWriter"; // Path to your TypeWriterEffect component
import plant from "../images/plant.png";
import ModalComponent from "./modal";

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
  const [showModal, setShowModal] = useState(false);
  const [modalResult, setModalResult] = useState("");
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
        setModalResult(predictedCrop);
        setShowModal(true);
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
  const closeModal = () => {
    setShowModal(false);
    setModalResult("");
  };

  const [imageToShow, setImageToShow] = useState(null);

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100 mb-8 mt-8 relative">
      <img src={plant} className="absolute truncate sm:block" />
      <div className="bg-white w-full md:w-3/5 lg:w-3/5 xl:w-3/5 p-5 border-t-4 border-4 border-[#396E8D] relative p-5">
        <h1 className="text-[#396E8D] font-Poppins font-bold text-xl sm:text-lg md:text-xl lg:text-2xl xl:text-3xl mb-4 border-b-4 border-[#396E8D]">
          Please Enter These Values:
        </h1>
        <br />
        <div className="flex items-center justify-center h-full">

            {/* ******************************Navigation Class****************************** */}

            <div className={`Nav absolute items-center inset-0 flex flex-col  justify-center ${
              showDiv ? "visible" : "hidden"
            } w-4/5 h-full overflow-hidden m-auto border-2 border-[#396E8D] bg-slate-100 p-6 rounded-lg shadow-lg transition-transform transform flex-col`}
          >
            <div className="new flex items-center justify-center">
                <h1 className="text-left absolute top-0 font-bold mb-4 border-b-4 border-[#396E8D] w-full text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-5xl py-2 px-4 sm:px-6 lg:px-8 text-[#396E8D]">
              Result
            </h1>
            <button
              onClick={() => {
                setShowDiv(false);
                setDescriptionToShow(""); // Passing an empty string to clear the description
              }}
              className="absolute top-0 right-0 p-2 text-gray-600 hover:text-red-600 focus:outline-none"
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
            

{/* ******************************Prediction Class****************************** */}
            <div className="Prediction flex flex-col absolute">

            <h1 className="text-[#396E8D] font-bold mt-0 ">
                {" "}
                The Predicted Crop is:{" "}
                <span className="text-red-800 font-bold ">{alertMessage} </span>
                <br />
              </h1>
                  </div>
{/* ******************************Description Class****************************** */}
              <div className="Description h-80">
              <p className=" mb-5 ml-auto mr-auto py-5 w-10/12">
                {descriptionToShow && (
                  <DescriptionPage predictedCrop={descriptionToShow} />
                )}
              </p>
            </div>
{/* ******************************Image Class****************************** */}
            <div className="image flex items-center justify-center mt-14">

            {imageToShow && (
                <img
                  src={imageToShow}
                  alt="Predicted Crop"
                  className="w-28 h-28 absolute bottom-10 mt-20"
                />
              )}

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
                placeholder="-50 to 50"
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
                placeholder="0 to 100"
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
                placeholder="greater than or equal to 0"
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
