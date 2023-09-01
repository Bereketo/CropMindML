import React, { useEffect } from "react";
import Image from "../images/img1.jpg";
import logo from "../images/logo2.png";
import NavigationBar from "./Nav"; // Import your NavigationBar component
import Maximize from "./Maximize";
import Pre from "./predict";
import { useLocation } from "react-router-dom";
import { Element } from "react-scroll";

const HomePage = () => {
  const location = useLocation();

  useEffect(() => {
    if (location.pathname === "/Home/pre") {
      const predictSection = document.getElementById("pre");
      if (predictSection) {
        predictSection.scrollIntoView({ behavior: "smooth" });
      }
    }
  }, [location.pathname]);
  return (
    <div>
      <div className="relative min-h-[calc(100vh-73px)] h-full overflow-hidden sm:p-8 px-4 py-8 w-full">
        <div
          className="absolute inset-0 bg-cover bg-center"
          style={{ backgroundImage: `url(${Image})` }}
        ></div>

        <div className="absolute m-auto top-0 left-0 w-full flex justify-center items-center p-4 md:p-6 lg:p-8">
          <img
            src={logo}
            alt="Logo"
            className="w-20 md:w-24 lg:w-32 xl:w-40 transform -translate-x-1/2 border-4 rounded-full border-[#396E8D] m-8 p-5"
          />
        </div>

        <h1
          className="absolute bottom-5 right-0 p-4 text-white text-right text-2xl md:text-2xl lg:text-9xl xl:text-6xl 2xl:text-7xl font-bold
             sm:absolute sm:bottom-0 sm:right-0 sm:text-3xl
             md:bottom-0 md:right-0"
        >
          Discover the
          <br />
          perfect crop
          <br />
          for Your
          <br />
          Environment
        </h1>
      </div>

      {/* Other content */}
      <Element name="predictComponent" id="pre">
        <Pre />
      </Element>
      <div id="pred"></div>
      <Maximize />
    </div>
  );
};

export default HomePage;
