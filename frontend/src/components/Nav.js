import React, { useState } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { Link as ScrollLink} from 'react-scroll';

import logo from "../images/logo.png";
import { hover } from "@testing-library/user-event/dist/hover";

const NavigationBar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const [isMobile, setIsMobile] = useState(false);
  const isSamePage = location.pathname === '/';
  const linkId = isSamePage ? 'predict' : 'pre';

  const handleLinkClick = () => {
    setMenuOpen(false); // Close the menu when a link is clicked
  };


  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <nav className="border-b border-b-[#e6ebf4] p-5 nav">
      <div className="flex items-center justify-between relative">
        <div className="flex items-center">
          <img src={logo} alt="Logo" className="w-20 h-15 mr-4" />
        </div>
        <div className="flex items-center space-x-4">
          <div className="hidden lg:flex space-x-4">
            <NavLink
              to="/Home"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === "/Home"
                  ? "text-[#396E8D] border-b-2 border-[#396E8D] "
                  : "opacity-50"
              }`}
            >
              Home
            </NavLink>
            <NavLink
              to="/about"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === "/about"
                  ? "text-[#396E8D] border-b-2 border-[#396E8D]"
                  : "opacity-50"
              }`}
            >
              About
            </NavLink>
            <NavLink
              to="/contact"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === "/contact"
                  ? "text-blue-[#396E8D] border-b-2 border-[#396E8D]"
                  : "opacity-50"
              }`}
            >
              Contact
            </NavLink>
            <NavLink  to="/faq"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === "/faq"
                  ? "text-blue-[#396E8D] border-b-2  border-[#396E8D]"
                  : "opacity-50"
              }`} >
              FAQ
            </NavLink>
          </div>
          <NavLink
              smooth
              duration={500}
              offset={-50}
              to='Home/pre'
              className={"bg-[#396E8D] hover:bg-[#2d5973] px-[10px] py-[5px] text-white rounded-lg font-bold"}
            >
              Predict
            </NavLink>
            <button className="text-[#396E8D] lg:hidden" onClick={toggleMenu}>
      {menuOpen ? (
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Add SVG content for the cross (X) shape */}
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      ) : (
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Add SVG content for the hamburger icon */}
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      )}
    </button>
        </div>
      </div>
      {menuOpen && (
        <div className="lg:hidden m-auto w-full flex flex-col items-center border-b-2 border-[#396e8d] shadow-md justify-center absolute top-20 right-0 z-10 bg-white">
          <div className="block text-[#396E8D] font-semibold mb-2">
            <NavLink
              to="/Home"
              className={`relative w-fit text-[#396E8D] block font-semibold ...`}
              onClick={handleLinkClick} // Close the menu when the link is clicked
            >
              Home
            </NavLink>
            <NavLink
              to="/about"
              className={`relative w-fit text-[#396E8D] block font-semibold ...`}
              onClick={handleLinkClick} // Close the menu when the link is clicked
            >
              About
            </NavLink>
            <NavLink
              to="/contact"
              className={`relative w-fit text-[#396E8D] block font-semibold ...`}
              onClick={handleLinkClick} // Close the menu when the link is clicked
            >
              Contact
            </NavLink>
            <NavLink
              to="/faq"
              className={`relative w-fit text-[#396E8D] block font-semibold ...`}
              onClick={handleLinkClick} // Close the menu when the link is clicked
            >
              FAQ
            </NavLink>
          </div>
        </div>
      )}
    </nav>
  );
};

export default NavigationBar;
