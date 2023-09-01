import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import logo from '../images/logo.png';
import { hover } from '@testing-library/user-event/dist/hover';

const NavigationBar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const closeMenu = () => {
    setMenuOpen(false);
  };

  return (
    <nav className="border-b border-b-[#e6ebf4] p-5 nav">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <img src={logo} alt="Logo" className="w-20 h-15 mr-4" />
        </div>
        <div className="flex items-center space-x-4">
          <div className="hidden lg:flex space-x-4">
            <Link
              to="/"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === '/' ? 'text-[#396E8D] border-b-2 border-[#396E8D] ' :  'opacity-50'
              }`}
            >
              Home
            </Link>
            <Link
              to="/about"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === '/about' ? 'text-[#396E8D] border-b-2 border-[#396E8D]' : 'opacity-50'
              }`}
            >
              About
            </Link>
            <Link
              to="/contact"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === '/contact' ? 'text-blue-[#396E8D] border-b-2 border-blue-[]' : 'opacity-50'
              }`}
            >
              Contact
            </Link>
            <Link
              to="/predict"
              className={`relative w-fit text-[#396E8D] block font-semibold after:block after:content-[''] after:absolute after:h-[2px] after:bg-blue-500 after:w-full after:scale-x-0 after:hover:scale-x-100 after:transition after:duration-300 after:origin-left ${
                location.pathname === '/predict' ? 'text-blue-[#396E8D] border-b-2 border-blue-[]' : 'opacity-50'
              }`}
            >
              Predict
            </Link>
          </div>
          <button className="text-[#396E8D] lg:hidden" onClick={toggleMenu}>
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Add SVG content here */}
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </div>
      {menuOpen && (
        <div className="lg:hidden mt-4">
          <div className="block text-[#396E8D] font-semibold mb-2">
            <Link
              to="/"
              className={`${
                location.pathname === '/' ? 'text-blue-500' : ''
              }`}
              onClick={closeMenu}
            >
              Home
            </Link>
          </div>
          <div className="block text-[#396E8D] font-semibold mb-2">
            <Link
              to="/about"
              className={`${
                location.pathname === '/about' ? 'text-blue-500' : ''
              }`}
              onClick={closeMenu}
            >
              About
            </Link>
          </div>
          <div className="block text-[#396E8D] font-semibold mb-2">
            <Link
              to="/contact"
              className={`${
                location.pathname === '/contact' ? 'text-blue-500' : ''
              }`}
              onClick={closeMenu}
            >
              Contact
            </Link>
            <Link
              to="/predict"
              className={`${
                location.pathname === '/predict' ? 'text-blue-500' : ''
              }`}
              onClick={closeMenu}
            >
              Predict
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
};


export default NavigationBar;