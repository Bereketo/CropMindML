import React from 'react';
import logoImage from '../images/logo.png'; // Replace with your logo image file
import { FaTwitter, FaFacebook, FaLinkedin } from 'react-icons/fa';

function Footer() {
  return (
    <footer className="bg-[#f9f9f9] px-5 py-4 sticky">
      <div className="container mx-auto flex justify-between items-center">
        {/* Logo */}
        <div className="flex items-center space-x-2">
          <img src={logoImage} alt="Logo" className="w-5 md:w-15 lg:w-20 xl:w-20" />
        </div>

        {/* Copyright */}
        <div className="text-[#407391] text-center text-xs sm:text-base md:text-lg lg:text-xl">
  &copy; {new Date().getFullYear()} SebilAi. All rights reserved.
</div>

        {/* Social Media Icons */}
        <div className="flex items-center space-x-4">
          <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-[#407391]">
            <FaTwitter className="text-2xl" />
          </a>
          <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="text-[#407391]">
            <FaFacebook className="text-2xl" />
          </a>
          <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className="text-[#407391]">
            <FaLinkedin className="text-2xl" />
          </a>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
