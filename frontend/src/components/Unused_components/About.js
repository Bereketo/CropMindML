import React from 'react';
import sebil from '../images/talk.jpg';

const AboutPage = () => {
  return (
    <div className="bg-gray-100 min-h-screen flex flex-col lg:flex-row items-center p-8">
      <div className="lg:flex-shrink-0">
        <img src={sebil} alt="Profile Picture" className="w-11/12 rounded-xl m-auto h-auto lg:w-640 lg:h-400 object-cover  shadow-lg" />
      </div>
      <div className="mt-4 lg:mt-0 bg-slate-800 p-6 rounded-xl">
      <h1 className="text-3xl lg:text-5xl font-bold mb-4 text-slate-100 border-b-2 border-[#396E8D]">About Us</h1>
        <p className="text-slate-100 mb-4 capitalize">
          Welcome to Sebil AI! We are a team of passionate individuals dedicated to revolutionizing crop selection practices in agriculture. Our goal is to provide intelligent decision-making tools that optimize crop choices, leading to improved yields, financial outcomes, and sustainable farming practices.
        </p>
        <p className="text-slate-100 mb-4 capitalize">
          Through our innovative solution, Sebil AI, we aim to analyze critical factors such as soil quality, nutrient levels, and local climate conditions to equip agricultural stakeholders with the information they need to make informed decisions.
        </p>
        <p className="text-slate-100 capitalize border-b-2 border-[#396E8D]">
          Feel free to explore Sebil AI to learn more about our mission and the services we offer. If you have any questions or inquiries, please don't hesitate to contact us. We look forward to assisting you!
        </p>
      </div>
    </div>
  );
};

export default AboutPage;
