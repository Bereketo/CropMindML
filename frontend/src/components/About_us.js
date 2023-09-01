import React from "react";
import about from "../images/aboutt.jpg.png"; // Import your About image

import TeamMembers from "./TeamMembers";

const AboutPage = () => {
  return (<div>
    <div className="bg-gray-100  p-5 flex flex-col lg:flex-row-reverse items-center border-b-2 border-[#396E8D] mb-5">
      <div className="lg:w-2/4 bg-white p-6 rounded-xl shadow-md mb-8 ">
        <h1 className="text-4xl font-semibold text-[#396E8D] border-b-2 border-[#396E8D]">
          About Us
        </h1>
        <p className="text-[#396E8D] text-lg mb-4">
          Welcome to Sebil AI! We are a team of passionate individuals dedicated
          to revolutionizing crop selection practices in agriculture. Our goal is
          to provide intelligent decision-making tools that optimize crop
          choices, leading to improved yields, financial outcomes, and
          sustainable farming practices.
        </p>
        <p className="text-[#396E8D] text-lg mb-4">
          Through our innovative solution, Sebil AI, we aim to analyze critical
          factors such as soil quality, nutrient levels, and local climate
          conditions to equip agricultural stakeholders with the information
          they need to make informed decisions.
        </p>
        <p className="text-[#396E8D] text-lg border-b-2 border-[#396E8D] pb-4">
          Feel free to explore Sebil AI to learn more about our mission and the
          services we offer. If you have any questions or inquiries, please
          don't hesitate to contact us. We look forward to assisting you!
        </p>
      </div>
     
      <div className="lg:w-2/5 mx-auto">
  <img src={about} alt="About Us" className="w-full lg:w-800 rounded-lg" />
</div>
</div>
      <TeamMembers />
    </div>
  );
};

export default AboutPage;
