import React, { useState } from "react";
import style from "./style.css";

const FAQ = () => {
  const faqData = [
    {
      question: "What is a Crop?",
      answer: (
        <>
          A crop is a cultivated plant that is grown on a large scale for
          agricultural or commercial purposes. Crops are essential for food,
          fibers, medicinal plants, and more. Sebil AI focuses on recommending
          suitable crops based on various factors to enhance agricultural
          practices.
        </>
      ),
    },
    {
      question: "What Does This System Work?",
      answer: (
        <>
          Sebil AI is a revolutionary tool that leverages artificial
          intelligence to provide personalized crop recommendations. By
          analyzing factors like humidity, pH, temperature, and rainfall, the
          system assists farmers in making informed decisions about which crops
          to cultivate. This leads to improved yields, efficient resource
          utilization, and sustainable practices.
        </>
      ),
    },
    {
      question: "How Does It Predict Crop Recommendations?",
      answer: (
        <>
          Sebil AI employs machine learning algorithms to analyze data collected
          from various sources, such as climate stations and soil testing. These
          algorithms learn patterns from historical data and correlate them with
          crop growth conditions. When a user inputs specific parameters, Sebil
          AI matches them against the learned patterns to provide accurate crop
          recommendations tailored to the user's location and conditions.
        </>
      ),
    },
    {
      question: "What is Humidity?",
      answer: (
        <>
          Humidity refers to the amount of moisture present in the air. In
          agriculture, humidity levels influence plant growth, transpiration,
          and disease susceptibility. Sebil AI considers humidity data to
          recommend crops that thrive in varying humidity conditions.
        </>
      ),
    },
    {
      question: "What is pH?",
      answer: (
        <>
          pH measures the acidity or alkalinity of the soil. Different crops
          have specific pH preferences. Sebil AI utilizes pH information to
          suggest crops that are well-suited to the soil’s pH level, promoting
          optimal growth.
        </>
      ),
    },
    {
      question: "What is Temperature?",
      answer: (
        <>
          Temperature plays a crucial role in determining crop growth and
          development. Each crop has a range of temperature tolerance. Sebil AI
          analyzes temperature data to suggest crops that flourish in specific
          temperature conditions.
        </>
      ),
    },
    {
      question: "What is Rainfall?",
      answer: (
        <>
          Rainfall quantifies the amount of precipitation received in an area.
          It directly impacts water availability for crops. Sebil AI factors in
          rainfall data to recommend crops adapted to different levels of
          precipitation.
        </>
      ),
    },
    {
      question: "How Accurate are the Recommendations?",
      answer: (
        <>
          Sebil AI strives for high accuracy by utilizing advanced machine
          learning techniques. The recommendations are based on extensive data
          analysis, resulting in reliable guidance for crop selection.
        </>
      ),
    },
    {
      question: "Can I Use Sebil AI for Different Crops?",
      answer: (
        <>
          Absolutely! Sebil AI supports a diverse range of crops, making it
          adaptable to various agricultural needs. Whether you're cultivating
          grains, vegetables, fruits, or more, Sebil AI can assist you.
        </>
      ),
    },
    {
      question: "Is Sebil AI Accessible on Mobile Devices?",
      answer: (
        <>
          Yes, Sebil AI is designed to be accessible on mobile devices. The
          responsive design ensures that you can use the system seamlessly on
          smartphones and tablets, allowing for convenient access while you're
          in the field.
        </>
      ),
    },
    {
      question: "How Can I Provide Feedback on the Recommendations?",
      answer: (
        <>
          Your feedback is valuable! Sebil AI encourages users to provide
          feedback through the platform. Your insights help refine the system's
          accuracy and improve future recommendations, making the tool even more
          beneficial.
        </>
      ),
    },
    // Add more questions and answers here...
  ];

  const [searchQuery, setSearchQuery] = useState("");
  const [filteredFAQ, setFilteredFAQ] = useState(faqData);
  const [expandedQuestionIndex, setExpandedQuestionIndex] = useState(null);
  const [itemFound, setItemFound] = useState(true);

  const handleSearchChange = (e) => {
    const query = e.target.value.toLowerCase();
    const filteredItems = faqData.filter((item) =>
      item.question.toLowerCase().includes(query)
    );

    setSearchQuery(query);
    setFilteredFAQ(filteredItems);
  };

  const toggleAnswer = (index) => {
    if (expandedQuestionIndex === index) {
      // Clicking the same question again, so close it.
      setExpandedQuestionIndex(null);
    } else if (index >= 0 && index < faqData.length) {
      // Clicking a different question, so close the currently open one (if any) and open the new one.
      setExpandedQuestionIndex(index);
    } else {
      // Item not found, update state to indicate that the item is not found
      setItemFound(false);
    }
  };

  return (
    <div className="w-full min-h-[calc(100vh-150px)] overflow-y-auto">
      <div className="w-[80%] lg:w-[50%] md:w-[80%] mx-auto mt-8 p-4 overflow-y-hidden">
        <h1 className="text-3xl font-semibold mb-4 text-[#396E8D]">
          Frequently Asked Questions
        </h1>

        {/* Search input */}
        <input
          type="text"
          className="mb-4 p-2 rounded-lg shadow w-[60%] transition-transform duration-300 transform focus:scale-105"
          placeholder="Search..."
          value={searchQuery}
          onChange={handleSearchChange}
        />

        <ul>
          {filteredFAQ.length > 0 ? (
            filteredFAQ.map((item, index) => (
              <li key={index} className="mb-4">
                <button
                  className="text-left w-full text-[#396E8D] bg-white hover:bg-gray-100 p-2 rounded-lg shadow transition-colors duration-300 ease-in-out"
                  onClick={() => toggleAnswer(index)}
                >
                  {item.question}
                </button>
                {/* Use conditional rendering to display the answer */}
                {expandedQuestionIndex === index && (
                  <div className="p-2 text-left text-[#396E8D] mt-2 bg-gray-100 rounded-lg transition-opacity duration-300 ease-in-out opacity-100">
                    {item.answer}
                  </div>
                )}
              </li>
            ))
          ) : (
            /* Display the "Item not found" message */
            <li className="mb-4 font-semibold text-2xl text-[#396E8D] flex items-center justify-center opacity-30">
              <div className="flex flex-col justify-center items-center font-normal">
                <p class="text-center text-lg mb-2">
                  Your search -{searchQuery} - did not match any documents.
                </p>
                <div class="text-start mt-5 sm:text-sm lg:text-base md:text-base">
                  <p class="text-center text-lg font-semibold mb-2">
                    Suggestions:
                  </p>
                  <ul class=" text-lg list-disc list-inside">
                    <li class="mb-1">
                      Make sure that all words are spelled correctly.
                    </li>
                    <li class="mb-1">Try different keywords.</li>
                    <li class="mb-1">Try more general keywords.</li>
                  </ul>
                </div>
              </div>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default FAQ;
