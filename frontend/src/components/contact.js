import React, { useState } from "react";
import sebil from "../images/logo2.png";
import {
  FaPhone,
  FaEnvelope,
  FaMapMarker,
  FaTwitter,
  FaFacebook,
  FaLinkedin,
} from "react-icons/fa";
import emailjs from "emailjs-com";

const PhoneButton = () => {
  const handlePhoneClick = () => {
    window.location.href = "tel:+1234567890";
  };

  return (
    <button
      className="contact-button flex items-center gap-5 border-2 border-slate-900 p-2 rounded"
      onClick={handlePhoneClick}
    >
      <FaPhone className="contact-icon" />
      <span>+251912345678</span>
    </button>
  );
};

const EmailButton = () => {
  const handleEmailClick = () => {
    window.location.href = "mailto:sebilaidevelopers@gmail.com";
  };

  return (
    <button
      className="contact-button flex items-center gap-5 border-2 border-slate-900 p-2 rounded"
      onClick={handleEmailClick}
    >
      <FaEnvelope className="contact-icon" />
      <span>sebilaidevelopers@gmail.com</span>
    </button>
  );
};

const LocationButton = () => {
  const handleLocationClick = () => {
    window.open("https://maps.google.com?q=Your+Location");
  };

  return (
    <button
      className="contact-button flex items-center gap-5 border-2 border-slate-900 p-2 rounded"
      onClick={handleLocationClick}
    >
      <FaMapMarker className="contact-icon" />
      <span>AA CMC Dan Energy Research & Developmet PLC</span>
    </button>
  );
};

function ContactForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [showModal, setShowModal] = React.useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();

    const serviceID = "service_7gmalpj";
    const templateID = "template_oqcmv2q";
    const userID = "U9Af-ThoZYV_wQLsR";

    const templateParams = {
      from_name: name,
      from_email: email,
      message: message,
    };

    emailjs
      .send(serviceID, templateID, templateParams, userID)
      .then((response) => {
        // alert("Email sent successfully!", response);
        setName("");
        setEmail("");
        setMessage("");
        setShowModal(true, response);
      })
      .catch((error) => {
        alert("Error sending email:", error);
      });
  };

  return (
    <div className="flex flex-col m-auto lg:flex-row bg-slate-100 min-h-screen items-center">
      {/* Left Side (Quote) */}
      <div className="p-6 max-w-md mx-auto text-center lg:text-left text-slate-900">
        <h1 className="text-2xl font-bold mb-4 capitalize">Get a quote</h1>
        <p className="capitalize mb-6">
          Fill up the form and our team will get back to you in 24 hours.
        </p>
        <div className="flex flex-col gap-5">
          <PhoneButton />
          <EmailButton />
          <LocationButton />
        </div>
        <div className="flex items-center m-auto flex-col  mt-10 justify-center h-[100px] gap-2">
          <div>
            <h1>Social Media</h1>
          </div>
          <div className="flex gap-5">
            <a
              href="https://twitter.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-900 p-2 hover:border-2 border-slate-900 rounded-full"
            >
              <FaTwitter className="text-2xl" />
            </a>
            <a
              href="https://facebook.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-900 p-2 hover:border-2 border-slate-900 rounded-full"
            >
              <FaFacebook className="text-2xl" />
            </a>
            <a
              href="https://linkedin.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-900 p-2 hover:border-2 border-slate-900 rounded-full"
            >
              <FaLinkedin className="text-2xl" />
            </a>
          </div>
        </div>
      </div>
      {/* Right Side (Form) */}
      <div className="w-full bg-light-green-200 flex flex-col items-center lg:w-1/2">
        <div className="bg-white p-5 rounded-lg shadow-md">
          <img
            src={sebil}
            alt="Sebil AI"
            className="w-40 h-25 rounded-full mb-4 mx-auto "
          />
          <h1 className="text-2xl font-bold mb-4 text-center capitalize">
            Share your comment or idea on Sebil AI
          </h1>
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold">Name:</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold">
                Email:
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500"
              />
            </div>
            <div className="mb-4">
              <label className="block text-gray-700 font-semibold">
                Message:
              </label>
              <textarea
                value={message}
                required
                onChange={(e) => setMessage(e.target.value)}
                className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500 h-32 resize-none"
              />
            </div>
            <div className="text-center">
              <button
                type="submit"
                className="px-4 py-2 bg-[#396E8D] text-white rounded-lg hover:bg-blue-600 focus:outline-none"
              >
                Send Message
              </button>
            </div>
          </form>
        </div>
      </div>
      {/* Modal */}
      {showModal ? (
        <>
          <div className="justify-center items-center flex overflow-x-hidden overflow-y-auto fixed inset-0 z-50 outline-none focus:outline-none">
            <div className="relative w-full my-6 mx-auto max-w-3xl">
              {/*content*/}
              <div className="border-0 rounded-lg w-[60%] text-center m-auto shadow-lg relative flex flex-col bg-white outline-none focus:outline-none">
                {/*header*/}
                <div className="flex items-center justify-center m-auto p-5 border-b border-solid border-slate-200 rounded-t">
                  <h3 className="text-3xl text-center text-[#396E8D] font-bold">
                    SebilAi
                  </h3>
                  <button
                    className="p-1 ml-auto bg-transparent border-0 text-black opacity-5 float-right text-3xl leading-none font-semibold outline-none focus:outline-none"
                    onClick={() => setShowModal(false)}
                  >
                    <span className="bg-transparent text-black opacity-5 h-6 w-6 text-2xl block outline-none focus:outline-none">
                      ×
                    </span>
                  </button>
                </div>
                {/*body*/}
                <div className="relative p-6 flex-auto">
                  <p className="my-4 text-slate-500 text-lg leading-relaxed">
                    Email Sent successfully
                  </p>
                </div>
                {/*footer*/}
                <div className="flex items-center justify-center p-6 border-t border-solid border-slate-200 rounded-b">
                  <button
                    className="text-red-500 background-transparent font-bold uppercase px-6 py-2 text-sm outline-none focus:outline-none mr-1 mb-1 ease-linear transition-all duration-150"
                    type="button"
                    onClick={() => setShowModal(false)}
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div className="opacity-25 fixed inset-0 z-40 bg-black"></div>
        </>
      ) : null}
    </div>
  );
}

export default ContactForm;
