import React, { useState } from 'react';
import sebil from "../images/logo.png";

function ContactForm() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();

    alert(`Form submitted:\nName: ${name}\nEmail: ${email}\nMessage: ${message}`);

    setName('');
    setEmail('');
    setMessage('');
  };

  return (
    <div className="bg-light-green-200 min-h-screen p-8 flex flex-col items-center">
      <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-md">
        <img src={sebil} alt="Sebil AI" className="w-40 h-25 rounded-full mb-4 mx-auto" />
        <h1 className="text-2xl font-bold mb-4 text-center">
          You can freely share your comment or any idea on Sebil AI
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
            <label className="block text-gray-700 font-semibold">Email:</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:border-blue-500"
            />
          </div>
          <div className="mb-4">
            <label className="block text-gray-700 font-semibold">Message:</label>
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
              className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ContactForm;
