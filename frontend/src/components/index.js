import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Home';
import Nav from './Nav';
import PredictionPage from './predict';
import AboutPage from './About_us';
import ContactForm from './contact';
import NavigationBar from './Nav';
import Footer from './Footer_page';
import FAQ from './FAQ'

const App = () => {
  return (
    <Router>
      <div className='index'>
        <Nav />
        <Routes>
        <Route path="/" element={<Home />} />
          <Route path="/Home" element={<Home />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/contact" element={<ContactForm />} />
          <Route path="/Home/pre" element={<Home />}/>
          <Route path="/faq" element={<FAQ />} />
        </Routes>
  
        <Footer/>
      </div>
    </Router>
  );
};

export default App;
