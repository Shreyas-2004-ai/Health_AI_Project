import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/navbarcomponents/Navbar';
import Home from './components/democomponents/demo';
import About from './components/aboutusComponents/About';
import Blogs from './components/blogsComponents/Blogs';
import Feedback from './components/feedbackComponents/Feedback';
import PredictionPage from './components/PredictionPages/PredictionPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/blogs" element={<Blogs />} />
            <Route path="/demo" element={<Home />} />
            <Route path="/feedback" element={<Feedback />} />
            <Route path="/prediction" element={<PredictionPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
