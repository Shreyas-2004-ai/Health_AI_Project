import React from 'react';
import { Link } from 'react-router-dom';
import Footer from '../footerComponents/Footer';
import HeroBackground from '../HeroBackground';
import './demo.css';

const Demo = () => {
  // Removed unused state/effect and import to satisfy linter

  return (
    <div className="demo-container">
      {/* Hero Section */}
      <section className="hero">
        <HeroBackground />
        <div className="hero-content">
          <h1 className="hero-title animate-fade-in-up">
            AI-Powered Health Insights
          </h1>
          <p className="hero-subtitle animate-fade-in-up delay-200">
            Discover personalized health predictions and recommendations powered by advanced machine learning
          </p>
          <div className="hero-buttons animate-fade-in-up delay-400">
            <Link to="/prediction" className="btn btn-primary">
              Start Health Analysis
            </Link>
            <Link to="/about" className="btn btn-outline">
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Why Choose Health AI?</h2>
            <p className="section-subtitle">
              Our advanced AI system provides accurate health predictions and personalized recommendations
            </p>
          </div>
          
          <div className="card-grid">
            <div className="feature-card animate-fade-in-up delay-100">
              <div className="feature-icon">
                <i className="fas fa-brain"></i>
              </div>
              <h3 className="feature-title">Advanced AI</h3>
              <p className="feature-description">
                Powered by multiple machine learning models for accurate predictions
              </p>
            </div>
            
            <div className="feature-card animate-fade-in-up delay-200">
              <div className="feature-icon">
                <i className="fas fa-shield-alt"></i>
              </div>
              <h3 className="feature-title">Secure & Private</h3>
              <p className="feature-description">
                Your health data is protected with industry-standard security measures
              </p>
            </div>
            
            <div className="feature-card animate-fade-in-up delay-300">
              <div className="feature-icon">
                <i className="fas fa-clock"></i>
              </div>
              <h3 className="feature-title">Instant Results</h3>
              <p className="feature-description">
                Get health insights and recommendations in seconds
              </p>
            </div>
            
            <div className="feature-card animate-fade-in-up delay-400">
              <div className="feature-icon">
                <i className="fas fa-user-md"></i>
              </div>
              <h3 className="feature-title">Professional Guidance</h3>
              <p className="feature-description">
                Based on medical expertise and comprehensive health databases
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="section bg-secondary">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">How It Works</h2>
            <p className="section-subtitle">
              Simple steps to get your personalized health insights
            </p>
          </div>
          
          <div className="steps-container">
            <div className="step-item animate-fade-in-left delay-100">
              <div className="step-number">1</div>
              <h3>Enter Symptoms</h3>
              <p>Describe your symptoms in natural language</p>
            </div>
            
            <div className="step-arrow animate-fade-in-up delay-200">
              <i className="fas fa-arrow-right"></i>
            </div>
            
            <div className="step-item animate-fade-in-up delay-300">
              <div className="step-number">2</div>
              <h3>AI Analysis</h3>
              <p>Our AI analyzes your symptoms using multiple models</p>
            </div>
            
            <div className="step-arrow animate-fade-in-up delay-400">
              <i className="fas fa-arrow-right"></i>
            </div>
            
            <div className="step-item animate-fade-in-right delay-500">
              <div className="step-number">3</div>
              <h3>Get Results</h3>
              <p>Receive detailed health insights and recommendations</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section">
        <div className="container">
          <div className="cta-card">
            <h2 className="cta-title">Ready to Get Started?</h2>
            <p className="cta-subtitle">
              Begin your health journey with AI-powered insights
            </p>
            <Link to="/prediction" className="btn btn-primary btn-large">
              Start Health Analysis Now
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Demo;