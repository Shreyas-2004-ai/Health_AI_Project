import React from 'react';
import './about.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="container">
        {/* Hero Section */}
        <section className="about-hero">
          <div className="hero-content">
            <h1 className="hero-title animate-fade-in-up">
              About Health AI
            </h1>
            <p className="hero-subtitle animate-fade-in-up delay-200">
              Revolutionizing healthcare with artificial intelligence and machine learning
            </p>
          </div>
        </section>

        {/* Mission Section */}
        <section className="section">
          <div className="section-header">
            <h2 className="section-title">Our Mission</h2>
            <p className="section-subtitle">
              Empowering individuals with AI-driven health insights for better decision making
            </p>
          </div>
          
          <div className="mission-content">
            <div className="mission-card animate-fade-in-left">
              <div className="mission-icon">
                <i className="fas fa-heart"></i>
              </div>
              <h3>Health First</h3>
              <p>
                We prioritize your health and well-being by providing accurate, 
                reliable health predictions and recommendations based on advanced AI algorithms.
              </p>
            </div>
            
            <div className="mission-card animate-fade-in-up">
              <div className="mission-icon">
                <i className="fas fa-brain"></i>
              </div>
              <h3>AI-Powered</h3>
              <p>
                Our system uses multiple machine learning models and comprehensive 
                health databases to provide the most accurate predictions possible.
              </p>
            </div>
            
            <div className="mission-card animate-fade-in-right">
              <div className="mission-icon">
                <i className="fas fa-users"></i>
              </div>
              <h3>User-Centric</h3>
              <p>
                Designed with users in mind, our platform provides intuitive, 
                easy-to-understand health insights and actionable recommendations.
              </p>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="section bg-secondary">
          <div className="section-header">
            <h2 className="section-title">Why Choose Health AI?</h2>
            <p className="section-subtitle">
              Discover the advantages of our advanced health prediction system
            </p>
          </div>
          
          <div className="features-grid">
            <div className="feature-item animate-fade-in-up delay-100">
              <div className="feature-icon">
                <i className="fas fa-shield-alt"></i>
              </div>
              <h3>Secure & Private</h3>
              <p>Your health data is protected with industry-standard security measures and privacy controls.</p>
            </div>
            
            <div className="feature-item animate-fade-in-up delay-200">
              <div className="feature-icon">
                <i className="fas fa-clock"></i>
              </div>
              <h3>Instant Results</h3>
              <p>Get health insights and personalized recommendations in seconds, not hours or days.</p>
            </div>
            
            <div className="feature-item animate-fade-in-up delay-300">
              <div className="feature-icon">
                <i className="fas fa-chart-line"></i>
              </div>
              <h3>Accurate Predictions</h3>
              <p>Our AI models are trained on extensive health data for reliable and precise predictions.</p>
            </div>
            
            <div className="feature-item animate-fade-in-up delay-400">
              <div className="feature-icon">
                <i className="fas fa-mobile-alt"></i>
              </div>
              <h3>Accessible Anywhere</h3>
              <p>Use our platform on any device, anywhere, anytime for convenient health insights.</p>
            </div>
            
            <div className="feature-item animate-fade-in-up delay-500">
              <div className="feature-icon">
                <i className="fas fa-user-md"></i>
              </div>
              <h3>Professional Guidance</h3>
              <p>Based on medical expertise and comprehensive health databases for reliable advice.</p>
            </div>
            
            <div className="feature-item animate-fade-in-up delay-600">
              <div className="feature-icon">
                <i className="fas fa-sync-alt"></i>
              </div>
              <h3>Continuous Learning</h3>
              <p>Our AI system continuously improves and learns from new data for better accuracy.</p>
            </div>
          </div>
        </section>

        {/* Technology Section */}
        <section className="section">
          <div className="section-header">
            <h2 className="section-title">Our Technology</h2>
            <p className="section-subtitle">
              Powered by cutting-edge AI and machine learning technologies
            </p>
          </div>
          
          <div className="tech-container">
            <div className="tech-card animate-fade-in-left">
              <h3>Machine Learning Models</h3>
              <ul>
                <li>Support Vector Classifier (SVC)</li>
                <li>Random Forest Classifier</li>
                <li>Logistic Regression</li>
                <li>Ensemble Learning Methods</li>
              </ul>
            </div>
            
            <div className="tech-card animate-fade-in-up">
              <h3>Data Sources</h3>
              <ul>
                <li>Comprehensive Health Databases</li>
                <li>Medical Research Publications</li>
                <li>Clinical Trial Data</li>
                <li>Expert Medical Knowledge</li>
              </ul>
            </div>
            
            <div className="tech-card animate-fade-in-right">
              <h3>AI Capabilities</h3>
              <ul>
                <li>Symptom Pattern Recognition</li>
                <li>Disease Prediction Algorithms</li>
                <li>Personalized Recommendations</li>
                <li>Real-time Analysis</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="section bg-secondary">
          <div className="section-header">
            <h2 className="section-title">Our Team</h2>
            <p className="section-subtitle">
              Dedicated professionals committed to improving healthcare through technology
            </p>
          </div>
          
          <div className="team-grid">
            <div className="team-member animate-fade-in-up delay-100">
              <div className="member-avatar">
                <i className="fas fa-user-circle"></i>
              </div>
              <h3>AI Researchers</h3>
              <p>Expert machine learning engineers developing advanced prediction algorithms</p>
            </div>
            
            <div className="team-member animate-fade-in-up delay-200">
              <div className="member-avatar">
                <i className="fas fa-user-md"></i>
              </div>
              <h3>Medical Experts</h3>
              <p>Healthcare professionals ensuring accuracy and medical validity</p>
            </div>
            
            <div className="team-member animate-fade-in-up delay-300">
              <div className="member-avatar">
                <i className="fas fa-code"></i>
              </div>
              <h3>Software Engineers</h3>
              <p>Skilled developers building robust and user-friendly platforms</p>
            </div>
            
            <div className="team-member animate-fade-in-up delay-400">
              <div className="member-avatar">
                <i className="fas fa-chart-bar"></i>
              </div>
              <h3>Data Scientists</h3>
              <p>Analytics experts processing and analyzing health data</p>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="section">
          <div className="cta-container">
            <div className="cta-card">
              <h2>Ready to Experience AI-Powered Health Insights?</h2>
              <p>Start your health journey today with our advanced prediction system</p>
              <div className="cta-buttons">
                <a href="/prediction" className="btn btn-primary">
                  Start Health Analysis
                </a>
                <a href="/blogs" className="btn btn-outline">
                  Read Our Blog
                </a>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
