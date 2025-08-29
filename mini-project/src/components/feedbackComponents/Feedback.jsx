import React, { useState } from 'react';
import './Feedback.css';

const Feedback = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
    rating: 5,
    category: 'general'
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus(null);

    try {
      // Send feedback to backend
      const response = await fetch('http://localhost:5000/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      const result = await response.json();
      
      if (result.success) {
        setSubmitStatus('success');
        setFormData({
          name: '',
          email: '',
          subject: '',
          message: '',
          rating: 5,
          category: 'general'
        });
      } else {
        setSubmitStatus('error');
      }
    } catch (error) {
      console.error('Error sending feedback:', error);
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStars = (rating) => {
    return Array.from({ length: 5 }, (_, index) => (
      <button
        key={index}
        type="button"
        className={`star-btn ${index < rating ? 'active' : ''}`}
        onClick={() => setFormData(prev => ({ ...prev, rating: index + 1 }))}
      >
        <i className="fas fa-star"></i>
      </button>
    ));
  };

  return (
    <div className="feedback-page">
      <div className="container">
        {/* Hero Section */}
        <section className="feedback-hero">
          <div className="hero-content">
            <h1 className="hero-title animate-fade-in-up">
              Share Your Feedback
            </h1>
            <p className="hero-subtitle animate-fade-in-up delay-200">
              Help us improve Health AI by sharing your thoughts, suggestions, and experiences
            </p>
          </div>
        </section>

        {/* Feedback Form Section */}
        <section className="feedback-section">
          <div className="feedback-container">
            <div className="feedback-info">
              <h2 className="section-title">We Value Your Input</h2>
              <p className="section-subtitle">
                Your feedback helps us enhance our AI-powered health prediction system and provide better user experience.
              </p>
              
              <div className="feedback-features">
                <div className="feature-item">
                  <div className="feature-icon">
                    <i className="fas fa-comments"></i>
                  </div>
                  <h3>Open Communication</h3>
                  <p>Share your thoughts and suggestions directly with our team</p>
                </div>
                
                <div className="feature-item">
                  <div className="feature-icon">
                    <i className="fas fa-rocket"></i>
                  </div>
                  <h3>Continuous Improvement</h3>
                  <p>Help us enhance our AI algorithms and user interface</p>
                </div>
                
                <div className="feature-item">
                  <div className="feature-icon">
                    <i className="fas fa-heart"></i>
                  </div>
                  <h3>User-Centric</h3>
                  <p>Your feedback shapes the future of Health AI</p>
                </div>
              </div>
            </div>

            <div className="feedback-form-container">
              <form onSubmit={handleSubmit} className="feedback-form">
                <div className="form-header">
                  <h3>Send Us Your Feedback</h3>
                  <p>We'd love to hear from you!</p>
                </div>

                <div className="form-group">
                  <label htmlFor="name">Full Name *</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter your full name"
                    className="form-input"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="email">Email Address *</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    required
                    placeholder="Enter your email address"
                    className="form-input"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="category">Feedback Category</label>
                  <select
                    id="category"
                    name="category"
                    value={formData.category}
                    onChange={handleInputChange}
                    className="form-select"
                  >
                    <option value="general">General Feedback</option>
                    <option value="bug">Bug Report</option>
                    <option value="feature">Feature Request</option>
                    <option value="improvement">Improvement Suggestion</option>
                    <option value="praise">Praise/Compliment</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="subject">Subject *</label>
                  <input
                    type="text"
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleInputChange}
                    required
                    placeholder="Brief description of your feedback"
                    className="form-input"
                  />
                </div>

                <div className="form-group">
                  <label>Rating</label>
                  <div className="rating-container">
                    {renderStars(formData.rating)}
                    <span className="rating-text">{formData.rating}/5 stars</span>
                  </div>
                </div>

                <div className="form-group">
                  <label htmlFor="message">Message *</label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleInputChange}
                    required
                    placeholder="Please share your detailed feedback, suggestions, or concerns..."
                    className="form-textarea"
                    rows="6"
                  ></textarea>
                </div>

                <div className="form-actions">
                  <button
                    type="submit"
                    className={`btn btn-primary submit-btn ${isSubmitting ? 'loading' : ''}`}
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? (
                      <>
                        <i className="fas fa-spinner fa-spin"></i>
                        Sending Feedback...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-paper-plane"></i>
                        Send Feedback
                      </>
                    )}
                  </button>
                </div>

                {submitStatus === 'success' && (
                  <div className="success-message">
                    <i className="fas fa-check-circle"></i>
                    <h4>Thank You!</h4>
                    <p>Your feedback has been sent successfully. We'll review it and get back to you soon.</p>
                  </div>
                )}

                {submitStatus === 'error' && (
                  <div className="error-message">
                    <i className="fas fa-exclamation-circle"></i>
                    <h4>Oops! Something went wrong.</h4>
                    <p>Please try again or contact us directly at shreyasssanil62@gmail.com</p>
                  </div>
                )}
              </form>
            </div>
          </div>
        </section>

        {/* Contact Info Section */}
        <section className="contact-section">
          <div className="contact-card">
            <div className="contact-info">
              <h2>Other Ways to Reach Us</h2>
              <p>Prefer to contact us directly? Here are alternative ways to get in touch.</p>
              
              <div className="contact-methods">
                <div className="contact-method">
                  <div className="contact-icon">
                    <i className="fas fa-envelope"></i>
                  </div>
                  <div className="contact-details">
                    <h3>Email</h3>
                    <p>shreyasssanil62@gmail.com</p>
                  </div>
                </div>
                
                <div className="contact-method">
                  <div className="contact-icon">
                    <i className="fas fa-clock"></i>
                  </div>
                  <div className="contact-details">
                    <h3>Response Time</h3>
                    <p>We typically respond within 24 hours</p>
                  </div>
                </div>
                
                <div className="contact-method">
                  <div className="contact-icon">
                    <i className="fas fa-shield-alt"></i>
                  </div>
                  <div className="contact-details">
                    <h3>Privacy</h3>
                    <p>Your information is kept confidential</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Feedback;
