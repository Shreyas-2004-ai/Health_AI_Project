import React from 'react';
import './footer.css';

function Footer() {
    return (
        <div className='footer-container'>
            <div className='footer-content'>
                <div className='top-footer-content'>
                    {/* Company Info */}
                    <div className='first_footer-content'>
                        <div className='f-logo'>
                            <div className='f-title'>Health AI</div>
                        </div>
                        <p className='footer-description'>
                            Empowering healthcare with advanced AI technology. 
                            Get personalized health insights and predictions 
                            powered by cutting-edge machine learning.
                        </p>
                        <div className='social-links'>
                            <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className='social-link' aria-label="Facebook">
                                <i className="fab fa-facebook-f"></i>
                            </a>
                            <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className='social-link' aria-label="Twitter">
                                <i className="fab fa-twitter"></i>
                            </a>
                            <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer" className='social-link' aria-label="LinkedIn">
                                <i className="fab fa-linkedin-in"></i>
                            </a>
                            <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className='social-link' aria-label="Instagram">
                                <i className="fab fa-instagram"></i>
                            </a>
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div className='second_footer-content'>
                        <h3 className='footer-section-title'>Quick Links</h3>
                        <div className='footer-links'>
                            <a href="/">Home</a>
                            <a href="/about">About Us</a>
                            <a href="/prediction">Health Analysis</a>
                            <a href="/blogs">Health Blog</a>
                            <a href="/feedback">Feedback</a>
                        </div>
                    </div>

                    {/* Services */}
                    <div className='second_footer-content'>
                        <h3 className='footer-section-title'>Services</h3>
                        <div className='footer-links'>
                            <a href="/prediction">Symptom Analysis</a>
                            <a href="/prediction">Health Predictions</a>
                            <a href="/prediction">Medical Recommendations</a>
                            <a href="/blogs">Health Insights</a>
                            <a href="/feedback">Support</a>
                        </div>
                    </div>

                    {/* Contact Info */}
                    <div className='third_footer-content'>
                        <h3 className='footer-section-title'>Contact Us</h3>
                        <div className='contact-info'>
                            <div className='contact-item'>
                                <i className="fas fa-map-marker-alt contact-icon"></i>
                                <div className='contact-text'>
                                    <strong>Address:</strong><br />
                                    123 Healthcare Street,<br />
                                    Medical District,<br />
                                    Health City, HC 56789
                                </div>
                            </div>
                            
                            <div className='contact-item'>
                                <i className="fas fa-envelope contact-icon"></i>
                                <div className='contact-text'>
                                    <strong>Email:</strong><br />
                                    General: shreyasssanil62@gmail.com<br />
                                    Support: support@healthai.com
                                </div>
                            </div>
                            
                            <div className='contact-item'>
                                <i className="fas fa-phone contact-icon"></i>
                                <div className='contact-text'>
                                    <strong>Phone:</strong><br />
                                    +1 (555) 123-4567<br />
                                    Mon-Fri: 9AM-6PM
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className='bottom-footer-content'>
                    <p>&copy; 2024 Health AI. All rights reserved. | Privacy Policy | Terms of Service</p>
                </div>
            </div>
        </div>
    );
}

export default Footer;