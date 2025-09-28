import React from 'react';

const MedicalBackground = () => {
  return (
    <div className="medical-background">
      {/* Test Element - Should be visible */}
      <div style={{ 
        position: 'fixed', 
        top: '50px', 
        left: '50px', 
        fontSize: '3rem', 
        color: 'red', 
        zIndex: '1000',
        animation: 'simplePulse 2s ease-in-out infinite'
      }}>
        <i className="fas fa-dna"></i>
      </div>
      {/* Moving Ambulances */}
      <div className="medical-bg-element ambulance animate-ambulance" style={{ top: '20%', left: '0%' }}>
        <i className="fas fa-ambulance"></i>
      </div>
      
      <div className="medical-bg-element ambulance animate-ambulance-reverse" style={{ top: '60%', animationDelay: '5s' }}>
        <i className="fas fa-ambulance"></i>
      </div>
      
      <div className="medical-bg-element ambulance animate-ambulance" style={{ top: '40%', animationDelay: '10s' }}>
        <i className="fas fa-ambulance"></i>
      </div>

      {/* Floating Medical Icons */}
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '15%', left: '10%', animationDelay: '1s' }}>
        <i className="fas fa-user-md"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '25%', left: '85%', animationDelay: '3s' }}>
        <i className="fas fa-hospital"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '70%', left: '15%', animationDelay: '5s' }}>
        <i className="fas fa-stethoscope"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '80%', left: '80%', animationDelay: '7s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>

      {/* Heartbeat Icons */}
      <div className="medical-bg-element heartbeat-icon animate-heartbeat" style={{ top: '30%', left: '20%', animationDelay: '0.5s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="medical-bg-element heartbeat-icon animate-heartbeat" style={{ top: '50%', left: '70%', animationDelay: '1.5s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="medical-bg-element heartbeat-icon animate-heartbeat" style={{ top: '75%', left: '40%', animationDelay: '2.5s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>

      {/* DNA Helix - More Prominent */}
      <div className="medical-bg-element dna-icon animate-dna-helix" style={{ top: '30%', left: '20%', animationDelay: '0s' }}>
        <i className="fas fa-dna"></i>
      </div>
      
      <div className="medical-bg-element dna-icon animate-dna-helix" style={{ top: '50%', left: '70%', animationDelay: '3s' }}>
        <i className="fas fa-dna"></i>
      </div>
      
      <div className="medical-bg-element dna-icon animate-dna-helix" style={{ top: '70%', left: '40%', animationDelay: '6s' }}>
        <i className="fas fa-dna"></i>
      </div>
      
      <div className="medical-bg-element dna-icon animate-dna-helix" style={{ top: '20%', left: '80%', animationDelay: '9s' }}>
        <i className="fas fa-dna"></i>
      </div>

      {/* Medical Cross */}
      <div className="medical-bg-element cross-icon animate-medical-cross" style={{ top: '45%', left: '25%', animationDelay: '1s' }}>
        <i className="fas fa-plus"></i>
      </div>
      
      <div className="medical-bg-element cross-icon animate-medical-cross" style={{ top: '55%', left: '75%', animationDelay: '4s' }}>
        <i className="fas fa-plus"></i>
      </div>

      {/* Stethoscope */}
      <div className="medical-bg-element stethoscope-icon animate-stethoscope" style={{ top: '20%', left: '60%', animationDelay: '3s' }}>
        <i className="fas fa-stethoscope"></i>
      </div>
      
      <div className="medical-bg-element stethoscope-icon animate-stethoscope" style={{ top: '85%', left: '50%', animationDelay: '6s' }}>
        <i className="fas fa-stethoscope"></i>
      </div>

      {/* Pill Bottle */}
      <div className="medical-bg-element pill-icon animate-pill-bottle" style={{ top: '40%', left: '5%', animationDelay: '2s' }}>
        <i className="fas fa-pills"></i>
      </div>
      
      <div className="medical-bg-element pill-icon animate-pill-bottle" style={{ top: '60%', left: '90%', animationDelay: '5s' }}>
        <i className="fas fa-pills"></i>
      </div>

      {/* Heartbeat Lines */}
      <div className="medical-bg-element heartbeat-line animate-heartbeat-line" style={{ top: '25%', left: '40%', animationDelay: '0.5s' }}></div>
      
      <div className="medical-bg-element heartbeat-line animate-heartbeat-line" style={{ top: '45%', left: '60%', animationDelay: '1s' }}></div>
      
      <div className="medical-bg-element heartbeat-line animate-heartbeat-line" style={{ top: '65%', left: '20%', animationDelay: '1.5s' }}></div>
      
      <div className="medical-bg-element heartbeat-line animate-heartbeat-line" style={{ top: '85%', left: '80%', animationDelay: '2s' }}></div>

      {/* Pulse Circles */}
      <div className="medical-bg-element pulse-circle animate-pulse-wave" style={{ top: '30%', left: '35%', animationDelay: '1s' }}></div>
      
      <div className="medical-bg-element pulse-circle animate-pulse-wave" style={{ top: '50%', left: '65%', animationDelay: '2s' }}></div>
      
      <div className="medical-bg-element pulse-circle animate-pulse-wave" style={{ top: '70%', left: '25%', animationDelay: '3s' }}></div>
      
      <div className="medical-bg-element pulse-circle animate-pulse-wave" style={{ top: '90%', left: '75%', animationDelay: '4s' }}></div>

      {/* Additional Medical Icons */}
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '10%', left: '50%', animationDelay: '4s' }}>
        <i className="fas fa-microscope"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '90%', left: '10%', animationDelay: '6s' }}>
        <i className="fas fa-syringe"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '35%', left: '90%', animationDelay: '8s' }}>
        <i className="fas fa-x-ray"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-medical-icon" style={{ top: '55%', left: '5%', animationDelay: '9s' }}>
        <i className="fas fa-thermometer-half"></i>
      </div>

      {/* More Heartbeat Icons */}
      <div className="medical-bg-element heartbeat-icon animate-heartbeat" style={{ top: '5%', left: '25%', animationDelay: '3.5s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="medical-bg-element heartbeat-icon animate-heartbeat" style={{ top: '95%', left: '60%', animationDelay: '4.5s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>

      {/* Simple Moving Elements for Visibility */}
      <div className="medical-bg-element medical-icon animate-simple-float" style={{ top: '10%', left: '30%', animationDelay: '1s', opacity: '0.7' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-simple-pulse" style={{ top: '80%', left: '70%', animationDelay: '2s', opacity: '0.7' }}>
        <i className="fas fa-user-md"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-simple-float" style={{ top: '40%', left: '10%', animationDelay: '3s', opacity: '0.7' }}>
        <i className="fas fa-stethoscope"></i>
      </div>
      
      <div className="medical-bg-element medical-icon animate-simple-pulse" style={{ top: '60%', left: '90%', animationDelay: '4s', opacity: '0.7' }}>
        <i className="fas fa-hospital"></i>
      </div>
      
      {/* More DNA Elements with Simple Animations */}
      <div className="medical-bg-element dna-icon animate-simple-float" style={{ top: '15%', left: '60%', animationDelay: '2s', opacity: '0.8' }}>
        <i className="fas fa-dna"></i>
      </div>
      
      <div className="medical-bg-element dna-icon animate-simple-pulse" style={{ top: '85%', left: '20%', animationDelay: '3s', opacity: '0.8' }}>
        <i className="fas fa-dna"></i>
      </div>
    </div>
  );
};

export default MedicalBackground;
