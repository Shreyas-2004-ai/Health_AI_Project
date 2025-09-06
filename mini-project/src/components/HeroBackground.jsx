import React from 'react';

const HeroBackground = () => {
  return (
    <div className="hero-background">
      {/* Floating Medical Icons */}
      <div className="hero-bg-element hero-drift animate-hero-float" style={{ top: '20%', left: '10%', animationDelay: '0s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-float" style={{ top: '30%', left: '85%', animationDelay: '2s' }}>
        <i className="fas fa-user-md"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-float" style={{ top: '70%', left: '15%', animationDelay: '4s' }}>
        <i className="fas fa-stethoscope"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-float" style={{ top: '80%', left: '80%', animationDelay: '6s' }}>
        <i className="fas fa-dna"></i>
      </div>

      {/* Floating Particles */}
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '20%', animationDelay: '0s' }}></div>
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '40%', animationDelay: '2s' }}></div>
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '60%', animationDelay: '4s' }}></div>
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '80%', animationDelay: '6s' }}></div>
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '30%', animationDelay: '1s' }}></div>
      <div className="hero-bg-element hero-particle animate-hero-particle" style={{ left: '70%', animationDelay: '3s' }}></div>

      {/* Wave Elements */}
      <div className="hero-bg-element hero-wave animate-hero-wave" style={{ top: '25%', left: '5%', animationDelay: '1s' }}></div>
      <div className="hero-bg-element hero-wave animate-hero-wave" style={{ top: '45%', left: '75%', animationDelay: '3s' }}></div>
      <div className="hero-bg-element hero-wave animate-hero-wave" style={{ top: '65%', left: '25%', animationDelay: '5s' }}></div>
      <div className="hero-bg-element hero-wave animate-hero-wave" style={{ top: '85%', left: '55%', animationDelay: '7s' }}></div>

      {/* Glow Elements */}
      <div className="hero-bg-element hero-glow animate-hero-glow" style={{ top: '15%', left: '30%', animationDelay: '0s' }}></div>
      <div className="hero-bg-element hero-glow animate-hero-glow" style={{ top: '35%', left: '70%', animationDelay: '2s' }}></div>
      <div className="hero-bg-element hero-glow animate-hero-glow" style={{ top: '55%', left: '20%', animationDelay: '4s' }}></div>
      <div className="hero-bg-element hero-glow animate-hero-glow" style={{ top: '75%', left: '60%', animationDelay: '6s' }}></div>

      {/* Orbiting Elements */}
      <div className="hero-bg-element hero-orbit animate-hero-orbit" style={{ top: '40%', left: '50%', animationDelay: '0s' }}></div>
      <div className="hero-bg-element hero-orbit animate-hero-orbit" style={{ top: '60%', left: '30%', animationDelay: '5s' }}></div>

      {/* Pulsing Elements */}
      <div className="hero-bg-element hero-drift animate-hero-pulse" style={{ top: '10%', left: '50%', animationDelay: '1s' }}>
        <i className="fas fa-plus"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-pulse" style={{ top: '90%', left: '40%', animationDelay: '3s' }}>
        <i className="fas fa-heartbeat"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-pulse" style={{ top: '50%', left: '10%', animationDelay: '5s' }}>
        <i className="fas fa-pills"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-pulse" style={{ top: '50%', left: '90%', animationDelay: '7s' }}>
        <i className="fas fa-microscope"></i>
      </div>

      {/* Drifting Elements */}
      <div className="hero-bg-element hero-drift animate-hero-drift" style={{ top: '25%', left: '60%', animationDelay: '2s' }}>
        <i className="fas fa-hospital"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-drift" style={{ top: '75%', left: '40%', animationDelay: '4s' }}>
        <i className="fas fa-syringe"></i>
      </div>
      
      <div className="hero-bg-element hero-drift animate-hero-drift" style={{ top: '45%', left: '80%', animationDelay: '6s' }}>
        <i className="fas fa-thermometer-half"></i>
      </div>
    </div>
  );
};

export default HeroBackground;
