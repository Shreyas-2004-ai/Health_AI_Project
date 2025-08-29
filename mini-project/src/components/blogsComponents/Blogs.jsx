import React from 'react';
import './Blogs.css';

const Blogs = () => {
  const blogPosts = [
    {
      id: 1,
      title: "The Future of AI in Healthcare",
      excerpt: "Discover how artificial intelligence is revolutionizing the healthcare industry and improving patient outcomes.",
      category: "AI & Healthcare",
      readTime: "5 min read",
      image: "/blog-ai-healthcare.jpg",
      date: "2024-01-15"
    },
    {
      id: 2,
      title: "Understanding Symptom-Based Diagnosis",
      excerpt: "Learn how modern AI systems analyze symptoms to provide accurate health predictions and recommendations.",
      category: "Diagnosis",
      readTime: "4 min read",
      image: "/blog-symptoms.jpg",
      date: "2024-01-12"
    },
    {
      id: 3,
      title: "Preventive Healthcare with AI",
      excerpt: "Explore how AI-powered tools can help prevent diseases and maintain better health through early detection.",
      category: "Prevention",
      readTime: "6 min read",
      image: "/blog-prevention.jpg",
      date: "2024-01-10"
    },
    {
      id: 4,
      title: "Machine Learning in Medical Diagnosis",
      excerpt: "Deep dive into the machine learning algorithms that power modern medical diagnosis systems.",
      category: "Technology",
      readTime: "7 min read",
      image: "/blog-ml-diagnosis.jpg",
      date: "2024-01-08"
    },
    {
      id: 5,
      title: "Digital Health Trends 2024",
      excerpt: "Stay updated with the latest trends in digital health and telemedicine technologies.",
      category: "Trends",
      readTime: "5 min read",
      image: "/blog-trends.jpg",
      date: "2024-01-05"
    },
    {
      id: 6,
      title: "AI Ethics in Healthcare",
      excerpt: "Understanding the ethical considerations and responsible use of AI in healthcare applications.",
      category: "Ethics",
      readTime: "8 min read",
      image: "/blog-ethics.jpg",
      date: "2024-01-03"
    }
  ];

  const categories = ["All", "AI & Healthcare", "Diagnosis", "Prevention", "Technology", "Trends", "Ethics"];

  return (
    <div className="blogs-page">
      <div className="container">
        {/* Hero Section */}
        <section className="blogs-hero">
          <div className="hero-content">
            <h1 className="hero-title animate-fade-in-up">
              Health AI Blog
            </h1>
            <p className="hero-subtitle animate-fade-in-up delay-200">
              Stay informed about the latest developments in AI-powered healthcare and wellness
            </p>
          </div>
        </section>

        {/* Categories Filter */}
        <section className="categories-section">
          <div className="categories-container">
            {categories.map((category, index) => (
              <button
                key={category}
                className={`category-btn ${category === 'All' ? 'active' : ''}`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {category}
              </button>
            ))}
          </div>
        </section>

        {/* Featured Post */}
        <section className="featured-section">
          <div className="featured-post animate-fade-in-up">
            <div className="featured-image">
              <div className="image-placeholder">
                <i className="fas fa-image"></i>
              </div>
              <div className="featured-badge">Featured</div>
            </div>
            <div className="featured-content">
              <div className="post-meta">
                <span className="category">AI & Healthcare</span>
                <span className="date">January 15, 2024</span>
                <span className="read-time">5 min read</span>
              </div>
              <h2 className="featured-title">
                The Future of AI in Healthcare: Transforming Patient Care
              </h2>
              <p className="featured-excerpt">
                Artificial intelligence is revolutionizing the healthcare industry, from diagnosis to treatment planning. 
                Discover how AI technologies are improving patient outcomes and transforming the way we approach healthcare.
              </p>
              <button className="btn btn-primary">Read Full Article</button>
            </div>
          </div>
        </section>

        {/* Blog Posts Grid */}
        <section className="blogs-section">
          <div className="section-header">
            <h2 className="section-title">Latest Articles</h2>
            <p className="section-subtitle">
              Explore our collection of informative articles about AI, healthcare, and wellness
            </p>
          </div>
          
          <div className="blogs-grid">
            {blogPosts.map((post, index) => (
              <article key={post.id} className="blog-card animate-fade-in-up" style={{ animationDelay: `${index * 0.1}s` }}>
                <div className="blog-image">
                  <div className="image-placeholder">
                    <i className="fas fa-image"></i>
                  </div>
                  <div className="category-badge">{post.category}</div>
                </div>
                
                <div className="blog-content">
                  <div className="post-meta">
                    <span className="date">{new Date(post.date).toLocaleDateString()}</span>
                    <span className="read-time">{post.readTime}</span>
                  </div>
                  
                  <h3 className="blog-title">{post.title}</h3>
                  <p className="blog-excerpt">{post.excerpt}</p>
                  
                  <button className="read-more-btn">
                    Read More
                    <i className="fas fa-arrow-right"></i>
                  </button>
                </div>
              </article>
            ))}
          </div>
        </section>

        {/* Newsletter Section */}
        <section className="newsletter-section">
          <div className="newsletter-card">
            <div className="newsletter-content">
              <h2>Stay Updated</h2>
              <p>Subscribe to our newsletter for the latest insights on AI healthcare and wellness tips</p>
              
              <div className="newsletter-form">
                <input
                  type="email"
                  placeholder="Enter your email address"
                  className="newsletter-input"
                />
                <button className="btn btn-primary">Subscribe</button>
              </div>
            </div>
            
            <div className="newsletter-icon">
              <i className="fas fa-envelope"></i>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="cta-section">
          <div className="cta-card">
            <h2>Ready to Experience AI-Powered Health Insights?</h2>
            <p>Start your health journey today with our advanced prediction system</p>
            <div className="cta-buttons">
              <a href="/prediction" className="btn btn-primary">
                Start Health Analysis
              </a>
              <a href="/about" className="btn btn-outline">
                Learn More About Us
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Blogs;
