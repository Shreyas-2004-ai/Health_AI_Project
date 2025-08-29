import react from 'react';
import './footer.css';

function Footer(){
    return(
    <div className='footer-container'>
        <div className='footer-content'>
            <div className='top-footer-content'> 
            <div className='first_footer-content'>
                <div className='f-title'>Health AI</div>
            </div>

            <div className='second_footer-content'>
                <a href="/">Home</a><br />
                <a href="/about">About Us</a><br />
                <a href="/prediction">Prediction</a><br />
                <a href="/blogs">Blogs</a><br />
                <a href="/feedback">Feedback</a><br />
            </div>

            <div className='third_footer-content'>
                <div className='Visit_Us-content'>
                    <pre>
                       <h2 className='heading1'>Visit Us: </h2>
                                  <p>
                                    Address: <br />
                                   123 XXXX, <br />
                                   YYYY City, <br /> 
                                   HC 56789
                                   </p> 
                    </pre><br />
                    
                    <pre>
                        <h2 className='heading2'>Contact Details:</h2> 
                                <p>For general inquiries: shreyasssanil62@gmail.com<br />   
                                   For technical assistance: shreyasssanil62@gmail.com <br />                                
                                </p>
                    </pre><br />
                </div>
            </div>
            </div>

            <div className='bottom-footer-content'>
                <hr />
                <p>&copy; Copyright All rights reserved </p>
            </div>

        </div>
    </div>
    );
}

export default Footer;