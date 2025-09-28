import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth"; // Importing getAuth for authentication
import { getFirestore } from "firebase/firestore"; // Importing getFirestore for Firestore

// Firebase configuration for login
const firebaseConfigLogin = {
  apiKey: "AIzaSyC04mDbrEVWtQBTDf5_6IIOIVeGZ5EKsdc",
  authDomain: "login-auth-62fc7.firebaseapp.com",
  projectId: "login-auth-62fc7",
  storageBucket: "login-auth-62fc7.appspot.com",
  messagingSenderId: "959979678292",
  appId: "1:959979678292:web:c642d52c7587d50de05bc1"
};

// Firebase configuration for feedback
const firebaseConfigFeedback = {
  apiKey: "AIzaSyBh8-Z1hXqoxLCpmZ45HfakEBdhD0vOiy8",
  authDomain: "feedback-auth-953e7.firebaseapp.com",
  projectId: "feedback-auth-953e7",
  storageBucket: "feedback-auth-953e7.appspot.com",
  messagingSenderId: "304012968483",
  appId: "1:304012968483:web:ddb17ac3a93f840d27a796"
};

// Initialize apps
const app = initializeApp(firebaseConfigLogin, "loginApp");
const feedbackApp = initializeApp(firebaseConfigFeedback, "feedbackApp");

// Initialize Firestore for feedback
export const feedbackDb = getFirestore(feedbackApp);

// Export authentication for login (auth needed for login)
export const auth = getAuth(app);
