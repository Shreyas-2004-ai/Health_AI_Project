import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Footer from '../footerComponents/Footer';
import './PredictionPage.css';

const PredictionPage = () => {
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';
  const [symptoms, setSymptoms] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [symptomSuggestions, setSymptomSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [currentInput, setCurrentInput] = useState('');
  // Removed unused selectedSymptoms state to satisfy linter
  const [lastFormattedSymptoms, setLastFormattedSymptoms] = useState([]);
  const [sendingFeedback, setSendingFeedback] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const inputRef = useRef(null);

  // Complete list of symptoms from the backend system
  const allSymptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 
    'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 
    'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 
    'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
    'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 
    'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 
    'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 
    'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
    'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
    'red_sore_around_nose', 'yellow_crust_ooze', 'sore_throat', 'body_aches'
  ];
  
  // Symptom synonyms for better matching
  const symptomSynonyms = {
    'cough': ['cough', 'hacking', 'dry cough', 'wet cough', 'persistent cough'],
    'fever': ['fever', 'high fever', 'mild fever', 'temperature', 'pyrexia'],
    'chills': ['chills', 'shivering', 'cold flashes', 'rigors'],
    'headache': ['headache', 'migraine', 'head pain', 'tension headache', 'severe headache'],
    'nausea': ['nausea', 'queasiness', 'upset stomach', 'sick feeling'],
    'fatigue': ['fatigue', 'tiredness', 'exhaustion', 'lethargy', 'weakness'],
    'abdominal_pain': ['abdominal pain', 'stomach pain', 'belly pain', 'gut pain', 'severe abdominal pain'],
    'diarrhoea': ['diarrhoea', 'diarrhea', 'loose motions', 'runny stools'],
    'vomiting': ['vomiting', 'throwing up', 'emesis'],
    'itching': ['itching', 'pruritus', 'skin irritation'],
    'skin_rash': ['skin rash', 'dermatitis', 'eruption'],
    'chest_pain': ['chest pain', 'thoracic pain', 'angina', 'chest tightness'],
    'shortness_of_breath': ['shortness of breath', 'breathlessness', 'dyspnea', 'difficulty breathing'],
    'dizziness': ['dizziness', 'vertigo', 'lightheadedness'],
    'anxiety': ['anxiety', 'nervousness', 'stress', 'worry'],
    'burning_micturition': ['burning micturition', 'painful urination', 'dysuria'],
    'sweating': ['sweating', 'perspiration', 'excessive sweating'],
    'body_aches': ['body aches', 'muscle pain', 'joint pain', 'general pain'],
    'loss_of_appetite': ['loss of appetite', 'decreased hunger', 'anorexia'],
    'runny_nose': ['runny nose', 'nasal discharge', 'rhinorrhea'],
    'sneezing': ['sneezing', 'continuous sneezing'],
    'sore_throat': ['sore throat', 'throat pain', 'pharyngitis'],
    'congestion': ['congestion', 'nasal congestion', 'stuffy nose']
  };

  // Extract the current input from the comma-separated string
  useEffect(() => {
    if (symptoms) {
      const parts = symptoms.split(',');
      const lastPart = parts[parts.length - 1].trim();
      setCurrentInput(lastPart);
    } else {
      setCurrentInput('');
    }
  }, [symptoms]);

  // Handle input changes
  const handleInputChange = (e) => {
    const value = e.target.value;
    setSymptoms(value);
    setError('');

    // Extract the current input for suggestions
    const parts = value.split(',');
    const currentTyping = parts[parts.length - 1].trim().toLowerCase();

    if (currentTyping) {
      // Match against symptoms and their synonyms
      const matches = [];
      
      // First check direct matches in the main symptom list
      allSymptoms.forEach(symptom => {
        const displaySymptom = symptom.replace(/_/g, ' ');
        if (displaySymptom.toLowerCase().includes(currentTyping)) {
          matches.push({
            value: displaySymptom,
            originalValue: symptom,
            matchType: 'direct'
          });
        }
      });
      
      // Then check synonym matches
      Object.entries(symptomSynonyms).forEach(([symptom, synonymList]) => {
        const displaySymptom = symptom.replace(/_/g, ' ');
        
        // Skip if already added as direct match
        if (!matches.some(m => m.originalValue === symptom)) {
          synonymList.forEach(synonym => {
            if (synonym.toLowerCase().includes(currentTyping)) {
              matches.push({
                value: displaySymptom,
                originalValue: symptom,
                matchType: 'synonym',
                matchedSynonym: synonym
              });
            }
          });
        }
      });
      
      // Sort matches: direct matches first, then by length (shorter first)
      const sortedMatches = matches
        .sort((a, b) => {
          // First by match type (direct before synonym)
          if (a.matchType !== b.matchType) {
            return a.matchType === 'direct' ? -1 : 1;
          }
          // Then by whether it starts with the input
          const aStartsWith = a.value.toLowerCase().startsWith(currentTyping);
          const bStartsWith = b.value.toLowerCase().startsWith(currentTyping);
          if (aStartsWith !== bStartsWith) {
            return aStartsWith ? -1 : 1;
          }
          // Then by length
          return a.value.length - b.value.length;
        })
        .slice(0, 5) // Limit to 5 suggestions
        .map(match => match.value); // Just return the display value
      
      // Remove duplicates
      const uniqueMatches = [...new Set(sortedMatches)];
      
      setSymptomSuggestions(uniqueMatches);
      setShowSuggestions(uniqueMatches.length > 0);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleSymptomSuggestion = (suggestion) => {
    const currentSymptoms = symptoms.split(',').map(s => s.trim()).filter(s => s);
    // Remove the current input (last item) if it's not complete
    const completedSymptoms = currentInput ? currentSymptoms.slice(0, -1) : currentSymptoms;
    
    // Check if the suggestion is already in the list
    if (!completedSymptoms.some(s => s.toLowerCase() === suggestion.toLowerCase())) {
      // Format the new symptoms string
      const newSymptoms = completedSymptoms.length > 0 
        ? `${completedSymptoms.join(', ')}, ${suggestion}`
        : suggestion;
      
      setSymptoms(`${newSymptoms}, `); // Add trailing comma and space for next input
      
      // Focus the input field after selection
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 0);
    } else {
      // If already selected, just add the comma for the next input
      setSymptoms(`${symptoms.trim()}, `);
    }
    
    setShowSuggestions(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!symptoms.trim()) {
      setError('Please enter at least one symptom');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      // Process symptoms to ensure they match the backend format
      const symptomList = symptoms.split(',').map(s => s.trim()).filter(s => s);
      
      // Convert display symptoms to backend format (replace spaces with underscores)
      const formattedSymptoms = symptomList.map(symptom => {
        // Check if this is a direct match in allSymptoms
        const directMatch = allSymptoms.find(s => 
          s.toLowerCase() === symptom.toLowerCase() || 
          s.replace(/_/g, ' ').toLowerCase() === symptom.toLowerCase()
        );
        
        if (directMatch) {
          return directMatch; // Use the exact format from allSymptoms
        }
        
        // Check if it matches any synonym
        for (const [originalSymptom, synonyms] of Object.entries(symptomSynonyms)) {
          if (synonyms.some(syn => syn.toLowerCase() === symptom.toLowerCase())) {
            return originalSymptom; // Return the original symptom key
          }
        }
        
        // If no match found, use the input with underscores
        return symptom.replace(/ /g, '_').toLowerCase();
      });
      
      const response = await axios.post(`${API_BASE}/api/predict`, {
        symptoms: formattedSymptoms
      });

      setResult(response.data);
      setLastFormattedSymptoms(formattedSymptoms);
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Unable to connect to the prediction service. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    setSymptoms('');
    setResult(null);
    setError('');
    setShowSuggestions(false);
    setLastFormattedSymptoms([]);
  };

  const sendFeedback = async () => {
    if (!result || !result.disease || lastFormattedSymptoms.length === 0) return;
    try {
      setSendingFeedback(true);
      await axios.post(`${API_BASE}/api/learn`, {
        symptoms: lastFormattedSymptoms,
        confirmed_disease: result.disease
      });
      alert('Thanks! Your feedback will help improve the model.');
    } catch (e) {
      console.error('Feedback error:', e);
      alert('Unable to send feedback right now.');
    } finally {
      setSendingFeedback(false);
    }
  };

  const retrainModel = async () => {
    try {
      setRetraining(true);
      const res = await axios.post(`${API_BASE}/api/retrain`, {});
      if (res.data && res.data.success) {
        alert('Model retrained successfully. New classes loaded.');
      } else {
        alert('Retrain failed. Check server logs.');
      }
    } catch (e) {
      console.error('Retrain error:', e);
      alert('Unable to retrain model right now.');
    } finally {
      setRetraining(false);
    }
  };

  return (
    <div className="prediction-page">
      <div className="container">
        {/* Header */}
        <div className="prediction-header">
          <h1 className="prediction-title">Health AI Analysis</h1>
          <p className="prediction-subtitle">
            Enter your symptoms below to discover tailored health insights
          </p>
        </div>

        {/* Input Section */}
        <div className="input-section">
          <form onSubmit={handleSubmit} className="prediction-form">
            <div className="input-container">
              <input
                type="text"
                value={symptoms}
                onChange={handleInputChange}
                placeholder="Enter symptoms (e.g., fever, cough, headache)"
                className="symptom-input"
                disabled={loading}
                ref={inputRef}
                autoComplete="off"
              />
              
              {showSuggestions && symptomSuggestions.length > 0 && (
                <div className="suggestions-container">
                  {symptomSuggestions.map((suggestion, index) => (
                    <div
                      key={index}
                      className="suggestion-item"
                      onClick={() => handleSymptomSuggestion(suggestion)}
                    >
                      {suggestion}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="button-container">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !symptoms.trim()}
              >
                {loading ? (
                  <>
                    <div className="spinner"></div>
                    Analyzing...
                  </>
                ) : (
                  'Get Health Insights'
                )}
              </button>
              
              {result && (
                <button
                  type="button"
                  onClick={clearForm}
                  className="btn btn-outline"
                >
                  New Analysis
                </button>
              )}
              {result && (
                <button
                  type="button"
                  onClick={sendFeedback}
                  className="btn btn-secondary"
                  disabled={sendingFeedback}
                  title="Mark this prediction as correct"
                >
                  {sendingFeedback ? 'Sending...' : 'Looks correct'}
                </button>
              )}
              <button
                type="button"
                onClick={retrainModel}
                className="btn btn-warning"
                disabled={retraining}
                title="Retrain model with collected feedback"
              >
                {retraining ? 'Retraining...' : 'Retrain model'}
              </button>
            </div>
          </form>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Analyzing your symptoms with AI...</p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error-container">
            <div className="error">
              <i className="fas fa-exclamation-triangle"></i>
              <span>{error}</span>
            </div>
            <button onClick={clearForm} className="btn btn-outline">
              Try Again
            </button>
          </div>
        )}

        {/* Results Display */}
        {result && !loading && (
          <div className="prediction-result animate-fade-in-up">
            <h2 className="result-title">Here Are Your Insights!</h2>
            
            <div className="result-card">
              <div className="disease-info">
                <h3 className="disease-name">{result.disease}</h3>
                <p className="disease-description">{result.description}</p>
              </div>

              <div className="recommendations-grid">
                <div className="recommendation-card">
                  <div className="recommendation-icon">
                    <i className="fas fa-shield-alt"></i>
                  </div>
                  <h4>Precautions</h4>
                  <ul>
                    {result.precautions.map((precaution, index) => (
                      <li key={index}>{precaution}</li>
                    ))}
                  </ul>
                </div>

                <div className="recommendation-card">
                  <div className="recommendation-icon">
                    <i className="fas fa-pills"></i>
                  </div>
                  <h4>Medications</h4>
                  <ul>
                    {result.medications.map((medication, index) => (
                      <li key={index}>{medication}</li>
                    ))}
                  </ul>
                </div>

                <div className="recommendation-card">
                  <div className="recommendation-icon">
                    <i className="fas fa-apple-alt"></i>
                  </div>
                  <h4>Dietary Suggestions</h4>
                  <ul>
                    {result.diets.map((diet, index) => (
                      <li key={index}>{diet}</li>
                    ))}
                  </ul>
                </div>

                <div className="recommendation-card">
                  <div className="recommendation-icon">
                    <i className="fas fa-dumbbell"></i>
                  </div>
                  <h4>Workout Suggestions</h4>
                  <ul>
                    {result.workouts.map((workout, index) => (
                      <li key={index}>{workout}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="result-footer">
                <p className="confidence-note">
                  <i className="fas fa-info-circle"></i>
                  Prediction confidence: {result.prediction_method === 'improved_ml' ? 'High' : 'Medium'}
                </p>
                <p className="medical-disclaimer">
                  * This is an AI-powered prediction based on the symptoms you provided. 
                  Please consult with a healthcare professional for accurate diagnosis and treatment.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
      <Footer />
    </div>
  );
};

export default PredictionPage;
