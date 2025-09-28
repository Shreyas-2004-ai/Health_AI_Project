import React, { useState } from "react";
import './sym.css';

const symptomsWithSynonyms = [
  { symptom: 'abdominal_pain', synonyms: ['stomach ache', 'belly pain', 'gut pain', 'stomach cramps'] },
  { symptom: 'acidity', synonyms: ['heartburn', 'acid reflux', 'indigestion', 'sour stomach'] },
  { symptom: 'altered_sensorium', synonyms: ['confusion', 'altered consciousness', 'disorientation', 'impaired awareness'] },
  { symptom: 'anxiety', synonyms: ['nervousness', 'stress', 'worry', 'unease'] },
  { symptom: 'back_pain', synonyms: ['lumbar pain', 'spine pain', 'lower back pain', 'backache'] },
  { symptom: 'blackheads', synonyms: ['clogged pores', 'open comedones', 'acne', 'pimples'] },
  { symptom: 'bladder_discomfort', synonyms: ['urinary discomfort', 'bladder irritation', 'pelvic pain', 'urinary urgency'] },
  { symptom: 'blister', synonyms: ['sore', 'bubble', 'welt', 'fluid-filled bump'] },
  { symptom: 'bloody_stool', synonyms: ['rectal bleeding', 'blood in stool', 'hematochezia'] },
  { symptom: 'blurred_and_distorted_vision', synonyms: ['unclear vision', 'distorted sight', 'hazy vision', 'impaired sight'] },
  { symptom: 'breathlessness', synonyms: ['shortness of breath', 'dyspnea', 'difficulty breathing', 'air hunger'] },
  { symptom: 'bruising', synonyms: ['contusion', 'black-and-blue marks', 'ecchymosis'] },
  { symptom: 'burning_micturition', synonyms: ['painful urination', 'dysuria', 'burning while peeing'] },
  { symptom: 'chest_pain', synonyms: ['thoracic pain', 'tight chest', 'angina', 'chest discomfort'] },
  { symptom: 'chills', synonyms: ['shivering', 'cold flashes', 'rigors'] },
  { symptom: 'cold_hands_and_feets', synonyms: ['cold extremities', 'chilled fingers', 'icy feet', 'frozen hands'] },
  { symptom: 'constipation', synonyms: ['difficulty in passing stools', 'hard stools', 'bowel blockage'] },
  { symptom: 'continuous_feel_of_urine', synonyms: ['frequent urination', 'urge to urinate', 'bladder pressure'] },
  { symptom: 'continuous_sneezing', synonyms: ['persistent sneezing', 'repeated sneezing', 'uncontrollable sneezes'] },
  { symptom: 'cough', synonyms: ['hack', 'throat clearing', 'dry cough', 'wet cough'] },
  { symptom: 'cramps', synonyms: ['muscle spasms', 'stomach cramps', 'leg cramps', 'abdominal cramps'] },
  { symptom: 'dark_urine', synonyms: ['concentrated urine', 'brownish urine', 'amber-colored urine'] },
  { symptom: 'dehydration', synonyms: ['lack of hydration', 'fluid loss', 'dryness'] },
  { symptom: 'diarrhoea', synonyms: ['loose motions', 'frequent stools', 'runny stools'] },
  { symptom: 'dischromic_patches', synonyms: ['skin discoloration', 'pigmented patches', 'uneven skin tone'] },
  { symptom: 'distention_of_abdomen', synonyms: ['bloating', 'abdominal swelling', 'distended stomach'] },
  { symptom: 'dizziness', synonyms: ['vertigo', 'lightheadedness', 'faintness'] },
  { symptom: 'excessive_hunger', synonyms: ['polyphagia', 'overeating', 'increased appetite'] },
  { symptom: 'extra_marital_contacts', synonyms: ['high-risk behavior', 'unfaithfulness', 'extramarital affairs'] },
  { symptom: 'family_history', synonyms: ['genetic predisposition', 'hereditary risk', 'family medical background'] },
  { symptom: 'fatigue', synonyms: ['tiredness', 'exhaustion', 'lack of energy'] },
  { symptom: 'foul_smell_of_urine', synonyms: ['strong urine odor', 'bad-smelling urine', 'pungent urine'] },
  { symptom: 'headache', synonyms: ['migraine', 'head pain', 'tension headache'] },
  { symptom: 'high_fever', synonyms: ['hyperpyrexia', 'elevated temperature', 'severe fever'] },
  { symptom: 'hip_joint_pain', synonyms: ['pelvic pain', 'hip discomfort', 'hip ache'] },
  { symptom: 'indigestion', synonyms: ['dyspepsia', 'upset stomach', 'digestive discomfort'] },
  { symptom: 'irregular_sugar_level', synonyms: ['unstable glucose', 'fluctuating blood sugar'] },
  { symptom: 'irritation_in_anus', synonyms: ['anal itching', 'anal discomfort', 'pruritus ani'] },
  { symptom: 'itching', synonyms: ['pruritus', 'skin irritation', 'scratchiness'] },
  { symptom: 'joint_pain', synonyms: ['arthralgia', 'joint ache', 'stiff joints'] },
  { symptom: 'knee_pain', synonyms: ['knee discomfort', 'knee ache', 'patellar pain'] },
  { symptom: 'lack_of_concentration', synonyms: ['poor focus', 'difficulty concentrating', 'inattention'] },
  { symptom: 'lethargy', synonyms: ['sluggishness', 'lack of energy', 'weariness'] },
  { symptom: 'loss_of_appetite', synonyms: ['reduced hunger', 'decreased appetite', 'anorexia'] },
  { symptom: 'loss_of_balance', synonyms: ['unsteadiness', 'instability', 'lack of equilibrium'] },
  { symptom: 'mood_swings', synonyms: ['emotional instability', 'rapid mood changes', 'emotional fluctuation'] },
  { symptom: 'movement_stiffness', synonyms: ['rigidity', 'stiff movements', 'reduced flexibility'] },
  { symptom: 'muscle_wasting', synonyms: ['muscle atrophy', 'loss of muscle mass', 'muscle shrinkage'] },
  { symptom: 'muscle_weakness', synonyms: ['reduced strength', 'muscle fatigue', 'weak muscles'] },
  { symptom: 'nausea', synonyms: ['queasiness', 'upset stomach', 'sick feeling'] },
  { symptom: 'neck_pain', synonyms: ['cervical pain', 'stiff neck', 'neck ache'] },
  { symptom: 'nodal_skin_eruptions', synonyms: ['skin nodules', 'eruptive skin lesions', 'bumpy rash'] },
  { symptom: 'obesity', synonyms: ['overweight', 'excess body fat', 'high BMI'] },
  { symptom: 'pain_during_bowel_movements', synonyms: ['defecation pain', 'anal pain during bowel movements', 'straining pain'] },
  { symptom: 'pain_in_anal_region', synonyms: ['anal pain', 'rectal discomfort', 'anus pain'] },
  { symptom: 'painful_walking', synonyms: ['difficulty walking', 'pain while walking', 'gait pain'] },
  { symptom: 'passage_of_gases', synonyms: ['flatulence', 'passing wind', 'intestinal gas'] },
  { symptom: 'patches_in_throat', synonyms: ['throat spots', 'pharyngeal patches', 'white patches in throat'] },
  { symptom: 'pus_filled_pimples', synonyms: ['acne pustules', 'infected pimples', 'pus-filled zits'] },
  { symptom: 'red_sore_around_nose', synonyms: ['nasal redness', 'nasal sores', 'redness near nostrils'] },
  { symptom: 'restlessness', synonyms: ['agitation', 'uneasiness', 'inability to relax'] },
  { symptom: 'shivering', synonyms: ['trembling', 'quivering', 'cold shivers'] },
  { symptom: 'skin_peeling', synonyms: ['skin flaking', 'desquamation', 'epidermal shedding'] },
  { symptom: 'skin_rash', synonyms: ['dermatitis', 'eruption', 'skin inflammation'] },
  { symptom: 'stiff_neck', synonyms: ['neck rigidity', 'neck stiffness', 'reduced neck mobility'] },
  { symptom: 'stomach_pain', synonyms: ['abdominal pain', 'tummy ache', 'gut discomfort'] },
  { symptom: 'sweating', synonyms: ['perspiration', 'excessive sweating', 'sweatiness'] },
  { symptom: 'vomiting', synonyms: ['emesis', 'throwing up', 'nauseous expulsion'] },
  { symptom: 'weight_gain', synonyms: ['increased weight', 'body mass gain', 'weight increase'] },
  { symptom: 'weight_loss', synonyms: ['unintended weight reduction', 'slimming', 'loss of body mass'] },
  { symptom: 'yellowing_of_eyes', synonyms: ['jaundiced eyes', 'scleral icterus', 'yellow eye whites'] },
  { symptom: 'yellowish_skin', synonyms: ['jaundice', 'yellow complexion', 'yellowed skin'] }
];

const SymptomsList = () => {
  const [searchTerm, setSearchTerm] = useState("");

  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  const filteredSymptoms = symptomsWithSynonyms.filter(({ symptom, synonyms }) => {
    return (
      symptom.toLowerCase().includes(searchTerm.toLowerCase()) ||
      synonyms.some(synonym => synonym.toLowerCase().includes(searchTerm.toLowerCase()))
    );
  });

  return (
    <div className="app-body">
      {/* Main container */}
      <div className="symptoms-container">
        <h1>Symptoms and Their Synonyms</h1>
        
        {/* Instruction for users */}
        <div className="instruction">
          <p>
            <strong>Note:</strong> Please enter the symptoms in the homepage exactly as listed.
            The synonyms below are provided for reference to help with search results.
          </p>
        </div>
        
        {/* Search Bar */}
        <input
          type="text"
          placeholder="Search Symptoms"
          value={searchTerm}
          onChange={handleSearchChange}
          className="search-bar"
        />
        
        {/* Symptoms List */}
        {filteredSymptoms.map(({ symptom, synonyms }) => (
          <div key={symptom} className="symptom-card">
            <h2>{symptom}</h2>
            <ul>
              {synonyms.map((synonym, index) => (
                <li key={index}>{synonym}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SymptomsList;