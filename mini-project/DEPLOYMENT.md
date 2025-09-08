# Vercel Deployment Guide

## Prerequisites
1. Vercel account (sign up at vercel.com)
2. Git repository (GitHub, GitLab, or Bitbucket)
3. Node.js installed locally (for building React app)

## Deployment Steps

### 1. Prepare Your Repository
- Ensure all files are committed and pushed to your Git repository
- Make sure your React app builds successfully locally

### 2. Deploy to Vercel

#### Option A: Deploy via Vercel Dashboard
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your Git repository
4. Vercel will automatically detect the configuration from `vercel.json`

#### Option B: Deploy via Vercel CLI
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in your project directory
3. Follow the prompts to configure your project

### 3. Configure Environment Variables
In your Vercel dashboard, go to Project Settings > Environment Variables and add:

```
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
RECIPIENT_EMAIL=shreyasssanil62@gmail.com
PYTHON_VERSION=3.11.9
```

### 4. Build Configuration
- **Framework Preset**: Create React App
- **Build Command**: `npm install && npm run build`
- **Output Directory**: `build`
- **Install Command**: `npm install`

### 5. API Routes
The following API endpoints will be available:
- `POST /api/predict` - Disease prediction
- `POST /api/feedback` - Send feedback

### 6. Frontend Configuration
Make sure your React app's API calls point to the correct endpoints:
- Update API base URL to your Vercel domain
- Example: `https://your-app.vercel.app/api/predict`

## File Structure for Vercel
```
mini-project/
├── api/
│   ├── predict.py          # Disease prediction API
│   └── feedback.py         # Feedback API
├── src/                    # React frontend
├── public/                 # Static assets
├── package.json           # Node.js dependencies
├── vercel.json           # Vercel configuration
├── .vercelignore         # Files to ignore
└── requirements.txt      # Python dependencies
```

## Troubleshooting

### Common Issues:
1. **Build Failures**: Check that all dependencies are in `requirements.txt` and `package.json`
2. **API Errors**: Verify environment variables are set correctly
3. **CORS Issues**: The Flask app includes CORS configuration
4. **File Path Issues**: Ensure all file paths are relative to the project root

### Local Testing:
1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel dev` to test locally
3. This will simulate the Vercel environment

## Post-Deployment
1. Test all API endpoints
2. Verify email functionality works
3. Check that the React frontend loads correctly
4. Test the disease prediction feature

## Custom Domain (Optional)
1. Go to Project Settings > Domains
2. Add your custom domain
3. Configure DNS settings as instructed by Vercel
