# Email Setup Guide for Health AI Feedback System

## 🚀 Quick Setup (3 Steps)

### Step 1: Set Up Gmail App Password

1. **Go to your Google Account settings**: https://myaccount.google.com/
2. **Enable 2-Step Verification** (if not already enabled)
   - Go to Security → 2-Step Verification
   - Follow the setup process
3. **Generate App Password**:
   - Go to Security → App passwords
   - Select "Mail" as the app
   - Click "Generate"
   - Copy the 16-character password (e.g., `abcd efgh ijkl mnop`)

### Step 2: Update Email Configuration

1. **Open the file**: `email_config.py`
2. **Replace the placeholder values**:
   ```python
   EMAIL_USER = 'your-actual-gmail@gmail.com'  # Your Gmail address
   EMAIL_PASSWORD = 'abcd efgh ijkl mnop'      # Your 16-character app password
   ```
3. **Save the file**

### Step 3: Test the Email System

1. **Restart your Flask server**:
   ```bash
   python main.py
   ```
2. **Go to your website** and submit feedback
3. **Check your email** at `shreyasssanil62@gmail.com`

## 📧 What You'll Receive

When someone submits feedback, you'll get an email with:
- **Subject**: "Health AI Feedback: [User's Subject]"
- **Content**: Name, email, category, rating, subject, and message

## 🔧 Troubleshooting

### If emails don't send:

1. **Check Gmail settings**:
   - Make sure 2-Step Verification is enabled
   - Verify the app password is correct
   - Check that "Less secure app access" is disabled (use app password instead)

2. **Check the console**:
   - Look for error messages in your Flask server console
   - Common errors: "Authentication failed", "Invalid credentials"

3. **Test with a simple email**:
   ```python
   # Add this to main.py temporarily for testing
   send_email("Test Email", "This is a test email from Health AI")
   ```

### Common Issues:

- **"Authentication failed"**: Wrong app password or email
- **"SMTP connection failed"**: Check internet connection
- **"Invalid credentials"**: Use app password, not regular password

## 🔒 Security Notes

- **Never commit** your actual email credentials to Git
- **Use environment variables** in production
- **App passwords are safer** than regular passwords
- **Keep your app password private**

## 📱 Alternative Email Services

If you prefer other email services:

### Outlook/Hotmail:
```python
EMAIL_HOST = 'smtp-mail.outlook.com'
EMAIL_PORT = 587
```

### Yahoo:
```python
EMAIL_HOST = 'smtp.mail.yahoo.com'
EMAIL_PORT = 587
```

## ✅ Success Indicators

When working correctly, you should see:
- Console message: "Email sent successfully to shreyasssanil62@gmail.com"
- Email received in your inbox
- Success message on the website

---

**Need help?** Check the console output for specific error messages!




