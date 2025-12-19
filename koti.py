from flask import Flask, render_template_string, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from google import genai
import re

app = Flask(__name__)

# =====================================
# Configuration
# =====================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# =====================================
# Load Tokenizer and Model
# =====================================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    "XSY/albert-base-v2-fakenews-discriminator"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "XSY/albert-base-v2-fakenews-discriminator"
)
model.eval()
print("Model loaded successfully.")

# =====================================
# Initialize Gemini Client
# =====================================
print("Initializing Gemini API...")
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
client = genai.Client()
print("Gemini API initialized successfully.")

# =====================================
# Prediction Function
# =====================================
def predict_fake_news(text):
    """Predict if text is fake or real news using transformer model"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    
    if real_prob > fake_prob:
        label = "REAL NEWS"
        confidence = real_prob
    else:
        label = "FAKE NEWS"
        confidence = fake_prob
    
    return label, confidence

# =====================================
# Format Gemini Response
# =====================================
def format_gemini_response(raw_text):
    """Convert Gemini markdown response to clean HTML"""
    # Remove markdown formatting
    text = raw_text
    
    # Extract credibility score if present
    score_match = re.search(r'(?:Overall Credibility Score|Credibility Score).*?(\d+)/10', text, re.IGNORECASE)
    credibility_score = int(score_match.group(1)) if score_match else None
    
    # Split into sections
    sections = re.split(r'\n(?=\d+\.\s+\*\*|\#{1,3}\s+)', text)
    formatted = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Handle numbered sections with bold titles
        section_match = re.match(r'(\d+)\.\s+\*\*(.+?)\*\*:?\s*(.*)', section, re.DOTALL)
        if section_match:
            num, title, content = section_match.groups()
            formatted.append(f'<div class="analysis-section">')
            formatted.append(f'<h3 class="section-title"><span class="section-number">{num}</span>{title}</h3>')
            
            # Process content
            content = content.strip()
            # Remove extra asterisks
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            
            # Handle bullet points
            if '•' in content or re.search(r'^\s*-\s+', content, re.MULTILINE):
                lines = content.split('\n')
                in_list = False
                for line in lines:
                    line = line.strip()
                    if line.startswith('•') or line.startswith('-'):
                        if not in_list:
                            formatted.append('<ul class="analysis-list">')
                            in_list = True
                        item = line.lstrip('•- ').strip()
                        if item:
                            formatted.append(f'<li>{item}</li>')
                    else:
                        if in_list:
                            formatted.append('</ul>')
                            in_list = False
                        if line:
                            formatted.append(f'<p class="analysis-text">{line}</p>')
                if in_list:
                    formatted.append('</ul>')
            else:
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for para in paragraphs:
                    if para:
                        formatted.append(f'<p class="analysis-text">{para}</p>')
            
            formatted.append('</div>')
        else:
            # Handle regular paragraphs
            content = section.strip()
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            if content:
                formatted.append(f'<p class="analysis-text">{content}</p>')
    
    html_output = ''.join(formatted)
    
    return html_output, credibility_score

# =====================================
# Media Credibility Analysis Function
# =====================================
def analyze_media_credibility(text):
    """Analyze media credibility using Gemini API"""
    prompt = f"""Analyze the credibility of the following news text. Provide a structured analysis with:

1. **Source Reliability**: Assess if the source appears credible (2-3 sentences)
2. **Bias Detection**: Check for political, emotional, or sensationalist bias (2-3 sentences)
3. **Fact-Checking Indicators**: Look for verifiable claims, citations, evidence (2-3 sentences)
4. **Language Quality**: Evaluate grammar and professionalism (2-3 sentences)
5. **Red Flags**: Identify any clickbait, misleading headlines, conspiracy theories (bullet points if present)
6. **Overall Credibility Score**: Rate from 1-10 (10 = highly credible) and provide a brief justification

News Text: {text}

Format your response with clear numbered sections. Keep each section concise and focused."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        formatted_html, score = format_gemini_response(response.text)
        return formatted_html, score
    except Exception as e:
        return f"<p class='error'>Error analyzing credibility: {str(e)}</p>", None

# =====================================
# AI Chat Assistant Function
# =====================================
def chat_with_ai(message):
    """Chat with AI assistant"""
    prompt = f"""You are a helpful AI assistant specializing in fake news detection and media literacy. 
Answer the user's question in a friendly, informative way.

User question: {message}

Provide a clear, concise answer."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# =====================================
# HTML Template
# =====================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Fake News Detection System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Preloader */
        .preloader {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            animation: fadeOut 0.5s ease-in-out 3.5s forwards;
        }

        .preloader-content {
            text-align: center;
            animation: fadeInScale 1s ease-in-out;
        }

        .preloader h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: clamp(1.3rem, 4vw, 2.5rem);
            color: #00d9ff;
            margin-bottom: 20px;
            font-weight: 900;
            letter-spacing: 3px;
            animation: fadeInUp 1s ease-in-out 0.5s both, glow 2s ease-in-out infinite;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5), 0 0 40px rgba(0, 217, 255, 0.3);
            padding: 0 20px;
            line-height: 1.4;
        }

        @keyframes glow {
            0%, 100% { text-shadow: 0 0 20px rgba(0, 217, 255, 0.5), 0 0 40px rgba(0, 217, 255, 0.3); }
            50% { text-shadow: 0 0 30px rgba(0, 217, 255, 0.8), 0 0 60px rgba(0, 217, 255, 0.5); }
        }

        .preloader-subtitle {
            font-size: clamp(0.9rem, 2vw, 1.1rem);
            color: #64b5f6;
            animation: fadeInUp 1s ease-in-out 1s both;
            padding: 0 20px;
            letter-spacing: 1px;
        }

        .loader {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(0, 217, 255, 0.2);
            border-top-color: #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite, fadeInUp 1s ease-in-out 1.5s both;
            margin: 30px auto 0;
        }

        @keyframes fadeOut {
            to {
                opacity: 0;
                visibility: hidden;
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInScale {
            from {
                opacity: 0;
                transform: scale(0.9);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Main Content */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            opacity: 0;
            animation: fadeIn 1s ease-in-out 4s forwards;
        }

        @keyframes fadeIn {
            to { opacity: 1; }
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 20px;
        }

        .header h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: clamp(1.8rem, 4vw, 3rem);
            color: #00d9ff;
            margin-bottom: 15px;
            text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
            font-weight: 900;
            letter-spacing: 2px;
        }

        .header p {
            font-size: clamp(1rem, 2vw, 1.2rem);
            color: rgba(255,255,255,0.8);
            letter-spacing: 0.5px;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: clamp(20px, 4vw, 40px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            margin-bottom: 30px;
            animation: slideInUp 0.6s ease-out;
        }

        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .input-group {
            margin-bottom: 25px;
        }

        .input-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: #00d9ff;
            font-size: clamp(0.95rem, 2vw, 1.1rem);
        }

        .input-group textarea {
            width: 100%;
            padding: 15px;
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid rgba(0, 217, 255, 0.3);
            border-radius: 12px;
            font-size: clamp(0.9rem, 2vw, 1rem);
            font-family: inherit;
            resize: vertical;
            min-height: 150px;
            transition: all 0.3s ease;
            color: #fff;
        }

        .input-group textarea::placeholder {
            color: rgba(255, 255, 255, 0.5);
        }

        .input-group textarea:focus {
            outline: none;
            border-color: #00d9ff;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
        }

        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn {
            flex: 1;
            min-width: 200px;
            padding: 15px 30px;
            border: none;
            border-radius: 12px;
            font-size: clamp(0.9rem, 2vw, 1.05rem);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .btn span {
            position: relative;
            z-index: 1;
        }

        .btn-detect {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            color: #fff;
        }

        .btn-detect:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.5);
        }

        .btn-credibility {
            background: linear-gradient(135deg, #7b2cbf 0%, #5a189a 100%);
            color: #fff;
        }

        .btn-credibility:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(123, 44, 191, 0.5);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .loading {
            text-align: center;
            padding: 40px;
            display: none;
            color: #00d9ff;
        }

        .loading.active {
            display: block;
            animation: fadeIn 0.3s ease-in;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(0, 217, 255, 0.2);
            border-top: 4px solid #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        .results {
            display: none;
            animation: fadeInUp 0.5s ease-out;
        }

        .results.active {
            display: block;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(0, 217, 255, 0.3);
            padding: clamp(20px, 4vw, 30px);
            border-radius: 15px;
            margin-bottom: 25px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }

        .result-header h3 {
            font-size: clamp(1.2rem, 3vw, 1.5rem);
            color: #00d9ff;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .label {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: clamp(0.85rem, 2vw, 1rem);
        }

        .label.fake {
            background: linear-gradient(135deg, #ff006e 0%, #d00056 100%);
            color: #fff;
            box-shadow: 0 0 20px rgba(255, 0, 110, 0.5);
        }

        .label.real {
            background: linear-gradient(135deg, #06ffa5 0%, #00cc7a 100%);
            color: #0a0e27;
            box-shadow: 0 0 20px rgba(6, 255, 165, 0.5);
        }

        .confidence-bar {
            margin-top: 15px;
        }

        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-weight: 600;
            color: #00d9ff;
            font-size: clamp(0.9rem, 2vw, 1rem);
        }

        .progress-bar {
            width: 100%;
            height: 12px;
            background: rgba(0, 217, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d9ff 0%, #0099cc 100%);
            border-radius: 10px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            width: 0;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.6);
        }

        .credibility-analysis {
            background: rgba(255, 255, 255, 0.05);
            padding: clamp(20px, 4vw, 30px);
            border-radius: 15px;
            color: #fff;
        }

        .score-summary {
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.2) 0%, rgba(123, 44, 191, 0.2) 100%);
            border: 2px solid rgba(0, 217, 255, 0.4);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            text-align: center;
        }

        .score-summary h3 {
            color: #00d9ff;
            font-size: clamp(1.1rem, 2.5vw, 1.3rem);
            margin-bottom: 15px;
        }

        .score-display {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .score-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            font-weight: 900;
            color: #fff;
            position: relative;
            box-shadow: 0 0 30px rgba(0, 217, 255, 0.5);
        }

        .score-circle.high {
            background: linear-gradient(135deg, #06ffa5 0%, #00cc7a 100%);
        }

        .score-circle.medium {
            background: linear-gradient(135deg, #ffd60a 0%, #ffa500 100%);
        }

        .score-circle.low {
            background: linear-gradient(135deg, #ff006e 0%, #d00056 100%);
        }

        .score-label {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 600;
        }

        .analysis-section {
            margin-bottom: 25px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            border-left: 4px solid #00d9ff;
        }

        .section-title {
            color: #00d9ff;
            font-size: clamp(1.05rem, 2.5vw, 1.2rem);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 600;
        }

        .section-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            border-radius: 50%;
            font-size: 0.9rem;
            font-weight: 700;
            color: #fff;
            flex-shrink: 0;
        }

        .analysis-text {
            color: rgba(255, 255, 255, 0.85);
            line-height: 1.7;
            margin-bottom: 10px;
            font-size: clamp(0.95rem, 2vw, 1.05rem);
        }

        .analysis-list {
            margin: 12px 0;
            padding-left: 25px;
            list-style: none;
        }

        .analysis-list li {
            position: relative;
            margin-bottom: 10px;
            color: rgba(255, 255, 255, 0.8);
            font-size: clamp(0.9rem, 2vw, 1rem);
            line-height: 1.6;
            padding-left: 15px;
        }

        .analysis-list li::before {
            content: '▸';
            position: absolute;
            left: 0;
            color: #00d9ff;
            font-weight: bold;
        }

        .credibility-analysis strong {
            color: #00d9ff;
            font-weight: 600;
        }

        .credibility-analysis em {
            color: #b388ff;
            font-style: normal;
        }

        /* Floating Chat Button */
        .chat-float-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0, 217, 255, 0.5);
            transition: all 0.3s ease;
            z-index: 1000;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .chat-float-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 30px rgba(0, 217, 255, 0.7);
        }

        .chat-float-btn svg {
            width: 30px;
            height: 30px;
            fill: white;
        }

        /* Chat Window */
        .chat-window {
            position: fixed;
            bottom: 100px;
            right: 30px;
            width: 350px;
            max-width: calc(100vw - 40px);
            height: 500px;
            max-height: calc(100vh - 150px);
            background: rgba(10, 14, 39, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 217, 255, 0.3);
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
            display: none;
            flex-direction: column;
            z-index: 999;
            animation: slideInRight 0.3s ease-out;
        }

        .chat-window.active {
            display: flex;
        }

        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .chat-header {
            padding: 20px;
            border-bottom: 1px solid rgba(0, 217, 255, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chat-header h3 {
            color: #00d9ff;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .chat-close {
            background: none;
            border: none;
            color: #fff;
            font-size: 24px;
            cursor: pointer;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .chat-close:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .chat-message {
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            animation: fadeInUp 0.3s ease-out;
        }

        .chat-message.user {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            color: #fff;
            align-self: flex-end;
            margin-left: auto;
        }

        .chat-message.bot {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            align-self: flex-start;
        }

        .chat-input-area {
            padding: 20px;
            border-top: 1px solid rgba(0, 217, 255, 0.3);
            display: flex;
            gap: 10px;
        }

        .chat-input {
            flex: 1;
            padding: 12px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(0, 217, 255, 0.3);
            border-radius: 8px;
            color: #fff;
            font-family: inherit;
            font-size: 0.95rem;
        }

        .chat-input:focus {
            outline: none;
            border-color: #00d9ff;
        }

        .chat-send-btn {
            padding: 12px 20px;
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .chat-send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.5);
        }

        @media (max-width: 768px) {
            .button-group {
                flex-direction: column;
            }

            .btn {
                width: 100%;
                min-width: unset;
            }

            .result-header {
                flex-direction: column;
                align-items: flex-start;
            }

            .chat-float-btn {
                bottom: 20px;
                right: 20px;
                width: 55px;
                height: 55px;
            }

            .chat-window {
                right: 20px;
                bottom: 85px;
            }
        }
    </style>
</head>
<body>
    <!-- Preloader -->
    <div class="preloader">
        <div class="preloader-content">
            <h1>A Multilingual AI-Based Fake News and Media Credibility Detection System</h1>
            <p class="preloader-subtitle">Powered by Advanced Machine Learning</p>
            <div class="loader"></div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="container">
        <div class="header">
            <h1>🎯 AI News Credibility Analyzer</h1>
            <p>Advanced fake news detection powered by AI & Machine Learning</p>
        </div>

        <div class="card">
            <div class="input-group">
                <label for="newsText">📝 Enter News Article or Headline:</label>
                <textarea 
                    id="newsText" 
                    placeholder="Paste your news article or headline here for analysis..."
                ></textarea>
            </div>

            <div class="button-group">
                <button class="btn btn-detect" onclick="detectFakeNews()">
                    <span>🔍 Detect Fake News</span>
                </button>
                <button class="btn btn-credibility" onclick="analyzeCredibility()">
                    <span>📊 Media Credibility</span>
                </button>
                <button class="btn btn-secondary" onclick="clearForm()">
                    <span>🔄 Clear</span>
                </button>
            </div>
        </div>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🤖 Analyzing with AI...</p>
        </div>

        <div class="results" id="results">
            <div class="card result-card" id="detectionResult" style="display: none;">
                <div class="result-header">
                    <h3>🎯 Fake News Detection</h3>
                    <span class="label" id="resultLabel">-</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-label">
                        <span>Confidence Level</span>
                        <span id="confidencePercent">0%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>
            </div>

            <div class="card" id="credibilityResult" style="display: none;">
                <h3 style="color: #00d9ff; margin-bottom: 20px; font-size: clamp(1.2rem, 3vw, 1.5rem);">📊 Media Credibility Analysis</h3>
                
                <div id="scoreContainer" style="display: none;" class="score-summary">
                    <h3>Overall Credibility Score</h3>
                    <div class="score-display">
                        <div class="score-circle" id="scoreCircle">
                            <span id="scoreValue">-</span>
                        </div>
                        <div class="score-label" id="scoreLabel">out of 10</div>
                    </div>
                </div>
                
                <div class="credibility-analysis" id="credibilityAnalysis">
                    Analysis will appear here...
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Chat Button -->
    <div class="chat-float-btn" onclick="toggleChat()">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/>
            <circle cx="12" cy="10" r="1.5"/>
            <circle cx="8" cy="10" r="1.5"/>
            <circle cx="16" cy="10" r="1.5"/>
        </svg>
    </div>

    <!-- Chat Window -->
    <div class="chat-window" id="chatWindow">
        <div class="chat-header">
            <h3>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="#00d9ff" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
                </svg>
                AI Assistant
            </h3>
            <button class="chat-close" onclick="toggleChat()">×</button>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="chat-message bot">
                👋 Hi! I'm your AI assistant. Ask me anything about fake news detection, media literacy, or how this system works!
            </div>
        </div>
        <div class="chat-input-area">
            <input type="text" class="chat-input" id="chatInput" placeholder="Type your message...">
            <button class="chat-send-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        let currentAnalysisType = null;

        async function detectFakeNews() {
            const text = document.getElementById('newsText').value.trim();
            
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }

            currentAnalysisType = 'detection';
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');
            document.getElementById('detectionResult').style.display = 'none';
            document.getElementById('credibilityResult').style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text, type: 'detection' })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                const label = document.getElementById('resultLabel');
                label.textContent = data.label;
                label.className = 'label ' + (data.label === 'FAKE NEWS' ? 'fake' : 'real');

                const confidence = Math.round(data.confidence * 100);
                document.getElementById('confidencePercent').textContent = confidence + '%';
                
                setTimeout(() => {
                    document.getElementById('progressFill').style.width = confidence + '%';
                }, 100);

                document.getElementById('loading').classList.remove('active');
                document.getElementById('results').classList.add('active');
                document.getElementById('detectionResult').style.display = 'block';

                document.getElementById('results').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });

            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.remove('active');
            }
        }

        async function analyzeCredibility() {
            const text = document.getElementById('newsText').value.trim();
            
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }

            currentAnalysisType = 'credibility';
            document.getElementById('loading').classList.add('active');
            document.getElementById('results').classList.remove('active');
            document.getElementById('detectionResult').style.display = 'none';
            document.getElementById('credibilityResult').style.display = 'none';

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text, type: 'credibility' })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                document.getElementById('credibilityAnalysis').innerHTML = data.credibility_analysis;

                // Display score if available
                if (data.credibility_score !== null && data.credibility_score !== undefined) {
                    const scoreContainer = document.getElementById('scoreContainer');
                    const scoreCircle = document.getElementById('scoreCircle');
                    const scoreValue = document.getElementById('scoreValue');
                    
                    scoreValue.textContent = data.credibility_score;
                    
                    // Color code based on score
                    scoreCircle.className = 'score-circle';
                    if (data.credibility_score >= 7) {
                        scoreCircle.classList.add('high');
                    } else if (data.credibility_score >= 4) {
                        scoreCircle.classList.add('medium');
                    } else {
                        scoreCircle.classList.add('low');
                    }
                    
                    scoreContainer.style.display = 'block';
                } else {
                    document.getElementById('scoreContainer').style.display = 'none';
                }

                document.getElementById('loading').classList.remove('active');
                document.getElementById('results').classList.add('active');
                document.getElementById('credibilityResult').style.display = 'block';

                document.getElementById('results').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });

            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.remove('active');
            }
        }

        function clearForm() {
            document.getElementById('newsText').value = '';
            document.getElementById('results').classList.remove('active');
            document.getElementById('progressFill').style.width = '0%';
            document.getElementById('detectionResult').style.display = 'none';
            document.getElementById('credibilityResult').style.display = 'none';
        }

        function toggleChat() {
            const chatWindow = document.getElementById('chatWindow');
            chatWindow.classList.toggle('active');
        }

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;

            const messagesContainer = document.getElementById('chatMessages');
            
            const userMsg = document.createElement('div');
            userMsg.className = 'chat-message user';
            userMsg.textContent = message;
            messagesContainer.appendChild(userMsg);
            
            input.value = '';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                const botMsg = document.createElement('div');
                botMsg.className = 'chat-message bot';
                botMsg.textContent = data.response;
                messagesContainer.appendChild(botMsg);
                
                messagesContainer.scrollTop = messagesContainer.scrollHeight;

            } catch (error) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'chat-message bot';
                errorMsg.textContent = 'Sorry, I encountered an error. Please try again.';
                messagesContainer.appendChild(errorMsg);
            }
        }

        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        document.getElementById('newsText').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                detectFakeNews();
            }
        });
    </script>
</body>
</html>
"""

# =====================================
# Routes
# =====================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        analysis_type = data.get('type', 'detection')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if analysis_type == 'detection':
            label, confidence = predict_fake_news(text)
            return jsonify({
                'label': label,
                'confidence': float(confidence)
            })
        elif analysis_type == 'credibility':
            credibility_analysis, score = analyze_media_credibility(text)
            return jsonify({
                'credibility_analysis': credibility_analysis,
                'credibility_score': score
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chat_with_ai(message)
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask Server")
    print("="*60)
    print("🌐 Access the app at: http://127.0.0.1:5000")
    print("="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
