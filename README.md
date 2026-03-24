# AI Chatbot from Scratch - Intent-Based NLP Project

**Academic AI Project** | Computer Engineering | CHARUSAT  
**Course:** Natural Language Processing / AI (2022)  
**Student:** Sahil Patel (20CE101)  
**Approach:** Intent Matching + Deep Learning  
**Model:** Neural Network (TensorFlow/Keras)

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)

---

## 📚 Project Overview

An **intent-based conversational AI chatbot** built from scratch using Natural Language Processing (NLP) and Deep Learning. The bot understands user queries through pattern matching and responds intelligently using a trained neural network.

**Key Features:**
- ✅ **Intent Classification** - Understands user intent from messages
- ✅ **NLP Processing** - Tokenization, lemmatization, bag-of-words
- ✅ **Deep Learning** - Custom neural network (128-64 neurons)
- ✅ **JSON-based Training** - Easy-to-extend intent patterns
- ✅ **Real-time Responses** - Interactive command-line chat
- ✅ **Pre-trained Model** - Ready to use chatbotmodel.h5

---

## 🎯 How It Works

### Architecture Pipeline:

```
User Input → Tokenization → Lemmatization → Bag of Words → Neural Network → Intent Prediction → Response Selection
```

### 1. **Training Phase** (training.py)
- Load intents from JSON
- Tokenize patterns using NLTK
- Lemmatize words (WordNetLemmatizer)
- Create bag-of-words representation
- Build neural network (Sequential model)
- Train on patterns-intent pairs
- Save model as chatbotmodel.h5

### 2. **Inference Phase** (chatbot.py)
- Load trained model
- Accept user input
- Process through NLP pipeline
- Predict intent with confidence threshold
- Select random response from matching intent
- Display to user

---

## 📂 Repository Structure

```
Chatbot-from-Scratch/
├── chatbot.py              # Main chatbot runtime (83 lines)
├── training.py             # Model training script (81 lines)
├── intents.json            # Intent patterns and responses (120 lines)
├── chatbotmodel.h5         # Trained model (174 KB)
├── words.pkl               # Vocabulary pickle file
├── classes.pkl             # Intent classes pickle file
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore patterns
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
TensorFlow 2.x
NLTK
```

### Installation

```bash
# Clone repository
git clone https://github.com/patelsahil2k03/Chatbot-from-Scratch.git
cd Chatbot-from-Scratch

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Quick Start

#### Option 1: Use Pre-trained Model
```bash
# Run chatbot with existing model
python chatbot.py
```

#### Option 2: Train from Scratch
```bash
# Train new model
python training.py

# Run chatbot
python chatbot.py
```

---

## 💬 Usage Example

```bash
$ python chatbot.py
GO! BOT IS RUNNING !
> Hello
Hi there, how can I help?

> What do you sell?
We sell coffee and tea

> Tell me a joke
Why did the hipster burn his mouth? He drank the coffee before it was cool.

> Thanks
Happy to help!

> Bye
See you later, thanks for visiting
```

---

## 🧠 Technical Details

### Neural Network Architecture

```python
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
```

**Layers:**
- Input Layer: Variable size (vocabulary length)
- Hidden Layer 1: 128 neurons + ReLU + 50% Dropout
- Hidden Layer 2: 64 neurons + ReLU + 50% Dropout
- Output Layer: Softmax (number of intent classes)

**Training Configuration:**
- Optimizer: SGD (Stochastic Gradient Descent)
- Learning Rate: 0.01
- Momentum: 0.9 (Nesterov)
- Loss: Categorical Cross-entropy
- Epochs: 200
- Batch Size: 5

---

## 📝 Intent Configuration (intents.json)

### Available Intents (10 total):

1. **greeting** - Hi, Hello, Hey
2. **goodbye** - Bye, See you later
3. **thanks** - Thank you, Thanks a lot
4. **items** - What do you sell?
5. **payments** - Payment methods
6. **delivery** - Shipping information
7. **funny** - Tell me a joke
8. **time period** - Internship duration (custom)
9. **hours** - Business hours
10. **name** - Bot name (Flash)

### Adding New Intents:

```json
{
  "tag": "your_intent",
  "patterns": [
    "User question 1",
    "User question 2",
    "User question 3"
  ],
  "responses": [
    "Bot response 1",
    "Bot response 2"
  ]
}
```

After adding, retrain the model:
```bash
python training.py
```

---

## 🛠️ Technologies Used

### Core Libraries:
- **TensorFlow/Keras** - Deep learning framework
- **NLTK** - Natural language processing
- **NumPy** - Numerical computations
- **Pickle** - Model serialization
- **JSON** - Intent data storage

### NLP Techniques:
- **Tokenization** - Split text into words
- **Lemmatization** - Reduce words to base form
- **Bag of Words** - Convert text to numerical vectors
- **Intent Classification** - Categorize user messages

### Model Components:
- **Sequential Neural Network** - Feedforward architecture
- **Dense Layers** - Fully connected neurons
- **Dropout** - Regularization to prevent overfitting
- **Softmax Activation** - Multi-class probability distribution

---

## 📊 Code Breakdown

### training.py (Model Training)

**Key Functions:**
1. Load intents from JSON
2. Tokenize and lemmatize patterns
3. Create vocabulary (words) and classes (intents)
4. Generate bag-of-words training data
5. Build and compile neural network
6. Train model (200 epochs)
7. Save model and pickles

**Output Files:**
- `chatbotmodel.h5` - Trained model
- `words.pkl` - Vocabulary list
- `classes.pkl` - Intent classes

---

### chatbot.py (Inference)

**Key Functions:**

```python
clean_up_sentence(sentence)
# Tokenizes and lemmatizes input

bag_of_words(sentence)
# Converts sentence to numerical vector

predict_class(sentence)
# Predicts intent with confidence threshold (0.25)

get_response(intents_list, intents_json)
# Selects random response from matched intent
```

**Workflow:**
1. Load model and pickles
2. Enter infinite chat loop
3. Process user input through NLP pipeline
4. Predict intent (with 25% confidence threshold)
5. Select and display response

---

## 🎓 Academic Context

**Course:** Natural Language Processing / Artificial Intelligence  
**Semester:** 4th/5th Semester (2022)  
**Institution:** CHARUSAT - CSPIT  
**Program:** B.Tech Computer Engineering  
**Student ID:** 20CE101

**Purpose:** Build conversational AI from scratch to understand NLP and neural networks.

---

## 🚀 From Basics to Production (2022 → 2026)

This repository represents foundational NLP/AI learning from 2022.

**Journey Since Then:**
- 🤖 Built **production chatbots with LangChain** and advanced LLMs
- 📊 Achieved **98%+ accuracy in AI models** (production systems)
- 🔬 Published **2 SCOPUS-indexed papers** on AI/ML
- 🏆 **Top 10 Finalist** in AI-Manthan Hackathon
- ☁️ Deployed **50+ AWS Lambda AI functions**
- 💼 **Associate Software Engineer** building AI solutions at Digiflux

**Current Expertise:** LangChain, GPT, BERT, Transformers, Production NLP  
**Portfolio:** [patelsahil2k03.github.io](https://patelsahil2k03.github.io)

---

## 💡 Key Learnings

### NLP Concepts:
- ✅ **Tokenization** - Text preprocessing fundamentals
- ✅ **Lemmatization** - Word normalization techniques
- ✅ **Bag of Words** - Text vectorization
- ✅ **Intent Recognition** - Classification problems

### Deep Learning:
- ✅ **Neural Networks** - Architecture design
- ✅ **Backpropagation** - Training process
- ✅ **Dropout** - Regularization techniques
- ✅ **Softmax** - Multi-class classification

### Software Engineering:
- ✅ **Modular Design** - Separate training and inference
- ✅ **Model Persistence** - Saving/loading models
- ✅ **JSON Configuration** - Data-driven approach
- ✅ **Error Handling** - Confidence thresholds

---

## 🔧 Customization Guide

### 1. Change Bot Personality
Edit `intents.json` responses to match desired tone.

### 2. Add Domain-Specific Intents
Add new intent blocks for your use case (e-commerce, support, etc.)

### 3. Improve Accuracy
- Increase epochs (200 → 500)
- Add more training patterns per intent
- Tune hyperparameters (learning rate, neurons)

### 4. Multi-language Support
- Use language-specific lemmatizers
- Translate intents.json

### 5. GUI Integration
Wrap chatbot.py logic in Flask/Streamlit for web interface

---

## 📈 Performance Metrics

- **Training Time:** ~2-5 minutes (200 epochs)
- **Model Size:** 174 KB (lightweight)
- **Response Time:** < 100ms per query
- **Confidence Threshold:** 25% (adjustable)
- **Intent Accuracy:** Depends on training data quality

---

## 🤝 Contributing

This is an academic learning project. Feedback welcome!

**Enhancement Ideas:**
- Add more intents (100+ for robust bot)
- Implement context handling (conversation memory)
- Add sentiment analysis
- Integrate with web/mobile interface
- Add voice input/output

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Sahil Patel**  
**Email:** patelsahil2k03@gmail.com  
**Portfolio:** [patelsahil2k03.github.io](https://patelsahil2k03.github.io)  
**GitHub:** [@patelsahil2k03](https://github.com/patelsahil2k03)  
**LinkedIn:** [sahil-patel-581226205](https://linkedin.com/in/sahil-patel-581226205)

---

## 🌟 Acknowledgments

- **CHARUSAT CSPIT** - NLP/AI curriculum
- **TensorFlow Team** - Excellent deep learning library
- **NLTK** - Comprehensive NLP toolkit
- **AI Community** - Inspiration and resources

---

## 📚 References

- NLTK Documentation: https://www.nltk.org/
- TensorFlow Keras: https://www.tensorflow.org/guide/keras
- Intent-based Chatbot tutorials
- Bag-of-Words model explanations

---

*From basic intent matching to building production LLM applications - this project marks the beginning of my AI journey.*  
*Last Updated: March 2026*
