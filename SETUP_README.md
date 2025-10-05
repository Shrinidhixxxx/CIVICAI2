# CivicMindAI - Complete Setup & Usage Guide

## 🏛️ Project Overview

**CivicMindAI** is a comprehensive AI-powered civic assistant for Chennai, covering all 15 zones, 200 wards, and every major civic department. It demonstrates advanced AI technologies including RAG, KAG, CAG, Federated Learning, and AutoML optimization.

## 📦 Quick Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your_api_key_here"

# Option 2: Enter in Streamlit sidebar when running
```

### 3. Run Application
```bash
streamlit run app.py
```

### 4. Access Application
- Open browser to: `http://localhost:8501`
- Login with: `admin` / `admin`
- Enter OpenAI API key in sidebar
- Start asking civic queries!

## 🗂️ Complete File Structure

```
CivicMindAI/
├── app.py                           # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # This setup guide
├── chennai_complete_civic_data.json # Complete Chennai civic data
├── chennai_pincode_mapping.json    # Pincode to area mapping
│
├── ai_engine/                      # AI processing modules
│   ├── __init__.py
│   ├── rag.py                     # Retrieval-Augmented Generation
│   ├── kag.py                     # Knowledge Graph Augmented Generation
│   ├── cag.py                     # Cache-Augmented Generation
│   ├── fl_manager.py              # Federated Learning Manager
│   └── automl_opt.py              # AutoML Optimization
│
└── cache/                         # Auto-created cache directory
    └── (cached responses)
```

## 🚀 Features Overview

### Core AI Technologies
- **RAG**: Live data from Chennai civic websites
- **KAG**: Graph reasoning over civic entities
- **CAG**: Intelligent response caching
- **FL**: User feedback learning
- **AutoML**: Dynamic parameter optimization

### Complete Chennai Coverage
- **15 Zones**: All Greater Chennai Corporation zones
- **200 Wards**: Complete ward-level coverage
- **4 Major Departments**: GCC, Metro Water, TANGEDCO, TNSTC
- **85+ Pincode Areas**: Comprehensive area mapping

## 💬 Sample Queries to Try

```
Water supply issue in Adyar
Garbage collection problem in T. Nagar Zone 9
Electricity outage in Velachery Ward 175
Road pothole complaint in Anna Nagar
Bus route information from Koyambedu to Marina
Building approval process in Zone 8
Street light not working in Mylapore pincode 600004
Property tax payment in Kodambakkam
New water connection in Sholinganallur
TANGEDCO billing complaint for Guindy area
```

## 🏛️ Departments & Contacts

### Greater Chennai Corporation
- **Contact**: 1913
- **Services**: Garbage, Roads, Building approvals, Property tax
- **App**: Namma Chennai App

### Chennai Metro Water (CMWSSB)  
- **Contact**: 044-4567-4567 (24x7)
- **Services**: Water supply, Sewerage, New connections
- **Website**: cmwssb.tn.gov.in

### TANGEDCO (Electricity)
- **Contact**: 94987-94987, 1912
- **Services**: Power supply, Billing, Fault repair
- **WhatsApp**: 94458508111

### TNSTC (Transport)
- **Contact**: 1800-599-1500
- **Services**: Bus transport, Route complaints
- **WhatsApp**: 94450-14448

## 📊 Using the Dashboard

### Chat Tab
- Ask questions about any civic issue
- Get actionable responses with contact numbers
- View processing time and department routing
- Provide feedback (👍/👎) to improve AI

### Insights Tab
- View analytics: query patterns, response times
- Department-wise performance metrics
- Cache hit rates and satisfaction scores
- Export analytics and chat history

### About Tab
- Technical details about AI modules
- Complete Chennai coverage information
- Usage instructions and examples

## ⚙️ Advanced Configuration

### AI Module Settings
```python
# In app.py, modify these settings:
class CivicMindAI:
    def setup_ai_modules(self):
        self.rag_module = RAGModule(self.civic_data)
        self.kag_module = KnowledgeGraphModule(self.civic_data)
        self.cache_module = CacheModule(ttl_hours=24)  # Cache expiry
        self.fl_manager = FederatedLearningManager(learning_rate=0.1)
        self.automl_optimizer = AutoMLOptimizer()
```

### Cache Configuration
```python
# Modify cache settings in ai_engine/cag.py
cache_module = CacheModule(
    cache_dir="cache",
    ttl_hours=24  # Cache expiry time
)
```

### Optimization Settings
```python
# Modify in ai_engine/automl_opt.py
optimizer = AutoMLOptimizer(
    optimization_metric="satisfaction_score"
)
```

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Install missing packages
pip install streamlit openai requests beautifulsoup4 networkx faiss-cpu pandas numpy optuna plotly sentence-transformers
```

**2. OpenAI API Errors**
- Ensure valid API key is entered
- Check API quota and billing
- Try GPT-3.5-turbo model first

**3. Cache Directory Errors**
```bash
# Create cache directory manually
mkdir cache
chmod 755 cache
```

**4. Slow Response Times**
- Use GPT-3.5-turbo instead of GPT-4
- Clear cache in Insights tab
- Check internet connection for live data fetching

### Performance Optimization

**For Better Speed:**
- Use cached responses when available
- Select GPT-3.5-turbo model
- Clear old cache files regularly

**For Better Accuracy:**
- Provide specific area names (e.g., "Adyar Ward 174")
- Include issue type clearly (water/electricity/garbage)
- Use feedback buttons to train the AI

## 📈 Monitoring & Analytics

### Key Metrics to Watch
- **Response Time**: Target <3 seconds
- **Cache Hit Rate**: Target >40%
- **User Satisfaction**: Target >80%
- **Department Accuracy**: Target >90%

### Export Options
- **Analytics JSON**: Performance metrics and trends
- **Chat History CSV**: All conversations and responses
- **Learning Data**: Federated learning insights

## 🎓 Academic Submission Guide

### Project Highlights
- **Complete AI Pipeline**: RAG → KAG → CAG → FL → AutoML
- **Real Data Integration**: Live Chennai civic sources
- **Scalable Architecture**: Handles all Chennai areas
- **Performance Analytics**: Comprehensive metrics tracking
- **User-Centric Design**: Feedback-driven improvements

### Submission Checklist
- [ ] All dependencies in requirements.txt
- [ ] Complete README with setup instructions
- [ ] Working demo with sample queries
- [ ] Analytics dashboard showing performance
- [ ] Documentation of AI modules
- [ ] Chennai civic data integration proof
- [ ] Deployment-ready configuration

## 🔐 Security & Privacy

- **No Data Storage**: Chat history in session only
- **API Key Security**: Encrypted in memory
- **Privacy-First**: No personal data collection
- **Secure Defaults**: All connections use HTTPS

## 🌟 Advanced Features

### Federated Learning Simulation
- Adapts to user feedback patterns
- Improves response quality over time
- Tracks department-wise performance

### AutoML Optimization  
- Dynamic hyperparameter tuning
- Performance-based model optimization
- Real-time parameter adjustment

### Knowledge Graph Reasoning
- Understands Chennai civic entity relationships
- Provides contextual escalation paths
- Maps areas to correct departments

## 🤝 Contributing

This is an academic project demonstrating AI capabilities in civic assistance. For improvements:

1. Fork the repository
2. Create feature branch
3. Test with multiple Chennai areas
4. Submit pull request with performance metrics

## 📞 Support

For technical issues:
- Check logs in Streamlit console
- Verify all dependencies installed
- Ensure OpenAI API key is valid
- Test with simple queries first

## 🏆 Project Success Metrics

**Technical Achievement:**
- ✅ Complete Chennai coverage (15 zones, 200 wards)
- ✅ Advanced AI integration (RAG+KAG+CAG+FL+AutoML)
- ✅ Real-time data processing
- ✅ Performance optimization
- ✅ User feedback integration

**Academic Value:**
- ✅ Demonstrates cutting-edge AI techniques
- ✅ Practical application for civic assistance
- ✅ Scalable and extensible architecture
- ✅ Comprehensive documentation
- ✅ Ready for deployment and evaluation

---

**🏛️ CivicMindAI - Empowering Chennai Citizens with AI**

*Built for academic excellence and civic impact*