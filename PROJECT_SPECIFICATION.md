# Bikini Bottom Current Classifier - Project Specification

## Executive Summary

The Bikini Bottom Current Classifier is a multi-modal machine learning system designed for **real-time ocean current classification in the Strait of Georgia, British Columbia, Canada**. The system combines deep learning (computer vision) with environmental sensor data to classify ocean currents into three categories: **Fraser River Outflow**, **Tidal Forcing**, and **Wind/Storm Forcing**.

**Project Name:** Bikini Bottom Current Classifier
**Created For:** OceanHack 2025
**Team:** The Bikini Bottom Team
**Model Accuracy:** 81.1% validation accuracy
**Training Time:** 35 minutes on NVIDIA H200 GPU
**Dataset Size:** 8,168 samples spanning 2 years (2023-2025)

---

## 1. Problem Statement

### 1.1 Challenge
Ocean currents in coastal regions like the Strait of Georgia are driven by multiple forcing mechanisms that interact in complex ways. Understanding which forcing mechanism dominates at any given time is critical for:

- **Maritime Safety:** Navigation, small craft warnings, search and rescue operations
- **Environmental Management:** Salmon migration tracking, estuarine water quality monitoring
- **Resource Management:** Oil spill response, tidal energy planning, harbor operations
- **Scientific Research:** Understanding coastal oceanography and climate impacts

Traditional methods require manual interpretation of current velocity plots and environmental data by trained oceanographers, which is time-consuming and not scalable for real-time operations.

### 1.2 Solution
This project develops an **automated classification system** that:
1. Analyzes ocean current velocity plots (visualizations from HF Radar data)
2. Integrates real-time environmental sensor data (wind, tides, river discharge)
3. Predicts the dominant forcing mechanism with confidence scores
4. Provides explanations and operational recommendations

---

## 2. System Architecture

### 2.1 Overview
The system uses a **Multi-Modal Deep Learning** approach that fuses:
- **Visual Information:** Current velocity vector plots (PNG images) from HF Radar
- **Environmental Features:** 11 numerical features from various sensors

```
┌─────────────────────┐      ┌──────────────────────┐
│ Current Plot (.png) │      │ Environmental Data   │
│ - Vector field      │      │ - Wind speed/dir     │
│ - Spatial patterns  │      │ - Tidal level        │
└──────────┬──────────┘      │ - Fraser River level │
           │                 │ - Atmospheric pressure│
           │                 └──────────┬───────────┘
           │                            │
           v                            v
┌──────────────────────┐    ┌──────────────────────┐
│ EfficientNet-B4      │    │ Environmental Encoder│
│ (Vision Backbone)    │    │ (11 → 128 features)  │
└──────────┬───────────┘    └──────────┬───────────┘
           │                            │
           └────────────┬───────────────┘
                        v
           ┌────────────────────────┐
           │ Fusion Classifier      │
           │ (1920 → 512 → 128 → 3) │
           └────────────┬───────────┘
                        v
           ┌────────────────────────┐
           │ Classification Output  │
           │ - OUTFLOW              │
           │ - TIDES                │
           │ - WIND/STORM           │
           └────────────────────────┘
```

### 2.2 Model Architecture

**Class:** `MultiModalClassifier` (PyTorch nn.Module)

**Components:**
1. **Vision Backbone (EfficientNet-B4)**
   - Pretrained on ImageNet
   - Input: 224×224 RGB images
   - Output: 1792-dimensional feature vector
   - Parameters: ~17.5M

2. **Environmental Encoder**
   - 2-layer MLP with ReLU activation
   - Architecture: Linear(11→64) → ReLU → Dropout(0.3) → Linear(64→128) → ReLU
   - Input: 11 environmental features (normalized)
   - Output: 128-dimensional feature vector
   - Parameters: ~9K

3. **Fusion Classifier**
   - 3-layer MLP with dropout regularization
   - Architecture: Concat(1792+128) → Dropout(0.4) → Linear(1920→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Linear(128→3)
   - Output: 3 class logits → Softmax probabilities
   - Parameters: ~1.1M

**Total Parameters:** 18.6 million

### 2.3 Environmental Features (11 dimensions)

| Feature | Description | Normalization Range |
|---------|-------------|---------------------|
| wind_speed | Wind speed from buoy | 0-60 km/h |
| wind_dir | Wind direction | 0-360° |
| tidal_level | Water level above chart datum | 0-6 meters |
| tidal_range | Absolute difference from mean tide | 0-3 meters |
| tidal_velocity | Rate of tidal change | -1 to +1 m/hr |
| fraser_level | Fraser River discharge level | 0-1 meter (normalized) |
| pressure | Atmospheric pressure | 980-1050 hPa |
| current_speed | Mean current speed (from .tuv) | 0-100 cm/s |
| is_spring_freshet | Binary flag (May-Jul) | 0 or 1 |
| wind_forcing | Wind speed (repeated for forcing) | 0-60 km/h |
| tidal_forcing | Tidal range (repeated for forcing) | 0-3 meters |

---

## 3. Training Details

### 3.1 Dataset

**Source:** HF Radar current measurements from Strait of Georgia Array
**Timespan:** January 2023 - July 2025 (2+ years)
**Total Samples:** 8,168 hourly observations
**Resolution:** 1 hour temporal resolution, ~2km spatial resolution

**Class Distribution:**
- **Fraser River Outflow:** 572 samples (7.0%)
- **Tidal Forcing:** 3,338 samples (40.9%)
- **Wind/Storm Forcing:** 4,258 samples (52.1%)

**Data Split:**
- Training: ~70%
- Validation: ~15%
- Test: ~15%

**Labeling Method:** Hybrid approach combining:
1. Rule-based classification using physical oceanography principles
2. Manual verification and correction
3. Cross-validation with domain experts

### 3.2 Training Configuration

**Hardware:**
- GPU: NVIDIA H200 (150GB HBM3)
- Training Time: 35 minutes

**Hyperparameters:**
- Optimizer: Adam
- Learning Rate: Adaptive (likely with scheduler)
- Batch Size: Not specified (estimated 16-32)
- Loss Function: CrossEntropyLoss
- Regularization: Dropout (0.3-0.4), Data augmentation

**Data Augmentation (for images):**
- Random rotations
- Color jittering
- Normalization: ImageNet statistics

### 3.3 Performance Metrics

**Validation Accuracy:** 81.1%

**Per-Class Performance:**
- **Fraser River Outflow:** 68.6% accuracy (challenging due to class imbalance)
- **Tidal Forcing:** 85.4% accuracy
- **Wind/Storm Forcing:** 89.4% accuracy (easiest to identify)

**Model Files:**
- Trained Model: `best_model_OPTION3.pth`
- Test Predictions: `test_predictions_FINAL.csv`

---

## 4. System Components

### 4.1 Core Files

| File | Purpose | Lines of Code |
|------|---------|---------------|
| `app.py` | Streamlit web application (main interface) | 751 |
| `live_data_fetcher.py` | Real-time data acquisition from APIs | 320 |
| `live_monitor.py` | Continuous monitoring and prediction system | 314 |
| `best_model_OPTION3.pth` | Trained PyTorch model checkpoint | Binary |
| `requirements.txt` | Python dependencies (220+ packages) | 222 |

### 4.2 Application (`app.py`)

**Framework:** Streamlit 1.49.1
**Interface Theme:** "SpongeBob SquarePants" themed ("Bikini Bottom")

**Features:**

**Tab 1: Upload & Classify**
- Upload current plot (.png file)
- Manual environmental parameter input (sliders/dropdowns)
- Real-time prediction with confidence scores
- Visual probability breakdown (bar chart)
- Detailed interpretation with oceanographic context

**Tab 2: Live from the Reef**
- Fetch real-time data from environmental APIs:
  - Buoy data (wind, pressure, temperature)
  - Tidal data from DFO IWLS API
  - Fraser River discharge from Water Office Canada
- Combine live data with uploaded plots
- Real-time classification

**Tab 3: Hall of Fame**
- Model performance statistics
- Training details
- Example predictions gallery
- Dataset information

**Styling:**
- Custom CSS with ocean/underwater theme
- Animated bubbles
- Responsive layout
- Color-coded predictions (green=outflow, blue=tides, yellow=wind)

### 4.3 Live Data Fetcher (`live_data_fetcher.py`)

**Class:** `LiveOceanData`

**Data Sources:**
1. **Buoy Data**
   - Endpoint: Environment Canada MSC Datamart / SmartAtlantic ERDDAP
   - Stations: 46131 (Sentry Shoal), 46146 (Halibut Bank), etc.
   - Parameters: Wind speed/direction, pressure, temperature

2. **Tidal Data**
   - Endpoint: DFO IWLS API (https://api-iwls.dfo-mpo.gc.ca)
   - Stations: 7735 (Point Atkinson), 7786 (Vancouver)
   - Parameters: Water level, tidal velocity (computed)

3. **Fraser River Data**
   - Endpoint: Water Office Canada (https://wateroffice.ec.gc.ca)
   - Station: 08MH053 (Fraser River at Deas Island)
   - Parameters: River level/discharge

**Fallback Mechanism:**
- If API fails, uses physics-based simulation:
  - M2 tidal constituent for tides (12.42 hour period)
  - Seasonal climatology for Fraser River
  - Realistic noise for buoy data

**Methods:**
- `get_latest_buoy_data()`: Fetch wind/pressure
- `get_latest_tidal_data()`: Fetch tides with velocity calculation
- `get_latest_fraser_data()`: Fetch river discharge
- `get_all_live_data()`: Combine all sources

### 4.4 Live Monitor (`live_monitor.py`)

**Class:** `LiveCurrentMonitor`

**Purpose:** Automated continuous monitoring system

**Features:**
1. Load trained model from checkpoint
2. Fetch live environmental data
3. Find latest current plot in directory
4. Run inference
5. Log predictions to CSV
6. Repeat at specified interval (default 60 minutes)

**Usage:**
```bash
# Single prediction
python live_monitor.py --model best_model_OPTION3.pth --image plot.png --once

# Continuous monitoring
python live_monitor.py --model best_model_OPTION3.pth --plots-dir plots --interval 60
```

**Output:**
- Real-time console display
- CSV log: `live_monitoring_log.csv`
- Includes timestamp, prediction, confidence, probabilities, environmental conditions

---

## 5. Data Flow

### 5.1 Offline (Web App)
```
User uploads PNG → Enters environmental params → Model inference → Display results
```

### 5.2 Live Mode (Web App)
```
Click "Fetch Live Data" → API calls → Retrieve env data → User uploads PNG → Model inference → Display
```

### 5.3 Automated Monitoring
```
Timer triggers → Fetch live APIs → Find latest plot → Model inference → Log to CSV → Sleep → Repeat
```

---

## 6. Classification Categories

### 6.1 Fraser River Outflow
**Physical Mechanism:** Freshwater discharge from Fraser River creates buoyant plume
**Characteristics:**
- Strong northward/westward flow at surface
- Seasonal pattern (peak in June during spring freshet)
- Low salinity signature
- Influenced by snowmelt, rainfall

**Importance:**
- Critical for salmon migration (creates "salmon highway")
- Affects estuarine water quality
- Nutrient transport
- Sediment dispersal

**Model Accuracy:** 68.6%

### 6.2 Tidal Forcing
**Physical Mechanism:** Gravitational pull of moon/sun drives water level changes
**Characteristics:**
- Regular oscillation (M2 period ~12.4 hours)
- Flood (rising) vs. Ebb (falling) phases
- Predictable from tidal predictions
- Bidirectional flow patterns

**Importance:**
- Navigation timing
- Harbor operations
- Tidal energy potential
- Coastal engineering

**Model Accuracy:** 85.4%

### 6.3 Wind/Storm Forcing
**Physical Mechanism:** Wind stress on ocean surface drives currents
**Characteristics:**
- Correlated with high wind speeds (>20 km/h)
- Low atmospheric pressure
- Irregular, episodic events
- Can override tidal/outflow signals

**Importance:**
- Maritime safety (small craft warnings)
- Search and rescue operations
- Oil spill response
- Storm surge prediction

**Model Accuracy:** 89.4%

---

## 7. Technical Requirements

### 7.1 Dependencies

**Core ML/DL:**
- PyTorch 2.9.0
- torchvision 0.24.0
- timm 1.0.20 (EfficientNet)
- scikit-learn 1.7.2
- numpy 1.26.4

**Web Application:**
- streamlit 1.49.1
- matplotlib 3.10.6
- Pillow 11.3.0

**Data Processing:**
- pandas 2.3.2
- requests 2.32.5

**Additional:**
- See `requirements.txt` for full list (220+ packages)

### 7.2 System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- CPU with AVX support

**Recommended:**
- Python 3.10+
- 16GB RAM
- GPU with CUDA support (optional, for faster inference)
- Internet connection (for live data APIs)

---

## 8. Usage Instructions

### 8.1 Installation
```bash
# Clone repository
git clone https://github.com/udbhav25kansal/Ocean-Hackathon-2025.git
cd Ocean-Hackathon-2025

# Install dependencies
pip install -r requirements.txt
```

### 8.2 Run Web Application
```bash
streamlit run app.py
```

Access at: http://localhost:8501

### 8.3 Run Live Monitor
```bash
# Single prediction
python live_monitor.py --image your_plot.png --once

# Continuous monitoring (every 60 minutes)
python live_monitor.py --plots-dir ./plots --interval 60
```

### 8.4 Test Data Fetcher
```bash
python live_data_fetcher.py
```

---

## 9. File Formats

### 9.1 Input Files

**Current Plots (.png):**
- Format: PNG image (RGB)
- Naming: `STRAITOFGEORGIAARRAY_YYYYMMDDThhmmss.sssZ.png`
- Content: Quiver/vector plot of ocean currents
- Dimensions: Varies (model resizes to 224×224)
- Example: `STRAITOFGEORGIAARRAY_20240709T090000.000Z.png`

**Raw Current Data (.tuv):**
- Format: Text file (not directly used by model, but source of plots)
- Content: Current velocity components (U, V)
- Grid-based spatial data

### 9.2 Output Files

**Model Checkpoint (.pth):**
```python
{
    'model_state_dict': OrderedDict,  # Model weights
    'label_map': {
        'OUTFLOW': 0,
        'TIDES': 1,
        'WIND/STORM': 2
    },
    'val_acc': 81.1,  # Validation accuracy
    'epoch': int,
    'optimizer_state_dict': OrderedDict  # (optional)
}
```

**Prediction Log (.csv):**
```csv
timestamp,prediction,confidence,prob_OUTFLOW,prob_TIDES,prob_WIND/STORM,wind_speed,tidal_level,fraser_level,image_file
2024-07-09 09:00:00,OUTFLOW,0.994,0.994,0.005,0.001,10.8,3.24,0.41,plot.png
```

---

## 10. API Integration

### 10.1 DFO IWLS Tidal API
**Endpoint:** https://api-iwls.dfo-mpo.gc.ca/api/v1/
**Authentication:** None (public)
**Rate Limits:** Reasonable use policy
**Response Format:** JSON

**Example Request:**
```
GET /api/v1/stations/7735/data?time-series-code=wlo&from=2024-01-01T00:00:00Z&to=2024-01-01T23:59:59Z
```

### 10.2 Water Office Canada
**Endpoint:** https://wateroffice.ec.gc.ca/services/real_time_data/json/
**Authentication:** None (public)
**Response Format:** JSON

**Example Request:**
```
GET /inline?stations[]=08MH053&parameters[]=47
```

### 10.3 Environment Canada Buoy Data
**Source:** MSC Datamart / SmartAtlantic ERDDAP
**Note:** Currently uses simulated data; production would parse SWOB-ML files

---

## 11. Future Enhancements

### 11.1 Model Improvements
- Address class imbalance (improve OUTFLOW detection)
- Incorporate temporal sequences (LSTM/Transformer)
- Multi-task learning (predict all forcing strengths simultaneously)
- Ensemble methods

### 11.2 Data Sources
- Direct integration with ONC (Ocean Networks Canada) current data
- Satellite imagery (SST, chlorophyll)
- Weather forecast integration
- Historical archive expansion

### 11.3 Features
- Mobile application
- Email/SMS alerts for storm conditions
- API endpoint for third-party integration
- Interactive map interface
- Uncertainty quantification

### 11.4 Operational
- Deploy to cloud (AWS/GCP/Azure)
- Real-time dashboard
- Database logging (PostgreSQL/TimescaleDB)
- User authentication
- Multi-region support

---

## 12. Known Limitations

1. **Class Imbalance:** OUTFLOW class only 7% of dataset → lower accuracy (68.6%)
2. **API Dependencies:** Live mode requires internet and functioning APIs
3. **Temporal Context:** Model treats each timestep independently (no memory)
4. **Geographic Scope:** Trained only on Strait of Georgia (not generalizable)
5. **Data Quality:** Relies on HF Radar availability (weather-dependent)
6. **Computational:** Requires significant resources for training (but inference is fast)

---

## 13. Scientific Basis

### 13.1 Physical Oceanography Principles
The classification is based on established oceanographic theory:

- **Estuarine Circulation:** Fraser River creates 2-layer flow (outflow)
- **Tidal Dynamics:** M2, S2, K1, O1 constituents drive semi-diurnal/diurnal tides
- **Ekman Transport:** Wind stress creates surface currents at 45° to wind direction
- **Coriolis Effect:** Deflects currents in rotating frame (Northern Hemisphere)

### 13.2 Validation
Model predictions align with:
- Known seasonal patterns (spring freshet)
- Tidal predictions from harmonic analysis
- Wind events from meteorological records

---

## 14. References & Acknowledgments

**Data Sources:**
- Ocean Networks Canada (ONC) - HF Radar
- Fisheries and Oceans Canada (DFO) - Tidal data
- Environment Canada - Weather/buoy data
- Water Office Canada - River discharge

**Frameworks:**
- PyTorch: https://pytorch.org/
- Streamlit: https://streamlit.io/
- timm: https://github.com/huggingface/pytorch-image-models

**Event:**
- OceanHack 2025, organized by UNESCO Ocean Decade

---

## 15. License & Contact

**Team:** The Bikini Bottom Team
**Event:** OceanHack 2025
**GitHub:** https://github.com/udbhav25kansal/Ocean-Hackathon-2025

---

## 16. Appendix: Example Prediction

**Input:**
- Image: `STRAITOFGEORGIAARRAY_20240709T090000.000Z.png`
- Wind Speed: 10.8 km/h
- Tidal Level: 3.24 m
- Fraser Level: 0.41 m
- Month: July (is_spring_freshet = 1)

**Output:**
```
Prediction: OUTFLOW
Confidence: 99.4%

Probabilities:
  OUTFLOW: 99.4%
  TIDES: 0.5%
  WIND/STORM: 0.1%

Interpretation:
Fraser River outflow detected! This is spring freshet season with
elevated river discharge. Critical for salmon migration and
estuarine water quality monitoring.
```

---

**Document Version:** 1.0
**Last Updated:** 2025-01-18
**Created with:** Claude Code (Anthropic)
