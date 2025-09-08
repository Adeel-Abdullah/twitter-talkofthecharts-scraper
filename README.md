# Chart Data Extraction & Processing Pipeline

A comprehensive Python pipeline for scraping Billboard Hot 100 chart prediction images from Twitter and converting them into structured, canonicalized datasets using OCR and MusicBrainz integration.

## 🎯 Project Overview

This project automates the extraction and processing of Billboard Hot 100 chart data from social media posts, converting image-based charts into clean, structured CSV datasets. The pipeline handles the complete workflow from data acquisition to final analysis-ready datasets.

### Key Features

- **🐦 Twitter Scraping**: Automated extraction of chart images from Twitter
- **🔍 OCR Processing**: GPU-accelerated text extraction from chart images  
- **🎵 Data Canonicalization**: MusicBrainz integration for authoritative song/artist names
- **⚡ Performance Optimized**: GPU acceleration, async processing, intelligent caching
- **📊 Data Quality**: Fuzzy matching, duplicate detection, comprehensive cleaning
- **☁️ Google Colab Ready**: Optimized for cloud execution with minimal setup

## 📋 Table of Contents

- [Pipeline Architecture](#-pipeline-architecture)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Google Colab Execution](#️-google-colab-execution)
- [Features & Capabilities](#-features--capabilities)
- [Output Examples](#-output-examples)
- [Performance Benchmarks](#-performance-benchmarks)
- [Troubleshooting](#-troubleshooting)
- [API Documentation](#-api-documentation)
- [Use Cases](#-use-cases)
- [Contributing](#-contributing)

## 🏗️ Pipeline Architecture

```
Twitter API → Image Scraper → Chart Images → OCR Processing → Raw Text Data
                                                                     ↓
Final Dataset ← MusicBrainz Canonicalization ← Fuzzy Standardization ← Data Cleaning

Data Processing Components:
├── Text Normalization (accent removal, punctuation cleanup)
├── Numeric Cleaning (K/M suffixes, OCR error correction)
└── Date Standardization (mm/dd/yy format)

Quality Enhancement:
├── Spelling Standardization (RapidFuzz similarity matching)
├── Authoritative Names (MusicBrainz database integration)
└── MBID Integration (unique identifiers for songs/artists)
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Twitter Developer Account (for API access)
- GPU recommended (for OCR acceleration)

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/chart-data-pipeline
cd chart-data-pipeline

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Colab Setup

```python
# Install dependencies in Colab
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install doctr[torch] pandas pillow python-dateutil openpyxl rapidfuzz aiohttp nest-asyncio

# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')
```

### Required Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
doctr[torch]>=0.6.0
pandas>=1.3.0
pillow>=8.0.0
python-dateutil>=2.8.0
openpyxl>=3.0.0
rapidfuzz>=2.0.0
aiohttp>=3.8.0
nest-asyncio>=1.5.0
twikit>=1.5.0
python-dotenv>=0.19.0
requests>=2.25.0
```

## 📖 Usage Guide

### 1. Twitter Data Scraping

Configure your Twitter API credentials:

```bash
# Create .env file
TWITTER_EMAIL=your_email@example.com
TWITTER_LOGIN_USERNAME=your_username
TWITTER_PASSWORD=your_password
```

Run the scraper:

```python
python pipeline.py
```

**Features:**
- Automatic checkpoint/resume functionality
- Rate limiting compliance
- Image filtering and date extraction
- Configurable date ranges (default: from Aug 10, 2023)

**Expected Output:**
```
charts_images/
├── 2025-08-10_1.jpg
├── 2025-08-10_2.jpg
├── 2025-08-17_1.jpg
└── 2025-08-17_2.jpg
```

### 2. OCR Processing

Process downloaded images:

```python
from OCR_batch_process_colab import run_ocr_batch_processing_colab

# Basic usage
result = run_ocr_batch_processing_colab()

# The function will:
# - Auto-detect GPU availability
# - Process all images in sorted_images/
# - Apply intelligent row/column detection
# - Generate clean CSV output
```

**Advanced Configuration:**
```python
# Custom processing with specific settings
result = process_directory_colab(
    input_dir="./charts_images",
    output_csv_name="custom_charts.csv",
    max_workers=1  # Sequential processing for memory efficiency
)
```

### 3. Data Canonicalization

Enhance data quality with MusicBrainz integration:

```python
from OCR_batch_process_colab import run_musicbrainz_canonicalization_colab

# Synchronous processing (reliable)
canonical_path = run_musicbrainz_canonicalization_colab(
    input_csv_path="final_merged_all_dates.csv",
    confidence_threshold=75,           # Overall match confidence
    individual_score_threshold=75,     # Both song AND artist must meet this
    enable_pre_filter=True,            # Filter non-music entries
    resume_from_checkpoint=True        # Resume interrupted runs
)

# Asynchronous processing (2-3x faster)
canonical_path = run_musicbrainz_canonicalization_colab_async(
    input_csv_path="final_merged_all_dates.csv",
    concurrent_requests=3,             # Number of concurrent API requests
    confidence_threshold=75,
    individual_score_threshold=75
)
```

## ☁️ Google Colab Execution

### Complete Workflow

#### 1. Setup Environment
```python
# IMPORTANT: Enable GPU runtime in Colab!
# Go to Runtime > Change runtime type > Hardware accelerator > GPU (T4/V100/A100)

# Verify GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### 2. Install Dependencies
```python
# Install with CUDA support for GPU acceleration
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install doctr[torch] pandas pillow python-dateutil openpyxl rapidfuzz aiohttp nest-asyncio
```

#### 3. Upload and Extract Data
```python
from google.colab import files
uploaded = files.upload()  # Upload your images zip file

import zipfile
for filename in uploaded.keys():
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('/content/')
```

#### 4. Run OCR Processing
```python
from OCR_batch_process_colab import run_ocr_batch_processing_colab

# Run the main OCR processing function
result = run_ocr_batch_processing_colab()

# Expected output files:
# - chart_dfs/final_merged_all_dates.csv
# - chart_dfs/final_merged_all_dates.xlsx
```

#### 5. Apply Canonicalization
```python
# OPTION 1: Synchronous (Standard, reliable)
canonical_path = run_musicbrainz_canonicalization_colab(
    input_csv_path="final_merged_all_dates.csv",
    confidence_threshold=75,           # Higher = fewer but more accurate matches
    individual_score_threshold=75,     # Both song AND artist must meet this
    enable_pre_filter=True,            # Filter out non-music entries
    resume_from_checkpoint=True        # Resume from previous run if interrupted
)

# OPTION 2: Asynchronous (2-3x faster)
canonical_path = run_musicbrainz_canonicalization_colab_async(
    input_csv_path="final_merged_all_dates.csv",
    concurrent_requests=3,             # Number of concurrent API requests (1-5)
    confidence_threshold=75,
    individual_score_threshold=75
)
```

#### 6. Download Results
```python
from google.colab import files
files.download('final_merged_all_dates_canonical.csv')
files.download('final_merged_all_dates_canonical.xlsx')
files.download('final_merged_all_dates_canonical_canonicalization.log')
```

### Expected Directory Structure
```
/content/
├── sorted_images/
│   ├── Aug 25/
│   │   ├── 2025-08-10_1.jpg
│   │   ├── 2025-08-10_2.jpg
│   │   └── ...
│   └── Aug 26/
│       └── ...
├── chart_dfs/          (created automatically)
│   ├── final_merged_all_dates.csv
│   └── final_merged_all_dates.xlsx
├── mb_checkpoints/     (MusicBrainz cache - sync version)
│   ├── api_cache.pkl
│   ├── canonical_cache.pkl
│   └── failed_queries.pkl
├── mb_checkpoints_async/  (MusicBrainz cache - async version)
│   ├── api_cache.pkl
│   ├── canonical_cache.pkl
│   └── failed_queries.pkl
├── final_merged_all_dates_canonical.csv      (sync output)
├── final_merged_all_dates_canonical_async.csv (async output)
└── *.log files (processing logs)
```

## ✨ Features & Capabilities

### OCR Processing
- **GPU Acceleration**: 3-10x performance improvement with CUDA support
- **Dynamic Row Detection**: Adapts to different chart layouts automatically
- **Smart Column Mapping**: Handles complex table structures with confidence
- **Text Normalization**: Comprehensive cleaning of OCR errors and inconsistencies

### Data Cleaning & Standardization
- **Text Normalization**: 
  - Lowercase conversion
  - Accent removal (á→a, ñ→n, ü→u, etc.)
  - Punctuation standardization
  - Whitespace cleanup
- **Music-Specific Rules**:
  - Featuring credit standardization: "featuring" → "feat.", "ft." → "feat."
  - Conjunction standardization: "&" → "and"
  - Remix/edit removal: "(Remix)", "(Radio Edit)", "(Live)", "(Acoustic)"
- **Numeric Processing**: 
  - K/M suffix handling (1.2K → 1200, 2.5M → 2500000)
  - OCR character fixes (S → 5, O → 0, I → 1)
- **Date Standardization**: Automatic conversion to mm/dd/yy format
- **Change Value Extraction**: Automatically extracts chart position changes from text

### Fuzzy Standardization (RapidFuzz)
- **Spelling Consistency**: Intelligent matching for OCR variations
- **Multi-level Approach**:
  - Artist standardization (80% threshold)
  - Song standardization by artist (85% threshold)
  - Combined song-artist standardization (85% threshold)
- **No Data Loss**: Preserves all original rows while improving consistency
- **Performance Optimized**: Processes large datasets efficiently

### MusicBrainz Canonicalization
- **Authoritative Names**: Official song and artist names from MusicBrainz database
- **Smart Search Strategies**: 8-tier search approach for maximum coverage:
  1. Partial matching without quotes (DEFAULT - OCR-friendly)
  2. Exact match with quotes (precise matching)
  3. OR search with quotes (broader net)
  4. OR search without quotes (very broad)
  5. Song-only exact search
  6. Artist-only exact search
  7. Song-only partial search
  8. Artist-only partial search
- **Intelligent Caching**: 
  - API response caching (threshold-independent)
  - Canonical result caching (threshold-specific)
  - Failed query tracking
- **Async Processing**: Concurrent API requests with rate limiting compliance
- **Quality Scoring**: Individual song/artist thresholds + combined confidence scoring

### Cache Optimization
- **API Cache Reuse**: Change thresholds without re-fetching data
- **Checkpoint Recovery**: Resume interrupted runs seamlessly
- **Smart Cache Management**: Automatic cleanup and optimization

## 📊 Output Examples

### Raw OCR Data
```csv
Chart_Date,Rank,Change,Song,Artist,Points,Percent,Peak,WoC,Sales,Sales %,Streams,Streams %,Airplay,Airplay %,Units
08/25/23,1,+2,Anti Hero,Taylor Swift,1000,100,1,15,450K,45,12.5M,55,8.2M,35,892K
08/25/23,2,-1,Flowers,Miley Cyrus,950,95,1,8,380K,40,11.8M,52,7.9M,33,795K
08/25/23,3,NEW,Unholy,Sam Smith feat. Kim Petras,920,92,3,2,290K,31,15.2M,66,6.1M,26,721K
```

### After Text Normalization
```csv
Chart_Date,Rank,Change,Song,Artist,Points,Percent,Peak,WoC,Sales,Sales %,Streams,Streams %,Airplay,Airplay %,Units
08/25/23,1,+2,anti hero,taylor swift,1000,100,1,15,450000,45,12500000,55,8200000,35,892000
08/25/23,2,-1,flowers,miley cyrus,950,95,1,8,380000,40,11800000,52,7900000,33,795000
08/25/23,3,NEW,unholy,sam smith feat. kim petras,920,92,3,2,290000,31,15200000,66,6100000,26,721000
```

### After Fuzzy Standardization
```csv
Chart_Date,Rank,Change,Song,Artist,Points,Percent,Peak,WoC,Sales,Sales %,Streams,Streams %,Airplay,Airplay %,Units
08/25/23,1,+2,anti hero,taylor swift,1000,100,1,15,450000,45,12500000,55,8200000,35,892000
08/25/23,2,-1,flowers,miley cyrus,950,95,1,8,380000,40,11800000,52,7900000,33,795000
08/25/23,3,NEW,unholy,sam smith feat. kim petras,920,92,3,2,290000,31,15200000,66,6100000,26,721000
```

### After MusicBrainz Canonicalization
```csv
Chart_Date,Rank,Change,Song,Artist,Points,Percent,Peak,WoC,Sales,Sales %,Streams,Streams %,Airplay,Airplay %,Units
08/25/23,1,+2,Anti-Hero,Taylor Swift,1000,100,1,15,450000,45,12500000,55,8200000,35,892000
08/25/23,2,-1,Flowers,Miley Cyrus,950,95,1,8,380000,40,11800000,52,7900000,33,795000
08/25/23,3,NEW,Unholy,Sam Smith feat. Kim Petras,920,92,3,2,290000,31,15200000,66,6100000,26,721000
```

### Processing Statistics
```
📊 CANONICALIZATION SUMMARY
=====================================
📊 Unique combinations: 2,547
✅ Successful matches: 2,156 (84.6%)
❌ Failed matches: 391 (15.4%)
📝 Rows updated: 12,847
📁 Output saved: final_merged_all_dates_canonical.csv
📁 Log saved: final_merged_all_dates_canonical_canonicalization.log
⚡ Processing time: 8.2 minutes (async)
💾 Cache entries: 2,547 API responses cached for reuse
```

## ⚡ Performance Benchmarks

### Processing Times (Google Colab with T4 GPU)

| Dataset Size | OCR Processing | Sync Canonicalization | Async Canonicalization |
|--------------|----------------|----------------------|------------------------|
| 1,000 songs  | ~2 minutes     | ~17 minutes (first run) | ~8 minutes (first run) |
|              |                | ~2 minutes (cached)     | ~1 minute (cached)     |
| 5,000 songs  | ~8 minutes     | ~1.4 hours (first run)  | ~35 minutes (first run)|
|              |                | ~8 minutes (cached)     | ~4 minutes (cached)    |
| 10,000 songs | ~15 minutes    | ~2.8 hours (first run)  | ~1.2 hours (first run) |
|              |                | ~15 minutes (cached)    | ~8 minutes (cached)    |

### GPU vs CPU Performance
- **OCR Processing**: 3-10x faster with GPU
- **Memory Usage**: Automatically managed with cache clearing
- **Concurrent Processing**: Async version 2-3x faster than sync

### Cache Performance
- **First Run**: Full API processing time
- **Subsequent Runs**: 90% time reduction with cache reuse
- **Threshold Changes**: Instant re-evaluation using cached API data
- **Cache Hit Rate**: Typically 95%+ on repeated processing

### API Efficiency
- **MusicBrainz Requests**: Up to 50 results per query
- **Result Comparison**: Up to 25 results analyzed per song
- **Rate Limiting**: 1.1-1.2 second delays for compliance
- **Success Rate**: 80-90% match rate depending on data quality

## 🔧 Troubleshooting

### Common Issues & Solutions

#### GPU Not Detected
```python
# Verify GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Solution: Change Colab runtime to GPU
# Runtime > Change runtime type > Hardware accelerator > GPU
```

#### OCR Processing Errors
```bash
# Install correct CUDA version
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify DocTR installation
from doctr.models import ocr_predictor
model = ocr_predictor(pretrained=True)
print("✅ DocTR working correctly")
```

#### Memory Issues
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Reduce batch size or use sequential processing
result = process_directory_colab(max_workers=1)
```

#### MusicBrainz Rate Limiting
```python
# Reduce concurrent requests
canonical_path = run_musicbrainz_canonicalization_colab_async(
    concurrent_requests=1  # More conservative
)

# Use sync version for unstable connections
canonical_path = run_musicbrainz_canonicalization_colab()
```

#### Low Match Rates
```python
# Lower thresholds for more permissive matching
canonical_path = run_musicbrainz_canonicalization_colab(
    confidence_threshold=65,           # Lower from 75
    individual_score_threshold=65      # Lower from 75
)

# Enable pre-filtering to focus on music entries
canonical_path = run_musicbrainz_canonicalization_colab(
    enable_pre_filter=True
)
```

#### Import Errors
```python
# Check all dependencies
!pip list | grep -E "(torch|doctr|rapidfuzz|aiohttp)"

# Reinstall if needed
!pip install --upgrade torch torchvision doctr rapidfuzz aiohttp
```

### Debug Mode
```python
# Enable detailed logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check processing logs
with open('final_merged_all_dates_canonical_canonicalization.log', 'r') as f:
    print(f.read())
```

### Performance Optimization
```python
# For large datasets, process in smaller chunks
# Split your data and process incrementally

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Monitor GPU memory
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
```

## 📚 API Documentation

### Core Functions

#### `run_ocr_batch_processing_colab()`
Processes chart images with OCR and creates structured datasets.

**Returns**: Final merged DataFrame with all processed charts
**Files Created**: 
- `chart_dfs/final_merged_all_dates.csv`
- `chart_dfs/final_merged_all_dates.xlsx`

#### `run_musicbrainz_canonicalization_colab()`
Synchronous canonicalization with MusicBrainz integration.

**Parameters**:
- `input_csv_path` (str): Path to input CSV file (default: "final_merged_all_dates.csv")
- `output_csv_path` (str): Path to output CSV file (auto-generated if None)
- `confidence_threshold` (int): Overall match confidence 0-100 (default: 75)
- `individual_score_threshold` (int): Minimum song/artist score 0-100 (default: 75)
- `enable_pre_filter` (bool): Filter non-music entries (default: True)
- `resume_from_checkpoint` (bool): Resume interrupted runs (default: True)

**Returns**: Path to canonicalized CSV file

#### `run_musicbrainz_canonicalization_colab_async()`
Asynchronous canonicalization for faster processing.

**Additional Parameters**:
- `concurrent_requests` (int): Concurrent API requests 1-5 (default: 3)

**Performance**: 2-3x faster than synchronous version

### Configuration Parameters

#### Confidence Thresholds
- **75-85%**: Balanced accuracy and coverage (recommended)
- **85-95%**: High accuracy, lower coverage
- **65-75%**: High coverage, moderate accuracy

#### Individual Score Thresholds
- **75%**: Both song AND artist must meet this threshold
- **Lower values**: More permissive matching
- **Higher values**: Stricter quality requirements

#### Concurrent Requests (Async Only)
- **1-2**: Conservative, stable connections
- **3**: Recommended balance
- **4-5**: Aggressive, requires stable internet

### Cache Management

The pipeline uses intelligent caching:

#### API Cache (`api_cache.pkl`)
- Stores raw MusicBrainz responses
- Threshold-independent
- Reused across different confidence settings

#### Canonical Cache (`canonical_cache.pkl`)
- Stores processed results
- Threshold-specific
- Rebuilt when thresholds change

#### Failed Queries (`failed_queries.pkl`)
- Tracks queries that returned no results
- Avoids re-querying known failures
- Improves performance on subsequent runs

### Error Handling

The pipeline includes comprehensive error handling:
- Network timeouts and retries
- API rate limiting compliance
- GPU memory management
- Graceful degradation for missing dependencies

## 🎯 Use Cases

### Music Industry Analysis
- **Chart Performance Tracking**: Monitor song positions over time
- **Artist Career Analysis**: Track artist success and trends
- **Genre Trend Identification**: Analyze genre popularity shifts
- **Market Share Analysis**: Compare label or distributor performance

### Academic Research
- **Music Consumption Patterns**: Study how music preferences change
- **Cultural Trend Analysis**: Examine cultural shifts through music
- **Data Quality Studies**: Research OCR and data cleaning methodologies
- **Natural Language Processing**: Study music text normalization techniques

### Business Intelligence
- **Competitive Analysis**: Monitor competitor artist performance
- **Marketing Insights**: Identify trending topics and artists
- **Streaming Optimization**: Understand streaming vs. sales patterns
- **A&R Decision Support**: Data-driven artist signing decisions

### Data Science Projects
- **Machine Learning Datasets**: Clean, structured music data for ML
- **Recommendation Systems**: Historical chart data for recommendation engines
- **Predictive Analytics**: Forecast chart performance
- **Sentiment Analysis**: Combine with social media data

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/chart-data-pipeline
cd chart-data-pipeline

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Contribution Process
1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/enhancement`)
3. **Make your changes** with appropriate tests
4. **Run tests** (`python -m pytest tests/`)
5. **Commit changes** (`git commit -am 'Add new feature'`)
6. **Push to branch** (`git push origin feature/enhancement`)
7. **Open a Pull Request**

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints where applicable
- Write tests for new functionality
- Update documentation for changes

### Areas for Contribution
- Additional OCR engines (Tesseract, EasyOCR)
- New data sources (Spotify, Apple Music)
- Enhanced fuzzy matching algorithms
- Performance optimizations
- Additional export formats
- UI/Web interface

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **MusicBrainz**: CC0 1.0 Universal (Public Domain)
- **DocTR**: Apache License 2.0
- **RapidFuzz**: MIT License
- **PyTorch**: BSD 3-Clause License

## 🙏 Acknowledgments

- **[MusicBrainz](https://musicbrainz.org/)**: The authoritative music database that powers our canonicalization
- **[DocTR](https://github.com/mindee/doctr)**: High-performance OCR framework for document analysis
- **[RapidFuzz](https://github.com/maxbachmann/RapidFuzz)**: Fast and accurate fuzzy string matching
- **[Google Colab](https://colab.research.google.com/)**: Free GPU access for development and testing
- **[Twikit](https://github.com/d60/twikit)**: Reliable Twitter scraping capabilities

## 📞 Contact & Portfolio

This project demonstrates expertise in:
- **Data Engineering**: End-to-end pipeline development
- **Machine Learning**: OCR, NLP, and fuzzy matching
- **API Integration**: Rate-limited external service integration
- **Performance Optimization**: GPU acceleration and async processing
- **Cloud Computing**: Google Colab optimization and deployment

### Professional Contact
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **Portfolio**: [Your Portfolio Website](https://yourportfolio.com)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

### Project Metrics
- **Languages**: Python
- **Frameworks**: PyTorch, DocTR, RapidFuzz
- **APIs**: MusicBrainz, Twitter
- **Performance**: 10x+ speedup with optimizations
- **Scale**: Handles 10,000+ songs efficiently
- **Accuracy**: 80-90% successful canonicalization rate

---

*This pipeline was developed to demonstrate practical applications of modern data engineering techniques, including OCR, natural language processing, API integration, and cloud computing. It showcases the ability to transform unstructured data sources into analysis-ready datasets while maintaining data quality and processing efficiency.*

## 🔄 Version History

- **v1.0.0** - Initial release with basic OCR and Twitter scraping
- **v1.1.0** - Added MusicBrainz canonicalization
- **v1.2.0** - Implemented fuzzy standardization with RapidFuzz
- **v1.3.0** - Added async processing and performance optimizations
- **v1.4.0** - Enhanced Google Colab integration and GPU acceleration
- **v1.5.0** - Improved search strategies and partial matching for OCR errors

*Latest updates focus on maximizing match rates for OCR data through intelligent search strategies and comprehensive result comparison.*
