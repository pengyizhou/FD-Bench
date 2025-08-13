# FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems

<div align="center">

[![Demo Page](https://img.shields.io/badge/Demo-Page-blue)](https://pengyizhou.github.io/FD-Bench/)
[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2507.19040)
[![Hugging Face](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/collections/pengyizhou/fd-bench-audio-68674bd6de6feea91ba3ce37)
[![License](https://img.shields.io/badge/License-NTUitive-green.svg)](LICENSE.txt)

</div>

## 📖 Abstract

FD-Bench is a comprehensive benchmarking pipeline specifically designed for evaluating Full-Duplex Spoken Dialogue Systems (FD-SDS). This benchmark provides standardized metrics and evaluation protocols to assess the performance of conversational AI systems in real-time, bidirectional communication scenarios.

## 🚀 Quick Links

- **🌐 [Demo Page](https://pengyizhou.github.io/FD-Bench/)** - More results
- **📄 [ArXiv Paper](https://arxiv.org/abs/2507.19040)** - Detailed methodology and findings
- **🤗 [Hugging Face Dataset](https://huggingface.co/collections/pengyizhou/fd-bench-audio-68674bd6de6feea91ba3ce37)** - Download the benchmark dataset

## ✅ Released
- [x] Inference sample released on the Demo website
- [x] All benchmarking metrics for all generated datasets and all FD-SDS models we have tested
- [x] All delay distributions for all generated datasets and all FD-SDS models we have tested
- [x] Release of code for generating simulated TTS data
  - [x] OpenAI API inference
  - [x] TTS models inference
- [x] Llama3 PPL calculation
- [x] Dataset release to Hugging Face

## 🏗️ Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.5.0
- CUDA >= 12.4

### Setup
```bash
git clone https://github.com/pengyizhou/FD-Bench.git
cd FD-Bench
```

## 📊 Dataset

FD-Bench includes comprehensive evaluation data for full-duplex spoken dialogue systems:

- **Objective Metrics**: WER, BLEU, response time, interruption handling
- **Subjective Metrics**: Naturalness, coherence, turn-taking appropriateness
- **Test Scenarios**: Various conversation types and interruption patterns

### Download Dataset
```bash
# Download from Hugging Face
huggingface-cli download pengyizhou/FD-Bench-Audio-Input --local-dir ./data
```

## 🔧 Usage

### TODO
[ ] Missing Model inference code with timestamp generation..

### Generating TTS Data
```bash
# Using ChatTTS
cd tts-generation/ChatTTS
python generate_using_chattts.py

# Using CosyVoice2
cd tts-generation/CosyVoice2
python generate_using_cosyvoice2.py

# Using F5TTS
cd tts-generation/F5TTS
python generate_using_f5tts.py
```

## 📈 Results

Our benchmark evaluates various state-of-the-art FD-SDS models:

- **Freeze-omni**: Performance metrics and analysis
- **Moshi**: Objective and subjective evaluation results
- **VITA-1.5**: Comprehensive benchmarking scores

Detailed results and comparisons are available on our [Demo Page](https://pengyizhou.github.io/FD-Bench/).

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on how to submit improvements, bug reports, or new features.

## 📜 License

This project is licensed under the NTUitive License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 📚 Citation

If you use FD-Bench in your research, please cite our paper:

```bibtex
@article{fd-bench2024,
  title={FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🙏 Acknowledgments

- Thanks to all contributors and the open-source community
- Special acknowledgments to the teams behind the evaluated models
- Funding and institutional support acknowledgments

## 📞 Contact

For questions or collaboration opportunities, please reach out:
- Email: yizhou004@e.ntu.edu.sg
- GitHub Issues: [Submit an issue](https://github.com/pengyizhou/FD-Bench/issues)

---

<div align="center">
Made with ❤️ by the FD-Bench Team
</div>
