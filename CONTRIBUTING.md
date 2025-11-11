# Contributing to VitalNet

## Important Notice

⚠️ **This repository is currently under peer review.**

The core VitalNet algorithms (Transformer-CNN fusion and MPC-based dosing) are **proprietary** and will be released after paper acceptance. This partial release is intended for:

1. Data preprocessing reproducibility
2. Feature extraction standardization
3. Evaluation metrics transparency
4. Community engagement

## What You Can Contribute

### ✅ Welcomed Contributions

- **Bug fixes** in preprocessing or feature extraction
- **Documentation improvements**
- **Additional evaluation metrics**
- **Data loading utilities** for other datasets
- **Visualization tools**
- **Unit tests** for existing modules
- **Examples and tutorials**

### ❌ Currently Not Accepting

- Model architecture implementations (proprietary)
- Training scripts for VitalNet core
- Drug dosing optimization algorithms

## How to Contribute

### Reporting Bugs

Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Python version and dependencies

### Suggesting Enhancements

Open an issue tagged "enhancement" with:
- Use case description
- Proposed solution
- Why it benefits the community

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Code Style

- Follow PEP 8
- Add docstrings (NumPy style)
- Include type hints where appropriate
- Comment complex logic

## Development Setup

```bash
git clone https://github.com/RegAItool/vitalnet_anesthesia.git
cd vitalnet_anesthesia
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

## Testing

```bash
# Run demos to verify setup
python examples/demo_preprocessing.py
```

## Questions?

- Open an issue for general questions
- Email: yu.han@eng.ox.ac.uk for research collaboration

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for your interest in VitalNet!**

We appreciate the community's support during the review process.
