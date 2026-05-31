# Contributing to CFECT Engine

We welcome contributions to the CFECT Engine project! This document outlines our guidelines for contributing.

## Code of Conduct

By participating in this project, you agree to uphold our Code of Conduct:
- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize reproducibility and scientific integrity
- Document all changes clearly

## How to Contribute

### Reporting Issues

1. Check existing issues before submitting new ones
2. Include:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, dependencies)

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with proper documentation
4. Run tests to ensure reproducibility
5. Submit a pull request with:
   - Clear description of changes
   - References to related issues
   - Verification that all tests pass

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/methods
- Include unit tests for new functionality

### Reproducibility Requirements

All contributions must maintain the deterministic reproducibility guarantee:
- Results must be verifiable against expected values from real data
- Random number generators must be seeded where appropriate
- Dependencies must be pinned to specific versions

### Verification

Run the verification suite before submitting:

```bash
# Reproducibility harness (permutation test + sensitivity analysis)
python reproducibility_harness.py

# Sleep staging benchmark (requires real data)
python pipelines/run_three_gateways.py

# Spectral separation verification (requires real data)
python _verify_spectral_separation.py
```

## Data Access

This project uses publicly available datasets from PhysioNet:
- [CHB-MIT Scalp EEG](https://physionet.org/content/chbmit/)
- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/)
- [BUT-PDB ECG](https://physionet.org/content/butpdb/)
- [SDDB Holter](https://physionet.org/content/sddb/)

See `data/DATA_DOWNLOAD_GUIDE.md` for download instructions.

## Pull Request Review Process

1. Automated CI pipeline runs verification tests
2. Reviewers check for:
   - Code quality and style
   - Scientific validity
   - Reproducibility of results
   - Documentation completeness
3. Changes may require additional testing before merging

## License

By contributing to this project, you agree that your contributions will be licensed under the Apache-2.0 License.
