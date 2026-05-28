# Contributing to CFECT Quantum Engine

We welcome contributions to the CFECT Quantum Engine project! This document outlines our guidelines for contributing.

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
- Results must be verifiable against expected values
- Random number generators must be seeded appropriately
- Dependencies must be pinned to specific versions

### Documentation

- Update README.md for user-facing changes
- Add comments for complex algorithms
- Document any changes to the verification pipeline

## Testing

Run the full verification suite before submitting:

```bash
python reproduce_all.py
```

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