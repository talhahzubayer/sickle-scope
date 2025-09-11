# Contributing to SickleScope

Thank you for your interest in contributing to SickleScope! This document provides guidelines for contributing to the project.

## Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/sickle-scope.git
   cd sickle-scope
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e .[dev]
   ```

4. **Run Tests**
   ```bash
   python -m pytest tests/
   ```

### Development Guidelines

#### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Include docstrings for all public functions and classes
- Keep functions focused and modular
- Use type hints where appropriate

#### Testing
- Write tests for new features and bug fixes
- Maintain test coverage above 80%
- Use pytest for testing framework
- Include both unit tests and integration tests

#### Documentation
- Update README.md for significant changes
- Add docstrings to new functions and classes
- Update API documentation if needed
- Include examples for new features

## Types of Contributions

### Bug Reports
- Use the GitHub issue template
- Include steps to reproduce
- Provide system information
- Include sample data if relevant

### Feature Requests
- Describe the problem you're trying to solve
- Explain why this feature would be useful
- Consider the scope and complexity
- Discuss implementation approach if possible

### Code Contributions

#### Pull Request Process
1. Create a feature branch from main
2. Make your changes with clear, focused commits
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass
6. Submit a pull request with clear description

#### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense
- Keep first line under 50 characters
- Add detailed description if necessary

Example:
```
Add severity prediction validation

- Implement cross-validation for ML model
- Add confidence interval calculations
- Update visualisation to show uncertainty
```

### Areas for Contribution

#### High Priority
- Additional genetic variant databases
- Support for VCF file format
- Performance optimisations
- Extended population data
- Improved visualisation options

#### Medium Priority
- Additional machine learning models
- Web interface development
- Additional file format support
- Enhanced error handling
- Documentation improvements

#### Research Contributions
- Validation studies with clinical data
- Algorithm improvement suggestions
- New risk scoring approaches
- Population-specific analysis methods

## Development Standards

### Code Quality
- Use meaningful variable names
- Avoid code duplication
- Handle edge cases appropriately
- Include comprehensive error handling
- Follow the DRY (Don't Repeat Yourself) principle

### Performance
- Optimize for memory usage with large datasets
- Use vectorized operations where possible
- Profile code for performance bottlenecks
- Consider scalability for production use

### Security
- Never commit sensitive data or credentials
- Validate all user inputs
- Follow secure coding practices
- Be mindful of potential security implications

## Getting Help

- Join discussions in GitHub Issues
- Check existing documentation and examples
- Review the codebase for similar implementations
- Ask questions in the development discussions

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- GitHub contributors list
- Documentation credits where appropriate

## License

By contributing to SickleScope, you agree that your contributions will be licenced under the MIT Licence.

Thank you for helping make SickleScope better!