# Contributing to Diuretic-AKI Causal Inference Study

Thank you for your interest in contributing to this project! 

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Detailed description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python/R version)

### Submitting Code

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: brief description of changes"
   ```
   
   Use conventional commits:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for improvements
   - `Docs:` for documentation
   - `Test:` for test additions

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide a clear description
   - Reference related issues
   - Wait for review

## Code Style

### Python
- Follow PEP 8
- Use `black` for formatting: `black src/`
- Use type hints where appropriate
- Maximum line length: 88 characters

### R
- Follow tidyverse style guide
- Use meaningful variable names
- Comment complex logic

### Documentation
- Add docstrings to all functions
- Update README if needed
- Add inline comments for complex code

## Testing

- Write unit tests for new functions
- Ensure all tests pass before submitting
- Aim for >80% code coverage

## Pull Request Review Process

1. Maintainer reviews the PR
2. Discussions/changes requested if needed
3. Once approved, PR will be merged
4. Your contribution will be acknowledged

## Questions?

Feel free to open an issue for questions or reach out to:
- Email: lenhartkoo@foxmail.com

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the science and code quality
- Help maintain a positive environment

Thank you for contributing! 🎉
