# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in SickleScope, please report it responsibly:

### How to Report

1. **Do not open a public GitHub issue** for security vulnerabilities
2. Send an email to: **talhahzubayer101@gmail.com**
3. Include "SickleScope Security" in the subject line
4. Provide detailed information about the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix and Release**: Within 2-4 weeks (depending on complexity)

## Security Considerations

### Data Handling
SickleScope processes genetic data which may be sensitive. We are committed to:

- **No Data Transmission**: All analysis is performed locally
- **No Data Storage**: No genetic data is stored by default
- **Input Validation**: All user inputs are validated
- **Error Handling**: Sensitive information is not exposed in error messages

### Best Practices for Users

#### Secure Data Handling
- Store genetic data files securely
- Use appropriate file permissions
- Consider encryption for sensitive datasets
- Follow your organisation's data governance policies

#### Environment Security
- Keep Python and dependencies updated
- Use virtual environments
- Verify package integrity before installation
- Be cautious with notebook sharing

#### Input Data Safety
- Validate data sources before analysis
- Be aware of potential data quality issues
- Use the built-in validation features
- Review results for reasonableness

### Known Security Considerations

#### File Processing
- Input files are parsed using pandas - ensure files are from trusted sources
- Large files may consume significant memory
- File validation helps detect format issues

#### Dependencies
- SickleScope relies on several third-party packages
- Regular updates help maintain security
- Development dependencies include additional tools

#### Output Files
- Generated reports may contain sensitive genetic information
- HTML reports include embedded data - share carefully
- Consider output file permissions and storage location

## Security Updates

Security updates will be:
- Released as patch versions when possible
- Documented in CHANGELOG.md
- Announced through GitHub releases
- Tagged as security releases when applicable

## Scope

This security policy covers:
- SickleScope core package code
- Command-line interface
- Data processing functions
- Output generation features

This policy does not cover:
- Third-party dependencies (report to respective maintainers)
- User-specific configurations or environments
- External data sources or databases

## Contact

For security-related questions or concerns:
- Email: talhahzubayer101@gmail.com
- Use "SickleScope Security" in subject line
- Expect response within 48 hours

Thank you for helping keep SickleScope secure!