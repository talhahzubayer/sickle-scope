# SickleScope Release Checklist

Use this checklist to ensure all necessary steps are completed before releasing a new version.

## Pre-Release Preparation

### Code Quality & Testing
- [ ] All tests pass locally (`python -m pytest tests/`)
- [ ] Code coverage is above 80%
- [ ] All linting checks pass (`flake8`, `black`)
- [ ] No critical security vulnerabilities identified
- [ ] Performance benchmarks are within acceptable ranges

### Documentation Updates
- [ ] README.md is up-to-date with new features
- [ ] CHANGELOG.md includes all changes for this release
- [ ] API documentation reflects current functionality
- [ ] Tutorial notebooks work with current version
- [ ] All docstrings are complete and accurate

### Version Management
- [ ] Version number updated in:
  - [ ] `setup.py`
  - [ ] `pyproject.toml` 
  - [ ] `sickle_scope/__init__.py`
  - [ ] CLI version option
- [ ] Version follows semantic versioning (MAJOR.MINOR.PATCH)
- [ ] Release date added to CHANGELOG.md

### Dependencies
- [ ] All dependencies are properly specified
- [ ] Minimum Python version requirements verified
- [ ] Optional dependencies clearly documented
- [ ] No unused dependencies in requirements

## Release Process

### Git Operations
- [ ] All changes committed to main branch
- [ ] Clean working directory (`git status`)
- [ ] Create release branch: `git checkout -b release/v0.1.0`
- [ ] Final commit with version bump
- [ ] Create and push git tag: `git tag v0.1.0`

### Package Building
- [ ] Clean previous builds: `rm -rf dist/ build/ *.egg-info/`
- [ ] Build source distribution: `python -m build --sdist`
- [ ] Build wheel distribution: `python -m build --wheel`
- [ ] Verify package contents: `twine check dist/*`
- [ ] Test installation in clean environment

### Testing Release
- [ ] Install from built package in fresh virtual environment
- [ ] Run basic functionality tests
- [ ] Verify CLI commands work correctly
- [ ] Test with sample data files
- [ ] Check that notebooks run without errors

## Publication

### PyPI Release
- [ ] Upload to Test PyPI: `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
- [ ] Test installation from Test PyPI
- [ ] Upload to Production PyPI: `twine upload dist/*`
- [ ] Verify package appears on PyPI with correct metadata

### GitHub Release
- [ ] Create GitHub release from tag
- [ ] Add release notes from CHANGELOG.md
- [ ] Attach distribution files (optional)
- [ ] Mark as pre-release if appropriate

### Documentation
- [ ] Update online documentation (if applicable)
- [ ] Announce release in README.md
- [ ] Update any external documentation or wikis

## Post-Release

### Verification
- [ ] Test `pip install sickle-scope` works correctly
- [ ] Verify package metadata on PyPI
- [ ] Check that GitHub release is properly formatted
- [ ] Confirm download counts are tracking

### Communication
- [ ] Update project status/badges if needed
- [ ] Share release announcement (if appropriate)
- [ ] Archive old versions in documentation

### Next Version Preparation
- [ ] Create new development branch
- [ ] Update version to next development version
- [ ] Add "Unreleased" section to CHANGELOG.md
- [ ] Update any version-specific documentation

## Emergency Procedures

### Rollback Process
If critical issues are discovered post-release:

1. **Immediate Response**
   - [ ] Document the issue in GitHub Issues
   - [ ] Assess severity and impact
   - [ ] Decide on hotfix vs. rollback

2. **Hotfix Release**
   - [ ] Create hotfix branch from release tag
   - [ ] Apply minimal fix
   - [ ] Follow abbreviated release process
   - [ ] Increment patch version

3. **PyPI Management**
   - [ ] Consider yanking problematic version if severe
   - [ ] Release fixed version immediately
   - [ ] Update documentation with migration notes

## Version History

| Version | Release Date | Status | Notes |
|---------|-------------|---------|-------|
| 0.1.0   | 11-09-2025  | ✅ Current | Initial release |

## Release Commands Reference

```bash
# Version tagging
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0

# Package building
python -m build

# PyPI upload
twine upload dist/*

# Testing installation
pip install --index-url https://test.pypi.org/simple/ sickle-scope
```

## Checklist Completion

**Release Manager:** _[Name]_  
**Release Date:** _[Date]_  
**Version:** _[Version Number]_

- [ ] All checklist items completed
- [ ] Release notes reviewed and approved
- [ ] Package successfully published
- [ ] Post-release verification completed

**Final Sign-off:** _[Signature/Approval]_