# TODO - Planned Features

This document provides a quick overview of features currently in development or planned for future implementation.

**Status Legend**: 🚧 In Progress | 🔜 Planned | ❓ Under Discussion

---

## High Priority

### 🚧 Improve Documentation

**Status**: In Progress

**Description**: Complete design documentation for all modules.

**Progress**:
- ✅ Architecture documentation (2025-03-08)
- ✅ Development guide (2025-03-08)
- ✅ Converters documentation (2025-03-08)
- ✅ Compute documentation (2025-03-08)
- ✅ Analysis documentation (2025-03-08)
- ✅ FAQ (2025-03-08)
- 🔜 User documentation improvements

**Expected Impact**:
- Better developer onboarding
- Easier for AI agents to understand codebase
- Clearer contribution guidelines

---

### 🔜 Add More DFT Converters

**Status**: Planned

**Description**: Add support for additional DFT codes.

**Planned Converters**:
- 🔜 VASP support (most requested)
- 🔜 PySCF support (molecular systems)
- 🔜 CP2K support
- 🔜 Quantum ESPRESSO improvements

**Expected Impact**:
- Broader user base
- More comprehensive DFT support

---

## Medium Priority

### 🔜 Performance Optimization

**Status**: Planned

**Description**: Improve performance for large-scale calculations.

**Planned Features**:
- 🔜 GPU acceleration for eigenvalue calculations
- 🔜 Streaming processing for large datasets
- 🔜 Memory-efficient matrix operations
- 🔜 Caching layer for intermediate results

**Expected Impact**:
- 2-5x speedup for large systems
- Reduced memory footprint
- Better scalability

---

### 🔜 Enhanced Analysis Tools

**Status**: Planned

**Description**: Add more analysis capabilities.

**Planned Features**:
- 🔜 Automated error diagnosis
- 🔜 Learning curve analysis
- 🔜 Feature importance analysis
- 🔜 Interactive visualization

**Expected Impact**:
- Better understanding of model performance
- Easier debugging
- More insights from data

---

### 🔜 Workflow Automation

**Status**: Planned

**Description**: Automate common workflows.

**Planned Features**:
- 🔜 End-to-end pipeline (DFT → training → prediction)
- 🔜 Automated hyperparameter tuning
- 🔜 Batch job management
- 🔜 Integration with HPC schedulers

**Expected Impact**:
- Reduced manual work
- Reproducible results
- Easier high-throughput calculations

---

## Future Considerations

### ❓ Plugin System

**Status**: Under discussion

**Description**: Allow external packages to extend DeepH-dock.

**Potential Impact**:
- Community contributions
- Modular architecture
- Easier customization

---

### ❓ Web Interface

**Status**: Under discussion

**Description**: Web-based interface for common tasks.

**Potential Impact**:
- Easier for non-experts
- Visual workflow builder
- Remote execution

---

### ❓ Machine Learning Integration

**Status**: Under discussion

**Description**: Direct integration with DeepH-pack training pipeline.

**Potential Impact**:
- Seamless workflow
- Automated data preparation
- Training monitoring

---

## Completed Features

- ✅ CLI auto-registration system
- ✅ Unified DeepH data format
- ✅ Multi-DFT converter support (SIESTA, OpenMX, FHI-aims, ABACUS, QE)
- ✅ Band structure calculation
- ✅ DOS calculation (Gaussian + Tetrahedron)
- ✅ Fermi level finding
- ✅ Ill-conditioned eigenvalue handling
- ✅ Multi-dimensional error analysis
- ✅ Dataset analysis tools
- ✅ Equivariance testing
- ✅ Parallel processing support
- ✅ Design documentation (2025-03-08)

---

## Contributing

To add a new planned feature:

1. Create a design document in `design/` directory
2. Set status to `🔜 Planned` or `🚧 In Progress`
3. Add an entry to this TODO.md
4. Update progress as implementation proceeds
5. Move to "Completed Features" when done

---

**Last Updated**: 2025-03-08

**Maintainer**: DeepH Team <deeph-pack@outlook.com>
