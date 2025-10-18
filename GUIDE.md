# Bayronik TUI Visualization Guide

**Learn how to read and interpret the baryonic field emulator interface**

---

## Quick Start

Launch the interactive TUI:
```bash
cd ~/Projects/bayronik
./run_bayronik.sh
```

Wait 10-15 seconds for loading, then you'll see three side-by-side panels showing matter distributions!

---

## Understanding the Interface

### **Main Layout**

```
┌──────────────────────────────────────────────────────────────┐
│ Bayronik: Baryonic Field Emulator                        │
│ Model: Trained on 1000 LH sims | Demo: 15 CV sims | ...     │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│ Controls: Simulation 5/15 | [←/→] Navigate | [r] Random     │
└──────────────────────────────────────────────────────────────┘
┌───────────────────┬───────────────────┬───────────────────┐
│ LEFT PANEL        │ CENTER PANEL      │ RIGHT PANEL       │
│ Input: Mcdm       │ Output: Mtot      │ Baryonic Δ       │
│ (Dark Matter)     │ (Total Matter)    │ (The Effect)      │
│ μ=25.78 σ=2.45    │ μ=25.97 σ=2.51    │ μ=0.18 σ=0.34     │
│                   │                   │                   │
│   ░░░▒▓█          │   ░░░▒▓█          │   ░░▒▒░           │
│   ░▒▓██▀          │   ░▒▓██▀          │   ▒▓█░░           │
│   ▒▓███▄          │   ▒▓███▄          │   ░█▒░░           │
│   ░░▒▓█           │   ░░▒▓█           │   ░░░░░           │
│                   │                   │                   │
└───────────────────┴───────────────────┴───────────────────┘
```

---

## 🔵 Left Panel: Input (Dark Matter - Mcdm)

### **What It Shows**
- **Cold dark matter** distribution from gravity-only N-body simulations
- The "skeleton" of cosmic structure
- Pure gravitational effects, NO baryonic physics

### **Physical Meaning**
- Represents ~**85% of all matter** in the universe
- Invisible to telescopes, only detected through gravity
- Sets the large-scale structure (filaments, halos, voids)

### **What to Look For**
- **Red/Yellow blobs** = Dark matter halos (future galaxies/clusters)
- **White filaments** = Cosmic web connecting halos
- **Dark gray regions** = Cosmic voids (empty space)

### **Statistics**
```
μ = 25.78  → Average log-density
σ = 2.45   → Structure variation (higher = more contrast)
⇵ [18, 32] → Range from voids to densest peaks
```

---

## 🟢 Center Panel: Output (Total Matter - Mtot)

### **What It Shows**
- **Total matter** = Dark matter + baryons (gas + stars)
- What the **ML model predicts** after learning baryonic physics
- What observers **actually measure** through gravitational lensing

### **Physical Meaning**
- Complete matter distribution including:
  - Dark matter (85%)
  - Gas (13%)
  - Stars (2%)
- Includes effects of:
  - Gas cooling
  - Star formation
  - Supernova feedback
  - AGN (black hole) feedback

### **What to Look For**
- **Sharper peaks** than left panel = gas concentrated in centers
- **Brighter halos** = baryons enhanced density
- **Slightly different structure** = baryonic effects visible

### **How It Differs from Input**
- Usually **similar** (dark matter dominates!)
- Subtle changes: ~10-30% density increase in halos
- Some halos may be **less dense** in centers (AGN feedback)

### **Statistics**
```
μ = 25.97  → Slightly higher than input (baryons add mass)
σ = 2.51   → Often slightly higher (baryons create sharper features)
```

**Typical increase**: μ(Mtot) - μ(Mcdm) = **0.1 to 0.3** (expected!)

---

## 🔴 Right Panel: Baryonic Effect (Δ)

### **What It Shows**
- **Δ = Mtot - Mcdm** (in log-space)
- The **pure baryonic correction**
- Where feedback processes **redistribute** matter

### **Physical Meaning**
This panel reveals where baryonic physics changes the matter distribution:
- **Positive Δ (bright)**: Baryons increase density
- **Negative Δ (dark)**: Baryons decrease density
- **Zero Δ (gray)**: No baryonic effect

### **What to Look For**

#### **Pattern 1: Cooling Flow** (Positive Center)
```
Δ panel shows: Bright blob ▓█
Physics: Gas cools and falls into halo center
Where: Low-mass halos (~10¹² M☉)
```

#### **Pattern 2: AGN Feedback** (Negative Center)
```
Δ panel shows: Dark center ░, bright edges
Physics: Black hole ejects gas from core
Where: Massive halos (>10¹³ M☉)
```

#### **Pattern 3: Filamentary Structure**
```
Δ panel shows: Linear bright features
Physics: Baryons trace cosmic web
Where: Along filaments connecting halos
```

### **Statistics Interpretation**

```
μ = 0.18  → Mean correction
```
**What this means in linear space**:
- Δμ = 0.18 → exp(0.18) = 1.20
- Total matter is **20% more** than dark matter
- **This is CORRECT!** (15-20% expected)

```
σ = 0.34  → Variation in correction
```
- Higher σ = more diverse baryonic effects
- Some regions enhanced, others depleted
- Sign of realistic physics!

```
⇵ [-2.5, +3.5]  → Range of effects
```
- Negative values = AGN blew gas out
- Positive values = Gas cooled and concentrated
- Large range = model captures both effects

---

## Color Scheme (All Panels)

All three panels use the **same color mapping** for matter density:

### **Color Table**

| Color | Symbol | Brightness | Density Range | Physical Region |
|-------|--------|------------|---------------|-----------------|
| **Dark Gray** | ░ | Darkest | 0-20% | Cosmic voids |
| **Gray** | ▒ | Dark | 20-40% | Underdense regions |
| **White** | ▓ | Medium | 40-60% | Cosmic web filaments |
| **Yellow** | █ | Bright | 60-80% | Dense halos, groups |
| **Red** | █ | Brightest | 80-100% | Massive galaxy clusters |

### **Enhanced Character Set**

For smoother gradients, we use **16 levels**:
```
░ ▒ ▓ █ ▀ ▄ ▌ ▐ ■ ▪ ● ◆ ◈ ★
```

Each character represents a specific density bin for fine detail.

### **Reading the Colors**

**In Input/Output Panels**:
- More red/yellow = denser regions (galaxy clusters)
- More white = average density (cosmic web)
- More gray = low density (voids)

**In Δ Panel** (difference):
- Red/Yellow = Positive correction (baryons added mass)
- White = Small correction (±10%)
- Gray/Dark = Negative correction (feedback removed mass)

---

## 📊 Statistics Bar Explained

Each panel title shows real-time statistics:

```
Input: Dark Matter (Mcdm) │ μ=25.784 σ=2.451 ⇵[18.23, 32.45]
```

### **μ (Mu) - Mean Value**
- **What it is**: Average log-density across entire 256×256 map
- **Typical range**: 18-30
- **Interpretation**:
  - μ ≈ 20: Low-density universe
  - μ ≈ 25: Typical cosmic density
  - μ ≈ 30: High-density structures

**Compare panels**:
- `μ(Mtot) > μ(Mcdm)` → Baryons add mass overall 
- `μ(Mtot) ≈ μ(Mcdm)` → Weak baryonic effects
- Difference should be **0.1 to 0.3** (10-30% increase)

### **σ (Sigma) - Standard Deviation**
- **What it is**: How much density varies across the map
- **Typical range**: 1.5-4.0
- **Interpretation**:
  - High σ: Strong structure (deep voids + dense clusters)
  - Low σ: Smooth distribution
  - σ of Δ panel: Strength of baryonic effects

### **⇵ [min, max] - Value Range**
- **What it is**: Minimum and maximum density values
- **Shows**: Dynamic range from voids to peaks
- **Interpretation**:
  - Large range (15+ difference): High contrast
  - Small range: Smooth field

**For Δ panel**:
- Negative min: AGN feedback present
- Positive max: Strong cooling flows
- Large range: Diverse baryonic physics

---

## Interactive Controls

### **Keyboard Commands**

| Key | Action | Description |
|-----|--------|-------------|
| `→` | Next | Move to next simulation |
| `←` | Previous | Go back one simulation |
| `n` | Next | Alternative to → |
| `p` | Previous | Alternative to ← |
| `r` | Random | Jump to random simulation |
| `q` | Quit | Exit application |

### **Navigation Tips**

**Sequential browsing** (→/←):
- Good for comparing similar parameter sets
- CV dataset: Same cosmology, different random seeds
- See how structure varies with different initial conditions

**Random jumping** (r):
- Explore diversity in dataset
- Find interesting examples quickly
- Compare very different structures

**Current simulation indicator**:
```
Simulation 5/15
```
Shows which of 15 local CV samples you're viewing.

---

## 🔬 Physical Interpretation Guide

### **Scenario 1: Typical Halo with Cooling**

**What you see**:
```
LEFT:   ███ (smooth red blob)
CENTER: ███ (brighter red blob)  
RIGHT:  ░█░ (bright center)
```

**Physics**:
1. Dark matter creates gravitational well (LEFT)
2. Gas falls in and cools (physics)
3. Density increases in center (CENTER)
4. Δ shows positive enhancement (RIGHT)

**Parameters**: Low-mass halo (~10¹² M☉), moderate cooling

---

### **Scenario 2: AGN Feedback Dominant**

**What you see**:
```
LEFT:   ███ (massive halo)
CENTER: █░█ (hollow center, dense edges)
RIGHT:  ░█░ (negative center, positive edges)
```

**Physics**:
1. Massive dark matter halo (LEFT)
2. Supermassive black hole heats gas (physics)
3. Gas ejected from center to outer regions (CENTER)
4. Δ negative in core, positive at edges (RIGHT)

**Parameters**: High-mass halo (>10¹³ M☉), strong AGN

---

### **Scenario 3: Cosmic Web Structure**

**What you see**:
```
ALL PANELS: Long white/yellow filaments connecting red nodes
```

**Physics**:
1. Matter flows along filaments to nodes
2. Baryons cool faster in denser regions
3. Enhanced contrast in filaments
4. Δ shows mild enhancement along web

**Parameters**: Large-scale structure, moderate feedback

---

### **Scenario 4: Weak Feedback (Similar Panels)**

**What you see**:
```
LEFT ≈ CENTER (almost identical)
RIGHT: Mostly gray with faint structure
```

**Physics**:
1. Low baryon fraction simulation
2. OR early redshift (z > 2)
3. OR weak feedback parameters
4. Baryons follow dark matter passively

**Parameters**: Weak AGN/SN feedback, or high-z

## Interpretation

### **Comparing Statistics Across Panels**

#### **Mean Ratio Analysis**:
```python
Δμ = μ(Mtot) - μ(Mcdm)
Linear ratio = exp(Δμ)
Baryon fraction = (ratio - 1) / ratio
```

**Example**:
```
μ(Mcdm) = 25.78
μ(Mtot) = 25.97
Δμ = 0.19
→ Linear ratio = exp(0.19) = 1.209
→ Baryon fraction = 0.209/1.209 = 17.3% 
```

#### **Variance Comparison**:
```
If σ(Mtot) > σ(Mcdm):
  → Baryons create sharper features
  → Cooling dominates in this simulation

If σ(Mtot) < σ(Mcdm):
  → Baryons smooth structure  
  → Feedback dominates
```

### **Reading the Δ Histogram** (mental model)

Imagine the Δ panel histogram:

```
     │     ╱╲
     │    ╱  ╲       Peak near 0 (most regions unaffected)
Count│   ╱    ╲      Long tail to right (cooling)
     │  ╱      ╲___  Short tail to left (feedback)
     │─┴────────────────→ Δ value
       -1  0   1   2
```

- **Symmetric**: Balanced cooling and feedback
- **Right-skewed**: Cooling dominant
- **Left-skewed**: Feedback dominant

---

## 🌟 Dataset Information

### **What You're Viewing**

**Model Training**:
- Trained on **1000 LH (Latin Hypercube) simulations**
- Varied cosmology: Ωₘ, σ₈, h
- Varied baryonic physics: A_AGN, A_SN

**Local Demo**:
- Viewing **15 CV (Cosmic Variance) simulations**
- Same cosmology, different random seeds
- Shows structure variation from initial conditions

**Why CV for demo?**:
- Smaller dataset (~300 MB vs 15 GB)
- Loads quickly on local machine
- Still shows diverse structures
- Model trained on LH works perfectly on CV!


## Further Reading

### **Understanding the Science**

- **CAMELS Project**: https://camels.readthedocs.io
- **Weak Lensing Review**: arXiv:1001.1739
- **Baryonic Effects**: arXiv:1403.4186
- **Field-Level Emulation**: arXiv:2211.09976

### **Technical Details**

- **U-Net Architecture**: arXiv:1505.04597
- **TorchScript**: pytorch.org/docs/stable/jit.html
- **Log-space Training**: Stabilizes learning for high dynamic range

### **Related Projects**

- **CAMELS-Multifield Emulator**: arXiv:2301.10515
- **GPemu**: Field-level emulator (arXiv:2408.10429)
- **CosmoPower**: Power spectrum emulator

---

## 🎓 Citation

If you use Bayronik in your research, please cite:

```bibtex
@software{bayronik2025,
  author = {Yuvraj Biswal},
  title = {Bayronik: Baryonic Field Emulator for Weak Lensing},
  year = {2025},
  url = {https://github.com/yourusername/bayronik}
}
```

---

## 📧 Contact

**Questions? Suggestions? Collaborations?**

- **Author**: Yuvraj Biswal
- **Email**: yuvrajbiswalofficial@gmail.com
- **Project**: Bayronik v1.0

---

**Last Updated**: October 2025  
**Status**: Production Ready  
**License**: MIT

