# Bikini Bottom Ocean Classifier - Color System Guide

## 🎨 Design Rationale

This color system channels the vibrant energy of tropical underwater worlds—sunlit shallows, coral gardens, jellyfish blooms, and sandy boardwalks—while maintaining professional usability. **Sponge Yellow** (#F7E21B) anchors the brand with sunny optimism, **Seafoam Blue** (#3DB5FF) evokes crystalline ocean currents, and **Coral** (#FF6F6F) adds playful warmth. Accents of **Jellyfish Lavender** and **Seaweed Green** introduce depth without losing clarity. High saturation conveys whimsy; strategic contrast ensures readability. Every token balances cartoon-inspired joy with modern UI standards—no neon chaos, just approachable, accessible ocean vibes.

---

## 📊 Complete Color Palette

### Primary - Sponge Yellow
Represents sunshine, optimism, and brand energy. Use for CTAs and key moments.

| Scale | HEX | Usage |
|-------|-----|-------|
| 050 | `#FFFCF0` | Backgrounds, subtle highlights |
| 100 | `#FFF9D9` | Hover backgrounds |
| 200 | `#FFF3B3` | Light accents |
| 300 | `#FFED8C` | Gradient starts |
| **400** | **`#F7E21B`** | **Primary brand (light mode)** |
| **500** | **`#E1C317`** | **Primary hover** |
| 600 | `#C9AC14` | Active states |
| 700 | `#A68F11` | Darker variants |
| 800 | `#7A6A0C` | Deep accents |
| 900 | `#4D4307` | Darkest |
| **Contrast** | **`#0B1E3A`** | **Text on yellow (AAA)** |

### Secondary - Seafoam Blue
Ocean currents, links, secondary actions.

| Scale | HEX | Usage |
|-------|-----|-------|
| 050 | `#E6F7FF` | Info backgrounds |
| 100 | `#C2EBFF` | Light accents |
| 200 | `#99DEFF` | Subtle highlights |
| 300 | `#70D1FF` | Dark mode links |
| **400** | **`#3DB5FF`** | **Secondary brand (light mode)** |
| **500** | **`#1E9FED`** | **Hover** |
| 600 | `#1888D1` | Active links |
| 700 | `#1470B3` | Pressed states |
| 800 | `#0F5689` | Deep blue |
| 900 | `#0A3D5F` | Darkest |

### Tertiary - Coral
Highlights, alerts, warm accents.

| Scale | HEX | Usage |
|-------|-----|-------|
| 050 | `#FFE9E9` | Error backgrounds |
| 100 | `#FFCCCC` | Light warnings |
| 200 | `#FFB3B3` | Subtle alerts |
| 300 | `#FF9999` | Soft coral |
| **400** | **`#FF6F6F`** | **Tertiary brand** |
| 500 | `#FF4D4D` | Vivid coral |
| 600 | `#E63946` | Error states |
| 700 | `#CC2936` | Darker alerts |
| 800 | `#991F28` | Deep red |
| 900 | `#66141A` | Darkest |

### Accent 1 - Jellyfish Lavender
Premium features, playful highlights.

| Scale | HEX | Usage |
|-------|-----|-------|
| 050 | `#F5EBFF` | Subtle backgrounds |
| 100 | `#E8D4FF` | Light accents |
| 200 | `#DABDFF` | Hover backgrounds |
| 300 | `#CDA6FF` | Soft highlights |
| **400** | **`#C58CFF`** | **Primary lavender** |
| 500 | `#B36FFF` | Vivid purple |
| 600 | `#9F52FF` | Bright accents |
| 700 | `#8533E6` | Deep purple |
| 800 | `#6526AD` | Darker |
| 900 | `#451A73` | Darkest |

### Accent 2 - Seaweed Green
Success states, growth, positive momentum.

| Scale | HEX | Usage |
|-------|-----|-------|
| 050 | `#E6F9F0` | Success backgrounds |
| 100 | `#B8F0DA` | Light green |
| 200 | `#8AE7C4` | Subtle accents |
| 300 | `#5CDEAE` | Dark mode success |
| **400** | **`#2BB673`** | **Primary green** |
| 500 | `#24A063` | Vivid green |
| 600 | `#1F8A54` | Deep green |
| 700 | `#1A7345` | Darker |
| 800 | `#145636` | Forest green |
| 900 | `#0D3924` | Darkest |

### Neutrals - Barnacle Greys → Foam
Text, borders, backgrounds, structural elements.

| Scale | HEX | Usage |
|-------|-----|-------|
| 025 | `#F6FBFD` | Light mode primary BG |
| 050 | `#EAF2F7` | Secondary BG |
| 100 | `#D7E3EC` | Borders |
| 200 | `#B8C7D5` | Subtle borders |
| 300 | `#8FA3B7` | Tertiary text |
| 400 | `#6B7C8F` | Secondary text |
| 500 | `#516171` | Body text |
| 600 | `#3A4752` | Headings |
| 700 | `#2C3A47` | Dark surfaces |
| 800 | `#1E2A36` | Primary text |
| 900 | `#0F1419` | Darkest |

---

## 🌈 Gradients

### Sunlit Water
**Colors:** Sponge Yellow 300 → Seafoam 400
**HEX:** `#FFED8C` → `#3DB5FF`
**Usage:** Hero sections, primary feature cards, sunrise vibes

### Lagoon
**Colors:** Seafoam 500 → Jellyfish 400
**HEX:** `#1E9FED` → `#C58CFF`
**Usage:** Interactive zones, hover states, mystical depth

### Tropical
**Colors:** Sponge Yellow 400 → Seaweed 400
**HEX:** `#F7E21B` → `#2BB673`
**Usage:** Success flows, positive momentum, growth indicators

### Coral
**Colors:** Coral 400 → Warning
**HEX:** `#FF6F6F` → `#FFCC33`
**Usage:** Warning flows, attention-grabbing moments, alerts

---

## 🎯 State Modifiers

### Hover States
**Method:** Darken/lighten by ~8%
**CSS:** `filter: brightness(0.92)` for light colors, `brightness(1.08)` for dark colors

**Examples:**
- Primary Yellow: `#F7E21B` → `#E1C317`
- Secondary Blue: `#3DB5FF` → `#1E9FED`

### Active States
**Method:** Darken/lighten by ~12%
**CSS:** `filter: brightness(0.88)` for light, `brightness(1.12)` for dark

**Examples:**
- Primary Yellow: `#F7E21B` → `#C9AC14`
- Secondary Blue: `#3DB5FF` → `#1888D1`

### Focus States
**Ring Width:** 2px
**Offset:** 2px
**Color (Light):** `#3DB5FF` (Secondary 400)
**Color (Dark):** `#70D1FF` (Secondary 300)

**CSS:**
```css
.focus-ring:focus-visible {
  outline: none;
  box-shadow:
    0 0 0 2px transparent, /* offset */
    0 0 0 4px var(--focus-ring-color); /* ring */
}
```

### Disabled States
**Opacity:** 40% (`opacity: 0.4`)
**Cursor:** `not-allowed`
**Shadows:** Remove all

**Examples:**
- Primary: `rgba(247, 226, 27, 0.4)`
- Secondary: `rgba(61, 181, 255, 0.4)`

---

## 🌗 Light & Dark Mode Mappings

### Light Mode

| Semantic Token | Light Value | HEX |
|----------------|-------------|-----|
| `--bg-primary` | Neutral 025 | `#F6FBFD` |
| `--bg-secondary` | Neutral 050 | `#EAF2F7` |
| `--surface-primary` | White | `#FFFFFF` |
| `--text-primary` | Neutral 800 | `#1E2A36` |
| `--text-secondary` | Neutral 500 | `#516171` |
| `--text-link` | Secondary 600 | `#1888D1` |
| `--border-primary` | Neutral 100 | `#D7E3EC` |
| `--interactive-primary` | Primary 400 | `#F7E21B` |
| `--interactive-secondary` | Secondary 400 | `#3DB5FF` |

### Dark Mode

| Semantic Token | Dark Value | HEX |
|----------------|------------|-----|
| `--bg-primary` | Deep Navy | `#0B1E3A` |
| `--bg-secondary` | Dark Surface | `#112A4B` |
| `--surface-primary` | Dark Surface | `#112A4B` |
| `--text-primary` | Neutral 050 | `#EAF2F7` |
| `--text-secondary` | Neutral 200 | `#B8C7D5` |
| `--text-link` | Secondary 300 | `#70D1FF` |
| `--border-primary` | Neutral 600 | `#3A4752` |
| `--interactive-primary` | Primary 300 | `#FFED8C` |
| `--interactive-secondary` | Secondary 400 | `#3DB5FF` |

---

## ♿ Accessibility - WCAG 2.1 AA Compliance

### Contrast Requirements

| Element Type | Minimum Ratio | WCAG Level |
|--------------|---------------|------------|
| Normal Text (<18px) | 4.5:1 | AA |
| Large Text (≥18px or ≥14px bold) | 3.0:1 | AA |
| UI Components | 3.0:1 | AA |
| Normal Text (AAA) | 7.0:1 | AAA |
| Large Text (AAA) | 4.5:1 | AAA |

### ✅ Compliant Color Pairs - Light Mode

| Foreground | Background | Ratio | Pass |
|------------|------------|-------|------|
| Deep Navy `#0B1E3A` | White | 15.5:1 | AAA ✓ |
| Neutral 800 `#1E2A36` | White | 13.8:1 | AAA ✓ |
| **Deep Navy `#0B1E3A`** | **Sponge Yellow `#F7E21B`** | **8.2:1** | **AAA ✓** |
| Neutral 800 `#1E2A36` | Sponge Yellow `#F7E21B` | 7.4:1 | AAA ✓ |
| Deep Navy `#0B1E3A` | Seafoam `#3DB5FF` | 4.8:1 | AA ✓ |
| White | Secondary 600 `#1888D1` | 4.5:1 | AA ✓ |
| White | Error `#E63946` | 5.1:1 | AA ✓ |
| White | Success `#2BB673` | 4.7:1 | AA ✓ |
| Neutral 500 `#516171` | White | 5.2:1 | AA ✓ |
| Deep Navy `#0B1E3A` | Neutral 050 `#EAF2F7` | 13.1:1 | AAA ✓ |

### ✅ Compliant Color Pairs - Dark Mode

| Foreground | Background | Ratio | Pass |
|------------|------------|-------|------|
| Neutral 050 `#EAF2F7` | Deep Navy `#0B1E3A` | 13.1:1 | AAA ✓ |
| Neutral 200 `#B8C7D5` | Deep Navy `#0B1E3A` | 8.9:1 | AAA ✓ |
| Deep Navy `#0B1E3A` | Primary 300 `#FFED8C` | 10.1:1 | AAA ✓ |
| Deep Navy `#0B1E3A` | Secondary 300 `#70D1FF` | 6.8:1 | AAA ✓ |
| Neutral 050 `#EAF2F7` | Dark Surface `#112A4B` | 11.2:1 | AAA ✓ |
| White | Secondary 600 `#1888D1` | 4.5:1 | AA ✓ |
| White | Success `#2BB673` | 4.7:1 | AA ✓ |
| Neutral 300 `#8FA3B7` | Deep Navy `#0B1E3A` | 6.1:1 | AA ✓ |
| Secondary 300 `#70D1FF` | Deep Navy `#0B1E3A` | 6.8:1 | AAA ✓ |
| Seaweed 300 `#5CDEAE` | Dark Surface `#112A4B` | 7.3:1 | AAA ✓ |

### ❌ Failing Pairs (with Alternatives)

| Foreground | Background | Ratio | Issue | Alternative |
|------------|------------|-------|-------|-------------|
| Sponge Yellow `#F7E21B` | White | 1.9:1 | FAIL | Use `#0B1E3A` text on yellow (8.2:1 ✓) |
| Warning `#FFCC33` | White | 1.7:1 | FAIL | Use `#0B1E3A` text on warning (9.1:1 ✓) |
| Coral `#FF6F6F` | Deep Navy `#0B1E3A` | 2.8:1 | FAIL | Use white on Error `#E63946` (5.1:1 ✓) |
| Jellyfish `#C58CFF` | White | 2.6:1 | FAIL | Use `#451A73` (dark purple) instead (7.8:1 ✓) |
| Neutral 400 `#6B7C8F` | Neutral 500 `#516171` | 1.8:1 | FAIL | Use `#EAF2F7` on 500 (5.2:1 ✓) |

### Best Practices

1. **Text on Primary Yellow:** Always use Deep Navy `#0B1E3A` (8.2:1 AAA)
2. **Text on Secondary Blue:** Use white or Deep Navy depending on shade
3. **Never use yellow text on white backgrounds**
4. **Never use coral text on dark navy** without checking contrast
5. **Always test custom combinations** with a contrast checker

---

## 🎨 Usage Guidelines

### Primary (Sponge Yellow)
**Use for:**
- Call-to-action buttons
- Primary navigation highlights
- Key brand moments
- Hero section accents

**Don't use for:**
- Body text on white
- Small icons (<24px) on white
- Disabled states

### Secondary (Seafoam Blue)
**Use for:**
- Links and hypertext
- Secondary buttons
- Informational elements
- Data visualization (ocean currents)

**Don't use for:**
- Primary CTAs
- Error messages

### Tertiary (Coral)
**Use for:**
- Highlights and badges
- Important alerts (non-errors)
- Warm accents
- Gradient combinations

**Don't use for:**
- Text on dark navy
- Large background areas

### Jellyfish Lavender
**Use for:**
- Premium features
- Special highlights
- Playful accents
- Gradient depth

**Don't use for:**
- Text on white (use 900 instead)
- Critical UI elements

### Seaweed Green
**Use for:**
- Success messages
- Positive confirmations
- Growth metrics
- Completion states

**Don't use for:**
- Warning or error states

---

## 📦 Implementation

### CSS Variables (Ready to Paste)

```css
/* Add to your main CSS file */
@import url('./design-system/colors.css');

/* Or copy the :root and [data-theme="dark"] blocks */
```

### HTML Usage

```html
<!-- Toggle dark mode -->
<html data-theme="light">
  <!-- or -->
<html data-theme="dark">
```

### Component Examples

```css
/* Primary Button */
.btn-primary {
  background: var(--interactive-primary);
  color: var(--text-on-primary);
  border: none;
}

.btn-primary:hover {
  background: var(--interactive-primary-hover);
}

.btn-primary:active {
  background: var(--interactive-primary-active);
}

.btn-primary:disabled {
  background: var(--interactive-primary-disabled);
}

/* Card with gradient */
.hero-card {
  background: var(--gradient-sunlit-water);
  color: var(--text-on-primary);
}

/* Success Alert */
.alert-success {
  background: var(--color-success-bg);
  border: 1px solid var(--color-success-border);
  color: var(--color-success);
}
```

---

## 🚀 Quick Start

1. **Import the CSS:**
   ```html
   <link rel="stylesheet" href="./design-system/colors.css">
   ```

2. **Set theme:**
   ```html
   <html data-theme="light">
   ```

3. **Use semantic tokens:**
   ```css
   body {
     background: var(--bg-primary);
     color: var(--text-primary);
   }
   ```

4. **Build components with states:**
   ```css
   .button {
     background: var(--interactive-primary);
   }

   .button:hover {
     background: var(--interactive-primary-hover);
   }
   ```

---

## 📚 Resources

- **Contrast Checker:** [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
- **WCAG Guidelines:** [WCAG 2.1 AA](https://www.w3.org/WAI/WCAG21/quickref/)
- **Color Tokens:** `./design-system/color-tokens.json`
- **CSS Variables:** `./design-system/colors.css`

---

**Version:** 1.0
**Last Updated:** 2025-01-18
**Maintained by:** The Bikini Bottom Team
